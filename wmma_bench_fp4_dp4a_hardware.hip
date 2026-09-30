#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
//
// DP4A dequant using __builtin_amdgcn_perm -- hardware-verified scheme for this toolchain
// (see PLAN.md "Empirically Verified Perm Semantics"):
//   - perm(X, Y, m): per mask byte, bit2 selects X[m&3] vs Y[m&3]; mask bytes >= 8 are undefined.
//   - Magnitude: one perm over {P_LO=[+0..+3], P_HI=[+4,+6,+8,+10]} with a <=7 mask built from
//     the nibble's low 3 bits (bit2 = magnitude half-select, low 2 bits = index within table).
//   - Sign: sign bit (nibble bit3) shifted into bit2 drives a second perm expanding an FF/00 lane mask.
//   - Signed code: pos ^ (mask & (neg ^ pos)) -- pure bitwise combine, no select builtin needed.

// Dequant 4 FP4 nibbles packed in a uint32 (each byte holds one nibble in bits [3:0], upper zeroed)
// Returns uint32 where each byte holds the dequantized int8 Codebook10 value.
__device__ __forceinline__ uint32_t dp4a_dequant_4(uint32_t nib) {
    uint32_t msel  = nib & 0x07070707u;          // magnitude index bits (bit3 clear by construction)
    uint32_t signm = (nib >> 1) & 0x0C0C0C0Cu;   // sign bit moved into bit2 position
    uint32_t sFF   = __builtin_amdgcn_perm(0xffffffffu, 0x00000000u, signm); // FF lanes where negative
    uint32_t pos   = __builtin_amdgcn_perm(CB_P_HI, CB_P_LO, msel);           // +magnitude codes
    uint32_t negv  = __builtin_amdgcn_perm(CB_N_HI, CB_N_LO, msel);           // -magnitude codes
    return pos ^ (sFF & (negv ^ pos));
}

extern "C" __global__ void wmma_bench_fp4_dp4a_hardware(
    const unsigned char* A,       // M x K FP4, row-major
    const unsigned char* B,       // K x N FP4, row-major
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    const int lane = threadIdx.x;          // 0..31
    const int rInTile = lane >> 1;         // 0..15 output row within tile (full coverage)
    const int cBase   = (lane & 1) * 8;    // first column for this thread (owns 8 columns)

    // Each block: one 16x16 output tile
    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    // Shared memory accumulator across K strips
    __shared__ float accum[16][16];

    // Initialize accumulator to zero
    for (int i = lane; i < 16*16; i += 32) {
        accum[i / 16][i % 16] = 0.0f;
    }
    __syncthreads();

    // Loop over K strips (each strip = 16 FP4 elements)
    const int stripSize = 16;
    const int numStrips = (K + stripSize - 1) / stripSize;

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * stripSize;
        int kEnd = min(kStart + stripSize, K);
        int kLen = kEnd - kStart;

        int realRow = baseRow + rInTile;

        if (realRow < M) {
            // Load + dequant the A row for this strip ONCE; reused across all columns.
            uint32_t a_codes_strip[HALF_BLOCK / 4];   // one dequanted group per 4 values
            for (int g = 0; g < HALF_BLOCK / 4; g++) {
                int kg = kStart + g * 4;
                int kEndGroup = min(kg + 4, K);
                uint32_t a_nibbles = 0;
                for (int i = 0; i < 4 && (kg + i) < kEndGroup; i++) {
                    uint8_t a_byte = A[realRow * K + kg + i];
                    a_nibbles |= ((uint32_t)(a_byte & 0x0F) << (i * 8));
                }
                a_codes_strip[g] = dp4a_dequant_4(a_nibbles);   // verified DP4A perm chain
            }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int realCol = baseCol + cBase + j;

                if (realCol >= N) continue;

                // Per column: load B, dequant, dot against the pre-dequanted A row.
                for (int k = 0; k < kLen; k += 4) {
                    int g = k / 4;

                    // Load B values (column-dependent) and dequant
                    uint32_t b_nibbles = 0;
                    for (int i = 0; i < 4 && (kStart + k + i) < K; i++) {
                        uint8_t b_byte = B[(kStart + k + i) * N + realCol];
                        b_nibbles |= ((uint32_t)(b_byte & 0x0F) << (i * 8));
                    }
                    uint32_t b_codes = dp4a_dequant_4(b_nibbles);

                    // sudot4: multiply 4 pairs of int8 and accumulate to int32
                    int acc = __builtin_amdgcn_sudot4(true, a_codes_strip[g], true, b_codes, 0, false);

                    // Scale from int32 to FP32
                    float partial = (float)acc * scale;
                    atomicAdd(&accum[rInTile][cBase + j], partial);
                }
            }
        }
        __syncthreads();   // unconditional: every thread reaches it each strip
    }

    // Write accumulated results: thread owns cells {row rInTile, cols cBase..cBase+7}
    int realRow = baseRow + rInTile;
    if (realRow < M) {
        for (int j = 0; j < 8; j++) {
            int realCol = baseCol + cBase + j;
            if (realCol < N) C[realRow * N + realCol] = accum[rInTile][cBase + j];
        }
    }
}
