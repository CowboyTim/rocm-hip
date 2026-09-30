// fp4_matmul_kernel.hip -- ROCmFP4 Codebook10 primary compute kernel for gfx1151
//
// Computes C(M x N) = A(M x K) * B(K x N). Both operands use the canonical packed layout:
//   - Data: one byte per two values, low nibble first.
//   - Lines are packed along the reduction (K) axis so every dot product reads contiguous data:
//       A: M lines of K values  -> A + i*(K/2) holds row i of A
//       B: N lines of K values  -> B + j*(K/2) holds column j of B (transposed storage)
//   - Scales: UE4M3 byte per 16-element half-block; A_scales[i][s], B_scales[j][s].
//
// Dequant path (verified on gfx1151 hardware):
//   - nibbles -> int8 codes via __builtin_amdgcn_perm table lookups + bitwise sign combine
//   - dot products via __builtin_amdgcn_sudot4 (int8xint8 -> int32)
//   - scales applied as FP32 multiply after each 16-wide strip
//
// NOTE on __builtin_amdgcn_perm semantics (empirically verified with probe kernels):
//   perm(X, Y, m): per mask byte, bit2 selects X[m&3] vs Y[m&3]; mask bytes >= 8 are
//   undefined behavior. All masks below are kept <= 7 by construction.

#include <hip/hip_runtime.h>
#include "fp4_common.h"

using namespace rocmfp4;

// Dequant 4 FP4 nibbles (one per uint32 byte, bits [3:0] valid, upper bits zeroed) to int8 codes.
__device__ __forceinline__ uint32_t fp4_dequant_4(uint32_t nib) {
    uint32_t msel  = nib & 0x07070707u;          // magnitude index bits (bit3 clear by mask)
    uint32_t signm = (nib >> 1) & 0x0C0C0C0Cu;   // sign bit moved down to bit2 position
    uint32_t sFF   = __builtin_amdgcn_perm(0xffffffffu, 0x00000000u, signm); // 0xFF in negative lanes
    uint32_t pos   = __builtin_amdgcn_perm(CB_P_HI, CB_P_LO, msel);           // +magnitude code
    uint32_t negv  = __builtin_amdgcn_perm(CB_N_HI, CB_N_LO, msel);           // -magnitude code
    return pos ^ (sFF & (negv ^ pos));                              // signed Codebook10 value per lane
}

// Pack two consecutive raw bytes into a dequant-ready uint32: values at k, k+1 (from x) and
// k+2, k+3 (from y), one value per byte. Requires k even.
__device__ __forceinline__ uint32_t pack4(const unsigned char* p_even_k) {
    uint8_t x = *p_even_k;
    uint8_t y = *(p_even_k + 1);
    return ((uint32_t)(x & 0x0F))            |
           (((uint32_t)x >> 4) << 8)         |
           (((uint32_t)(y & 0x0F)) << 16)    |
           (((uint32_t)y >> 4) << 24);
}

extern "C" __global__ void fp4_matmul_kernel(
    const unsigned char* A,        // M packed lines, K values each (row-major along K)
    const unsigned char* B,        // N packed lines, K values each (line j = column j of B)
    const unsigned char* A_scales, // M x (K/16) UE4M3 bytes
    const unsigned char* B_scales, // N x (K/16) UE4M3 bytes
    float* C,                      // M x N FP32 output, row-major
    int M, int N, int K)          // dimensions; K assumed multiple of 32
{
    const int lane = threadIdx.x;               // 0..31
    const int rInTile  = lane >> 1;             // 0..15 -- output row within tile
    const int cBase    = (lane & 1) * 8;        // 0 or 8  -- first output column for this thread

    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    __shared__ float accum[16][16];
    for (int i = lane; i < 256; i += 32) accum[i / 16][i % 16] = 0.0f;
    __syncthreads();

    int realRow = baseRow + rInTile;
    bool rowOK  = realRow < M;

    const unsigned char* Arow = A + (size_t)realRow * (K / NIBBLES_BYTE);   // valid pointer even if !rowOK? guard below
    int numStrips = K / HALF_BLOCK;                                          // 16-wide strips, one scale each

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * HALF_BLOCK;
        float aScale = rowOK ? ue4m3_to_float(A_scales[(size_t)realRow * numStrips + s]) : 0.0f;

        for (int j = 0; j < 8; j++) {
            int cInTile = cBase + j;
            int realCol = baseCol + cInTile;
            if (!rowOK || realCol >= N) continue;

            const unsigned char* Bcol = B + (size_t)realCol * (K / NIBBLES_BYTE);
            float bScale = ue4m3_to_float(B_scales[(size_t)realCol * numStrips + s]);

            int accInt = 0;
            #pragma unroll
            for (int g = 0; g < HALF_BLOCK; g += 4) {
                int k = kStart + g;                       // even (kStart mult of 16)
                uint32_t wA = pack4(Arow + (k >> 1));
                uint32_t wB = pack4(Bcol + (k >> 1));
                uint32_t cA = fp4_dequant_4(wA);
                uint32_t cB = fp4_dequant_4(wB);
                accInt = __builtin_amdgcn_sudot4(true, cA, true, cB, accInt, false);
            }

            float partial = (float)accInt * aScale * bScale;
            atomicAdd(&accum[rInTile][cInTile], partial);
        }
    }

    __syncthreads();

    if (!rowOK) return;
    for (int j = 0; j < 8; j++) {
        int realCol = baseCol + cBase + j;
        if (realCol < N) C[realRow * N + realCol] = accum[rInTile][cBase + j];
    }
}
