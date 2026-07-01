#include <hip/hip_runtime.h>

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
// ROCmFP4 is E2M1 (2 exponent bits, 1 mantissa bit)
// Stored as 4-bit nibbles, dequantized to int8 for DP4A matmul
static const int8_t fp4_codebook_int8[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

// Positive tables for DP4A perm (bit 0 selects between reg0/reg1)
static const uint32_t dp4a_pos_reg0 = 0x03020100u;  // [0, 1, 2, 3]
static const uint32_t dp4a_pos_reg1 = 0x0a080604u;  // [4, 6, 8, 10]

// Negative tables for DP4A perm (bit 1 of sign selects between reg2/reg3)
static const uint32_t dp4a_neg_reg2 = 0xfdfeff00u;  // [0, -1, -2, -3]
static const uint32_t dp4a_neg_reg3 = 0xf6f8fafcu;  // [-4, -6, -8, -10]

// Dequant 4 FP4 nibbles to 4 int8 values using DP4A perm
// Each byte in the uint32 holds one 4-bit nibble in the low 4 bits
__device__ __forceinline__ uint32_t dp4a_dequant_4(uint32_t nibbles) {
    // Step 1: perm positive values based on bit 0 of each nibble
    uint32_t pos = __builtin_amdgcn_perm(dp4a_pos_reg1, dp4a_pos_reg0, nibbles);
    
    // Step 2: extract sign bit (bit 1 of nibble = bit 2 shifted right by 1)
    uint32_t sign = nibbles >> 1;
    
    // Step 3: build selector - bit 1 of nibble selects between pos and neg tables
    uint32_t sel = (sign & 0x02020202u) | 0x03020100u;
    
    // Step 4: perm between positive and negative tables
    uint32_t result = __builtin_amdgcn_perm(dp4a_neg_reg3, dp4a_neg_reg2, sel);
    
    return result;
}

extern "C" __global__ void wmma_bench_fp4_dp4a_int8(
    const unsigned char* A,       // M x K FP4, row-major (1 nibble per byte)
    const unsigned char* B,       // K x N FP4, row-major (1 nibble per byte)
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    const int tid = threadIdx.x;  // 0..31
    const int lane = tid % 16;
    const int warpId = lane / 8;
    const int laneInWarp = lane % 8;
    
    // Each block: one 16x16 output tile
    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;
    
    if (baseRow >= M || baseCol >= N) return;
    
    int row_end = min(baseRow + 16, M);
    int col_end = min(baseCol + 16, N);
    
    // Accumulator for FP32 output
    float accum[16][16] = {0};
    
    // Loop over K dimension in strips
    const int stripSize = 16;
    const int numStrips = (K + stripSize - 1) / stripSize;
    
    for (int s = 0; s < numStrips; s++) {
        int kStart = s * stripSize;
        int kEnd = min(kStart + stripSize, K);
        int kLen = kEnd - kStart;
        
        int outRow = laneInWarp;
        int outCol = warpId * 8;
        
        if (outRow < 16 && outCol < 16) {
            int realRow = baseRow + outRow;
            int realCol = baseCol + outCol;
            
            if (realRow < M && realCol < N) {
                // Process 4 values at a time using DP4A
                for (int k = 0; k < kLen; k += 4) {
                    int kEndGroup = min(k + 4, kLen);
                    
                    // Pack 4 A nibbles and 4 B nibbles into uint32
                    // Each nibble in low 4 bits of each byte
                    uint32_t a_nibbles = 0;
                    uint32_t b_nibbles = 0;
                    for (int i = 0; i < 4 && (k + i) < kEndGroup; i++) {
                        uint8_t a_byte = A[realRow * K + kStart + k + i];
                        uint8_t b_byte = B[(kStart + k + i) * N + realCol];
                        
                        uint8_t a_nibble = a_byte & 0x0F;
                        uint8_t b_nibble = b_byte & 0x0F;
                        
                        a_nibbles |= ((uint32_t)a_nibble << (i * 8));
                        b_nibbles |= ((uint32_t)b_nibble << (i * 8));
                    }
                    
                    // Dequant using DP4A perm (FP4 nibble → int8)
                    uint32_t a_codes = dp4a_dequant_4(a_nibbles);
                    uint32_t b_codes = dp4a_dequant_4(b_nibbles);
                    
                    // DP4A: 4×int8 multiply-accumulate
                    // acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                    // We accumulate to float, but DP4A works on int32
                    int32_t acc_int = 0;
                    acc_int = __builtin_amdgcn_sudot4(true, a_codes, true, b_codes, acc_int, false);
                    
                    // Convert int32 to float and scale
                    float acc_float = (float)acc_int * scale;
                    
                    // Accumulate in FP32
                    accum[outRow][outCol] += acc_float;
                }
            }
        }
    }
    
    // Write results
    int outRow = laneInWarp;
    int outCol = warpId * 8;
    
    if (outRow < 16 && outCol < 16) {
        int realRow = baseRow + outRow;
        int realCol = baseCol + outCol;
        if (realRow < row_end && realCol < col_end) {
            C[realRow * N + realCol] = accum[outRow][outCol];
        }
    }
}
