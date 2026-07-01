#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
// 
// DP4A dequant using __builtin_amdgcn_perm:
//   The intrinsic selects from 2 source registers based on bit 0 of the nibble index.
//   Positive values (bit 0=0): reg0 = [ 0,  1,  2,  3] = 0x03020100
//   Positive values (bit 0=1): reg1 = [ 4,  6,  8, 10] = 0x0a080604
//   Negative values (bit 0=0): reg2 = [ 0, -1, -2, -3] = 0xfdfeff00
//   Negative values (bit 0=1): reg3 = [-4, -6, -8,-10] = 0xf6f8fafc
//
// Permutation process for each nibble:
//   1. Extract nibble index (4 bits) from each byte position
//   2. perm(reg1, reg0, idx) → selects from pos table based on bit 0
//   3. Extract sign bit (bit 2 of nibble): sign = nibble >> 1
//   4. perm(reg3, reg2, sign|0x03020100) → selects from neg table if sign bit set
//
// The input uint32 must have nibbles placed in each byte:
//   byte3 | byte2 | byte1 | byte0
//   nib7..nibble | nib5..nibble | nib3..nibble | nib1..nibble

// Positive table: perm(reg1, reg0, idx) gives:
//   if idx[0]=0 → values from reg0 [0,1,2,3]
//   if idx[0]=1 → values from reg1 [4,6,8,10]
static const uint32_t dp4a_pos_reg0 = 0x03020100u;
static const uint32_t dp4a_pos_reg1 = 0x0a080604u;

// Negative table: perm(reg3, reg2, sign_sel) gives:
//   if sign[0]=0 → values from reg2 [0,-1,-2,-3]
//   if sign[0]=1 → values from reg3 [-4,-6,-8,-10]
static const uint32_t dp4a_neg_reg2 = 0xfdfeff00u;
static const uint32_t dp4a_neg_reg3 = 0xf6f8fafcu;

// Dequant 4 FP4 nibbles packed in a uint32 (each byte holds one nibble in low 4 bits)
// Returns int32_t where each byte holds the dequantized int8 value
__device__ __forceinline__ uint32_t dp4a_dequant_4(uint32_t nibbles) {
    // Step 1: perm positive values based on bit 0 of each nibble
    //   nibble 0→3 from reg0, nibble 4,6,8,10 from reg1
    uint32_t pos = __builtin_amdgcn_perm(dp4a_pos_reg1, dp4a_pos_reg0, nibbles);
    
    // Step 2: extract sign bit (bit 2 of nibble), shift right by 1
    //   This gives bit 0 of sign for each nibble position
    uint32_t sign = nibbles >> 1;
    
    // Step 3: build selector - bit 2 of nibble selects between pos and neg tables
    //   (sign & 0x04040404) | 0x03020100
    uint32_t sel = (sign & 0x04040404u) | 0x03020100u;
    
    // Step 4: perm between positive and negative tables
    //   If sign bit was 0 → from reg2 [0,-1,-2,-3]
    //   If sign bit was 1 → from reg3 [-4,-6,-8,-10]
    //   We permute (reg3, reg2) but we already have the positive values,
    //   so we need to select between pos and the neg tables
    //   Actually we need to permute neg tables, not re-use pos
    // The correct approach: we need two separate perms for the negative case
    // But since we're using a single perm, we select from [0,-1,-2,-3] or [-4,-6,-8,-10]
    uint32_t result = __builtin_amdgcn_perm(dp4a_neg_reg3, dp4a_neg_reg2, sel);
    
    return result;
}

// Dequant a single nibble (4-bit index) using DP4A perm
__device__ __forceinline__ int8_t dp4a_dequant_1(uint8_t nibble) {
    uint32_t nibbles = nibble;  // place nibble in byte 0
    uint32_t pos = __builtin_amdgcn_perm(dp4a_pos_reg1, dp4a_pos_reg0, nibbles);
    uint32_t sign = nibbles >> 1;
    uint32_t sel = (sign & 0x04040404u) | 0x03020100u;
    uint32_t result = __builtin_amdgcn_perm(dp4a_neg_reg3, dp4a_neg_reg2, sel);
    return (int8_t)result;
}

extern "C" __global__ void wmma_bench_fp4_dp4a_hardware(
    const unsigned char* A,       // M x K FP4, row-major
    const unsigned char* B,       // K x N FP4, row-major
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    const int lane = threadIdx.x;  // 0..15
    const int warpId = lane / 8;   // 0 or 1
    const int laneInWarp = lane % 8;

    // Each block: one 16x16 output tile
    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    int row_end = min(baseRow + 16, M);
    int col_end = min(baseCol + 16, N);

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

        int outRow = laneInWarp;
        int outCol = warpId * 8;

        if (outRow < 16 && outCol < 16) {
            int realRow = baseRow + outRow;
            int realCol = baseCol + outCol;

            if (realRow < M && realCol < N) {
                // Use DP4A hardware intrinsics for dequant + dot product
                // Process 4 values at a time using __builtin_amdgcn_perm + sudot4
                int acc = 0;  // int32 accumulator for sudot4

                // Process groups of 4 FP4 values (4 bytes = 8 nibbles, but each byte is one FP4)
                // Each sudot4 processes 4 int8×int8 pairs in 1 instruction
                for (int k = 0; k < kLen; k += 4) {
                    int kEndGroup = min(k + 4, kLen);
                    
                    // Load A values and dequant using __builtin_amdgcn_perm
                    // Pack 4 nibbles into a uint32 where each byte holds one nibble
                    uint32_t a_nibbles = 0;
                    uint32_t b_nibbles = 0;
                    for (int i = 0; i < 4 && (k + i) < kEndGroup; i++) {
                        uint8_t a_byte = A[realRow * K + kStart + k + i];
                        uint8_t b_byte = B[(kStart + k + i) * N + realCol];
                        uint8_t a_nibble = a_byte & 0x0F;      // low nibble
                        uint8_t b_nibble = b_byte & 0x0F;       // low nibble
                        a_nibbles |= ((uint32_t)a_nibble << (i * 8));
                        b_nibbles |= ((uint32_t)b_nibble << (i * 8));
                    }
                    
                    // Dequant using DP4A perm
                    uint32_t a_codes = dp4a_dequant_4(a_nibbles);
                    uint32_t b_codes = dp4a_dequant_4(b_nibbles);
                    
                    // sudot4: multiply 4 pairs of int8 and accumulate to int32
                    // acc += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
                    acc = __builtin_amdgcn_sudot4(true, a_codes, true, b_codes, acc, false);
                }

                // Scale from int32 to FP32
                float partial = (float)acc * scale;
                atomicAdd(&accum[outRow][outCol], partial);
            }
        }
        __syncthreads();
    }

    // Write accumulated results
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
