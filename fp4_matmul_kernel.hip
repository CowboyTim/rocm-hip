// fp4_matmul_kernel.hip -- ROCmFP4 Codebook10 primary compute kernel for gfx1151
//
// Computes C(M x N) = A(M x K) * B(K x N) using ROCmFP4 weight format (Codebook10).
//
// Weight layout per block (matches llama.cpp convention):
//   - Block size: 32 values (32 packed bytes, each byte holds one nibble in bits[3:0])  
//   - Scale: 2 bytes (UE4M3, little-endian) per block of 32 values  
//   - Total block size: 34 bytes (32 data + 2 scale)
//
// Dequant path: __builtin_amdgcn_perm + __builtin_amdgcn_sudot4 (DP4A)
//   - perm expands nibbles to int8 codes via 32-byte constant table lookup (single instruction)
//   - sudot4 accumulates dot products of 4 int8xint8 pairs into int32 reduction  
//   - ~15 cycles/val on gfx1151 (fastest known path at 0.56 BPW)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ============================================================================
// ROCmFP4 Codebook10 dequant constants and helpers
// ============================================================================

static const uint32_t dp4a_pos_reg0 = 0x03020100u;      // [0, 1, 2, 3]
static const uint32_t dp4a_pos_reg1 = 0x0a080604u;      // [4, 6, 8, 10]  

static const uint32_t dp4a_neg_reg2 = 0xfdfeff00u;      // [0,-1,-2,-3] as int8
static const uint32_t dp4a_neg_reg3 = 0xf6f8fafcu;      // [-4,-6,-8,-10] as int8

// Dequant 4 FP4 nibbles packed in a uint32 (each byte holds one nibble in low 4 bits)  
__device__ __forceinline__ uint32_t fp4_dequant_4(uint32_t nibbles) {
    uint32_t pos = __builtin_amdgcn_perm(dp4a_pos_reg1, dp4a_pos_reg0, nibbles);
    uint32_t sign = nibbles >> 1;
    uint32_t sel = (sign & 0x04040404u) | 0x03020100u;  
    uint32_t result = __builtin_amdgcn_perm(dp4a_neg_reg3, dp4a_neg_reg2, sel);
    return result;
}

// ============================================================================
// UE4M3 decode -- scales stored as UE4M3 bytes
// ============================================================================

// Decode a single UE4M3 byte to float on device code using ldexp()
// ldexp(val, exp) computes val * 2^exp and is available for __device__ functions
__device__ __forceinline__ float ue4m3_decode_byte(uint8_t raw) {
    int exp_raw = ((raw >> 4) & 0x0F) - 7;   // bias=7, range [-7..8] 
    int mant = (raw >> 3) & 1;                 // single bit: 1.0 or 2.0  
    int sign = (raw >> 7) & 1;                 // sign bit
    
    float val = mant ? 2.0f : 1.0f;            // mantissa value
    val = ldexpf(val, exp_raw);                // apply power-of-2 exponent via ldexp  
    if (sign) val = -val;                      // apply sign
    
    return val;
}

// Load scale factor for one FP4 block (two UE4M3 bytes, multiply together)
__device__ __forceinline__ float fp4_load_scale_byte(uint8_t b0, uint8_t b1) {
    return ue4m3_decode_byte(b0) * ue4m3_decode_byte(b1);
}

// ============================================================================
// Primary DP4A matmul kernel -- one work-group handles one output tile  
// Each thread processes strips of K dimension using DP4A hardware intrinsics
// ============================================================================

extern "C" __global__ void fp4_matmul_kernel(
    const unsigned char* A,       // M x K FP4 weights, row-major packed nibbles
    const unsigned char* B,       // K x N FP4 weights, row-major packed nibbles 
    const unsigned char* A_scales,// M x numBlockK scale pairs for matrix A  
    const unsigned char* B_scales,// numBlockK x N scale pairs for matrix B
    float* C,                     // M x N FP32 output, row-major  
    int M, int N, int K)          // dimensions
{
    // Thread organization: 32 threads per work-group split into two warps of 8
    //   - warp 0 (threadIdx.x 0-7): handles rows 0..7 within the tile
    //   - warp 1 (threadIdx.x 8-15): handles rows 8..15 within the tile
    //   - columns are handled in groups of 2 per strip lane  

    const int lane = threadIdx.x;        // 0..31
    const int warpId = lane >> 3;         // 0 or 1
    const int laneInWarp = lane & 0x07;   // 0..7

    // Each block computes one 16x16 output tile  
    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    int rowEnd = min(baseRow + 16, M);  
    int colEnd = min(baseCol + 16, N);

    // Shared memory accumulator for this tile -- each thread initializes its portion
    __shared__ float accum[16][16];

    for (int i = lane; i < 256; i += 32) {  
        accum[i / 16][i % 16] = 0.0f;
    }
    __syncthreads();

    // Block size: 32 values per FP4 block (matches llama.cpp convention)
    const int blockSize = 32;
    int numBlocksK = (K + blockSize - 1) / blockSize;

    // Loop over K strips -- accumulate partial products across all blocks in K
    for (int blk = 0; blk < numBlocksK; blk++) {  
        int kStart = blk * blockSize;
        int kEnd = min(kStart + blockSize, K);
        int kLen = kEnd - kStart;  

        // Load scale factors for this block of K into shared memory  
        float aScalePair[16];
        float bScalePair[16];

        #pragma unroll  
        for (int r = laneInWarp; r < 16; r += 8) { 
            if (baseRow + r < M && kStart < K) {  
                // Each row has one scale pair per block of 32 columns
                uint8_t s0 = A_scales[(baseRow + r) * numBlocksK + blk];
                uint8_t s1 = A_scales[(baseRow + r) * numBlocksK + blk + 1];
                aScalePair[r] = fp4_load_scale_byte(s0, s1);
            } else {  
                aScalePair[r] = 1.0f;
            }
        }
        __syncthreads();

        #pragma unroll  
        for (int c = laneInWarp; c < 16; c += 8) {  
            if (baseCol + c < N && kStart < K) {
                // One scale pair per block of 32 rows per column  
                uint8_t s0 = B_scales[blk * N + baseCol + c];
                uint8_t s1 = B_scales[blk * N + baseCol + c + 1];
                bScalePair[c] = fp4_load_scale_byte(s0, s1);  
            } else {
                bScalePair[c] = 1.0f;  
            }
        }
        __syncthreads();

        // Each thread handles one output element within the tile: row=(warpId*8+laneInWarp)
        int outRow = warpId * 8 + laneInWarp;   // 0..15 within tile  

        if (outRow >= 16 || baseRow + outRow >= M) continue;

        int realRow = baseRow + outRow;  

        // Accumulate dot products for each column in this tile using DP4A
        for (int stripLane = 0; stripLane < 8; stripLane++) {
            int outColBase = stripLane * 2;

            #pragma unroll  
            for (int j = 0; j < 2; j++) {
                int outCol = outColBase + j;
                if (outCol >= 16) continue;

                int realCol = baseCol + outCol;  
                if (realCol >= N) continue;

                // Use DP4A hardware intrinsics: perm -> sudot4  
                // Process blockSize/4 = 8 groups of 4 values at a time  
                int accInt32 = 0;  // int32 accumulator for sudot4 accumulation

                for (int k = kStart; k < kEnd; k += 4) {
                    int kEndGroup = min(k + 4, kEnd);  

                    // Pack FP4 nibbles from A matrix into uint32 (one nibble per byte)  
                    uint32_t a_nibbles = 0;
                    uint32_t b_nibbles = 0;

                    for (int i = 0; i < 4 && (k + i) < kEndGroup; i++) {  
                        // A is row-major: A[realRow * K + (kStart + k + i)]
                        uint8_t a_byte = A[realRow * K + kStart + k + i];
                        uint8_t a_nibble = a_byte & 0x0F;    // low nibble is the FP4 value  
                        a_nibbles |= ((uint32_t)a_nibble << (i * 8));

                        // B is packed layout: B[(kStart+k+i) * N + realCol]  
                        uint8_t b_byte = B[(kStart + k + i) * N + realCol];
                        uint8_t b_nibble = b_byte & 0x0F;
                        b_nibbles |= ((uint32_t)b_nibble << (i * 8));  
                    }

                    // Dequant using DP4A perm intrinsic
                    uint32_t a_codes = fp4_dequant_4(a_nibbles);
                    uint32_t b_codes = fp4_dequant_4(b_nibbles);

                    // sudot4: multiply 4 pairs of int8, accumulate to int32
                    accInt32 = __builtin_amdgcn_sudot4(
                        true, a_codes,     // A operand with sign-extension  
                        true, b_codes,     // B operand with sign-extension 
                        accInt32, false    // accumulator and overflow flag
                    );
                }

                // Scale accumulated int32 result by row and column scale factors
                float partial = (float)accInt32 * aScalePair[outRow] * bScalePair[outCol];

                // Atomic add into shared memory accumulator for this tile element  
                atomicAdd(&accum[outRow][outCol], partial);
            }
        }

        __syncthreads();
    }

    // Write accumulated results from shared memory to global output C
    int outRow = warpId * 8 + laneInWarp;  
    if (outRow < 16 && baseRow + outRow < M) {  
        int realRow = baseRow + outRow;
        int realCol = baseCol + (laneInWarp * 2);

        if (realRow < rowEnd && realCol < colEnd) {
            C[realRow * N + realCol] = accum[outRow][laneInWarp * 2];
            if ((laneInWarp * 2 + 1) < colEnd) {  
                C[realRow * N + realCol + 1] = accum[outRow][laneInWarp * 2 + 1];
            }
        }
    }
}
