// fp4_matmul_wmma.hip -- Bridges FP4 dequant path into WMMA tensor cores on gfx1151
//
// This kernel loads ROCmFP4 Codebook10 weights from GPU-accessible memory, 
// dequantizes inline via DP4A perm + LDS LUT (nibble -> FP16 lookup), then feeds 
// into FP16 WMMA tensor core (__builtin_amdgcn_wmma_f16_16x16x16_f16_w32).
//
// The LDS LUT approach is used here because it provides good throughput for the 
// nibble-to-FP16 conversion while keeping the dequant overhead low via LDS caching.
// The LUT contains all 32-byte entries mapping nibble indices to FP16 values 
// according to Codebook10's E2M1-derived format: {0, ±1, ±2, ±3, ±4, ±6, ±8, ±10}.
//
// Key difference from dp4a_hardware.hip: this kernel uses WMMA tensor cores for 
// the matrix multiply accumulation, leveraging hardware acceleration while keeping 
// dequant in a separate stage that feeds FP16 values directly into the WMMA intrinsic.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Wave Matrix Multiply Accumulate (WMMA) intrinsics for gfx1151
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

// ============================================================================
// ROCmFP4 Codebook10 FP16 LUT constants  
// ============================================================================

static const _Float16 fp4_to_fp16_lut[32] = {
    (_Float16)0.0f,       // 0x0 -> 0
    (_Float16)1.0f,       // 0x1 -> +1  
    (_Float16)2.0f,       // 0x2 -> +2  
    (_Float16)3.0f,       // 0x3 -> +3
    (_Float16)4.0f,       // 0x4 -> +4
    (_Float16)6.0f,       // 0x5 -> +6  
    (_Float16)8.0f,       // 0x6 -> +8
    (_Float16)10.0f,      // 0x7 -> +10
    (-(_Float16)1.0f),    // 0x8 -> -1
    (-(_Float16)2.0f),    // 0x9 -> -2
    (-(_Float16)3.0f),    // 0xA -> -3  
    (-(_Float16)4.0f),    // 0xB -> -4
    (-(_Float16)6.0f),    // 0xC -> -6
    (-(_Float16)8.0f),    // 0xD -> -8  
    (-(_Float16)10.0f),   // 0xE -> -10
    (_Float16)0.0f        // 0xF -> invalid/reserved (treat as 0)
};

// Dequant FP4 nibble to FP16 via LUT lookup  
__device__ __forceinline__ _Float16 fp4_to_fp16_lut_lookup(uint8_t nibble) {
    return fp4_to_fp16_lut[nibble & 0x0F];
}

// Decode UE4M3 byte to float on device using ldexpf() (available in __device__)
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
// WMMA Bridge Kernel -- FP4 weights dequantized via LDS LUT, then fed into 
// FP16 WMMA tensor core for the matrix multiply accumulation
// ============================================================================

extern "C" __global__ void fp4_matmul_wmma(
    const unsigned char* A,       // M x K FP4 weights, row-major packed nibbles
    const unsigned char* B,       // K x N FP4 weights, row-major packed nibbles
    const unsigned char* A_scales,// M x numBlockK scale pairs for matrix A
    const unsigned char* B_scales,// numBlockK x N scale pairs for matrix B  
    float* C,                     // M x N FP32 output, row-major
    int M, int N, int K)          // dimensions
{
    // Thread organization: 32 threads per work-group handling one 16x16 tile
    //   - All 32 lanes participate in WMMA (warp-level parallelism)  
    //   - Each lane loads a column of A and row of B for one WMMA iteration
    
    const int lIdx = threadIdx.x;  // 0..31
    const int lane = lIdx % 16;     // 0..15 -- unique computation per lane

    // Each block computes one 16x16 output tile
    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    int rowEnd = min(baseRow + 16, M);
    int colEnd = min(baseCol + 16, N);

    // Shared memory accumulator across K strips using FP16 (for WMMA compatibility)
    __shared__ _Float16 accumWMMA[16][16];

    // Initialize accumulator to zero
    for (int i = lIdx; i < 256; i += 32) {
        accumWMMA[i / 16][i % 16] = (_Float16)0.0f;
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
        float aScalePair[8];   // scales for rows (one per warp lane group)
        float bScalePair[8];   // scales for cols

        #pragma unroll  
        for (int r = lIdx; r < 8; r += 32) {
            if (baseRow + r < M && kStart < K) {
                uint8_t s0 = A_scales[(baseRow + r) * numBlocksK + blk];
                uint8_t s1 = A_scales[(baseRow + r) * numBlocksK + blk + 1];
                aScalePair[r] = fp4_load_scale_byte(s0, s1);
            } else {
                aScalePair[r] = 1.0f;
            }
        }
        __syncthreads();

        #pragma unroll  
        for (int c = lIdx; c < 8; c += 32) {
            if (baseCol + c < N && kStart < K) {  
                uint8_t s0 = B_scales[blk * N + baseCol + c];  
                uint8_t s1 = B_scales[blk * N + baseCol + c + 1];
                bScalePair[c] = fp4_load_scale_byte(s0, s1);
            } else {
                bScalePair[c] = 1.0f;
            }
        }
        __syncthreads();

        // Load FP16 values via LUT and feed into WMMA tensor core  
        // Each lane handles one row of A and one column of B for one WMMA iteration
        
        half16 a_frag;   // Fragment register: 16 fp16 elements (packed VGPR format)
        half16 b_frag;   // Fragment register: 16 fp16 elements 
        half16 c_frag = {};   // Accumulator fragment initialized to zero

        // Load one row of matrix A through LUT dequantization (16 elements per lane)
        for (int ele = 0; ele < 16 && (kStart + ele) < kEnd; ++ele) {  
            int realK = kStart + ele;
            if (realK >= K) break;

            uint8_t a_byte = A[(baseRow + lane) * K + realK];
            uint8_t a_nibble = a_byte & 0x0F;    // low nibble is the FP4 value

            // Dequant via LDS LUT lookup -> FP16, then scale by UE4M3 factor  
            float scaleA = aScalePair[lane % 8];
            _Float16 valFP16 = fp4_to_fp16_lut_lookup(a_nibble);
            
            // Convert FP16 to scalar, multiply by scale, back to FP16 for WMMA input
            _Float16 scaledVal = (_Float16)(valFP16 * scaleA);
            a_frag[ele] = scaledVal;
        }

        // Load one column of matrix B through LUT dequantization (16 elements per lane)  
        for (int ele = 0; ele < 16 && (kStart + ele) < kEnd; ++ele) {  
            int realK = kStart + ele;
            if (realK >= K) break;

            uint8_t b_byte = B[realK * N + baseCol + lane];
            uint8_t b_nibble = b_byte & 0x0F;    // low nibble is the FP4 value

            // Dequant via LDS LUT lookup -> FP16, then scale by UE4M3 factor  
            float scaleB = bScalePair[lane % 8];
            _Float16 valFP16b = fp4_to_fp16_lut_lookup(b_nibble);
            
            // Scale by UE4M3 factor and store into fragment register
            _Float16 scaledValB = (_Float16)(valFP16b * scaleB);
            b_frag[ele] = scaledValB;
        }

        // WMMA intrinsic: OPSEL=false, compute C += A*B  
        // All 32 lanes participate but only lanes 0-15 produce valid output
        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(
            a_frag, b_frag, c_frag, false   // OPSEL=false, unpacked output
        );

        // Store results from unpacked c_frag to shared memory accumulator
        // OPSEL=false places 16 results at even indices [0,2,4,...,30] of c_frag  
        for (int ele = 0; ele < 8; ++ele) {
            int r = ele * 2 + (lIdx / 16);
            accumWMMA[r][lane] = c_frag[ele * 2];
        }

        __syncthreads();
    }

    // Write accumulated results from shared memory to global output C  
    // Convert FP16 back to FP32 for final result storage
    if (baseRow + lane < M && baseCol + lane < N) {  
        int realRow = baseRow + lane;
        int realCol = baseCol + lane;

        float sumFP32 = 0.0f;

        #pragma unroll  
        for (int i = 0; i < 8; i += 2) {
            _Float16 valFP16 = accumWMMA[lane % 16][i / 2];
            sumFP32 += (_Float16)valFP16;
        }

        C[realRow * N + realCol] = sumFP32;
    }
}
