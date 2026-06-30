#include <hip/hip_runtime.h>

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

__host__ __device__ __forceinline__ float fp4_dequant(uint8_t nibble) {
    return (float)fp4_codebook[nibble];
}

extern "C" __global__ void wmma_matmul_fp4_native(const unsigned char* A,
                                                    const unsigned char* B,
                                                    float* C,
                                                    float scale) {
    const int tid = threadIdx.x;  // 0-31
    
    // FP4 native WMMA: A[32x128] FP4 x B[16x128] FP4 -> C[32x16] FP32
    //
    // Each thread computes one output row (32 threads = 32 output rows)
    // C[tid][col] = sum_{k=0..127} A[tid][k] * B[col][k]
    // where col is 0..15
    
    float result[16] = {0};
    
    // Process K dimension
    for (int k = 0; k < 128; ++k) {
        uint8_t a_val = A[tid * 128 + k];
        float a_f = fp4_dequant(a_val) * scale;
        
        // Accumulate across all 16 output columns
        for (int col = 0; col < 16; ++col) {
            uint8_t b_val = B[col * 128 + k];
            float b_f = fp4_dequant(b_val) * scale;
            result[col] += a_f * b_f;
        }
    }
    
    // Store output
    for (int col = 0; col < 16; ++col) {
        C[tid * 16 + col] = result[col];
    }
}
