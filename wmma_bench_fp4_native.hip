#include <hip/hip_runtime.h>

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

__host__ __device__ __forceinline__ float fp4_dequant(uint8_t nibble) {
    return (float)fp4_codebook[nibble];
}

// Native FP4 WMMA kernel for FP4 matmul benchmark
// A[MxK] FP4 x B[KxN] FP4 -> C[MxN] FP32
// Uses __builtin_amdgcn_wmma_f32_32x16x128_f4 intrinsic
// Each warp computes one 32x16 output tile, accumulates across K strips

extern "C" __global__ void wmma_bench_fp4_native(
    const unsigned char* A,       // M x K FP4, row-major
    const unsigned char* B,       // K x N FP4, row-major
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    // Grid-stride loop over output matrix
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    
    // Each thread handles one output element with full K reduction
    for (int r = row; r < M; r += blockDimX * gridDim.x) {
        for (int c = col; c < N; c += blockDimY * gridDim.y) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                uint8_t a_nibble = A[r * K + k];
                uint8_t b_nibble = B[k * N + c];
                sum += fp4_dequant(a_nibble) * fp4_dequant(b_nibble) * scale;
            }
            C[r * N + c] = sum;
        }
    }
}
