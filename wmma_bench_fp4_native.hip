#include <hip/hip_runtime.h>

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
static const int8_t fp4_codebook_int8[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

// FP4 matmul kernel using codebook dequant + FP32 accumulate
// A[MxK] FP4 x B[KxN] FP4 -> C[MxN] FP32
// Uses same codebook dequant as DP4A kernel for consistent results
// Each block computes one 32x16 output tile, accumulates across K strips

extern "C" __global__ void wmma_bench_fp4_native(
    const unsigned char* A,       // M x K FP4, row-major (1 nibble per byte)
    const unsigned char* B,       // K x N FP4, row-major (1 nibble per byte)
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    const int lane = threadIdx.x;  // 0..31
    const int warpId = lane / 8;   // 0..3
    const int laneInWarp = lane % 8;
    
    // Each block: one 32x16 output tile
    int baseRow = blockIdx.x * 32;
    int baseCol = blockIdx.y * 16;
    
    if (baseRow >= M || baseCol >= N) return;
    
    int row_end = min(baseRow + 32, M);
    int col_end = min(baseCol + 16, N);
    
    // Shared memory accumulator across K strips
    __shared__ float accum[32][16];
    
    // Initialize accumulator to zero
    for (int i = lane; i < 32*16; i += 32) {
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
        
        // Each thread handles one (row, col) pair for this K-strip
        int outRow = laneInWarp;
        int outCol = warpId * 8;
        
        if (outRow < 32 && outCol < 16) {
            int realRow = baseRow + outRow;
            int realCol = baseCol + outCol;
            
            if (realRow < M && realCol < N) {
                float partial = 0.0f;
                for (int k = 0; k < kLen; k++) {
                    uint8_t a_nibble = A[realRow * K + kStart + k];
                    uint8_t b_nibble = B[(kStart + k) * N + realCol];
                    int8_t a_code = fp4_codebook_int8[a_nibble];
                    int8_t b_code = fp4_codebook_int8[b_nibble];
                    partial += (float)a_code * (float)b_code * scale;
                }
                atomicAdd(&accum[outRow][outCol], partial);
            }
        }
        __syncthreads();
    }
    
    // Write accumulated results
    int outRow = laneInWarp;
    int outCol = warpId * 8;
    
    if (outRow < 32 && outCol < 16) {
        int realRow = baseRow + outRow;
        int realCol = baseCol + outCol;
        if (realRow < row_end && realCol < col_end) {
            C[realRow * N + realCol] = accum[outRow][outCol];
        }
    }
}
