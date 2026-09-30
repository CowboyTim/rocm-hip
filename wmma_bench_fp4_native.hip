#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

// Scalar fallback for the native FP4 WMMA intrinsic __builtin_amdgcn_wmma_f32_32x16x128_f4,
// which REQUIRES gfx1250+ (RDNA4 / AMD AI engines) and is unavailable on gfx1151. On gfx1151
// this path dequantizes nibbles through the Codebook10 LUT and accumulates in scalar FP32.
//
// Tile: 32 rows x 16 cols per block. With 32 threads each thread owns one full output row
// (all 16 columns); partials accumulate in registers across K strips (strip = 128 values)
// and are written once at the end. No shared memory needed.
extern "C" __global__ void wmma_bench_fp4_native(
    const unsigned char* A,       // M x K FP4, row-major (1 nibble per byte)
    const unsigned char* B,       // K x N FP4, row-major (1 nibble per byte)
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    const int lane = threadIdx.x;            // 0..31

    // Each block: one 32x16 output tile; thread owns one row
    int baseRow = blockIdx.x * 32;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    int realRow = baseRow + lane;            // this thread's output row
    bool rowOK = realRow < M;

    float acc[16];

    #pragma unroll
    for (int j = 0; j < 16; j++) acc[j] = 0.0f;

    // Loop over K strips (each strip = 128 FP4 elements)
    const int stripSize = 128;
    const int numStrips = (K + stripSize - 1) / stripSize;

    if (rowOK) {
        for (int s = 0; s < numStrips; s++) {
            int kStart = s * stripSize;
            int kEnd = min(kStart + stripSize, K);

            for (int k = kStart; k < kEnd; k++) {
                uint8_t a_nibble = A[realRow * K + k];
                float av = dequant_code(a_nibble) * scale;

                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    int col = baseCol + j;
                    if (col < N) {
                        acc[j] += av * dequant_code(B[k * N + col]);
                    }
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < 16; j++) {
            int col = baseCol + j;
            if (col < N) C[realRow * N + col] = acc[j];
        }
    }
}
