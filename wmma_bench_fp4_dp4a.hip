#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

// Scalar LUT dequant bench kernel: per-value Codebook10 lookup + scalar float multiply.
// Thread mapping (full tile coverage with 32 threads): thread t owns output row t/2 and
// columns (t&1)*8 .. (t&1)*8+7 within the 16x16 tile. Per-strip partials accumulate in
// registers; shared-memory accumulator receives one add per strip; final write is guarded.
extern "C" __global__ void wmma_bench_fp4_dp4a(
    const unsigned char* A,       // M x K FP4, row-major (1 nibble per byte)
    const unsigned char* B,       // K x N FP4, row-major (1 nibble per byte)
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    const int lane = threadIdx.x;            // 0..31
    const int rInTile = lane >> 1;           // 0..15 output row within tile
    const int cBase   = (lane & 1) * 8;      // first column owned by this thread

    // Each block: one 16x16 output tile
    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    int realRow = baseRow + rInTile;
    bool rowOK = realRow < M;

    // Shared memory accumulator across K strips
    __shared__ float accum[16][16];

    for (int i = lane; i < 16*16; i += 32) {
        accum[i / 16][i % 16] = 0.0f;
    }
    __syncthreads();

    // Loop over K strips (each strip = 16 FP4 elements)
    const int stripSize = 16;
    const int numStrips = (K + stripSize - 1) / stripSize;

    float partial[8];

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * stripSize;
        int kEnd = min(kStart + stripSize, K);

        #pragma unroll
        for (int j = 0; j < 8; j++) partial[j] = 0.0f;

        if (rowOK) {
            for (int k = kStart; k < kEnd; k++) {
                int8_t a_code = CODEBOOK[A[realRow * K + k] & 0x0F];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int col = baseCol + cBase + j;
                    if (col < N) {
                        int8_t b_code = CODEBOOK[B[k * N + col] & 0x0F];
                        partial[j] += (float)a_code * (float)b_code * scale;
                    }
                }
            }
        }

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int col = baseCol + cBase + j;
            if (col < N) accum[rInTile][cBase + j] += partial[j];
        }
        __syncthreads();   // flush complete before the next strip adds to accum
    }

    // Write accumulated results (guarded: every thread writes only cells that exist)
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int col = baseCol + cBase + j;
        if (col < N) C[realRow * N + col] = accum[rInTile][cBase + j];
    }
}
