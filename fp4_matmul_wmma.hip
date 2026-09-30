// fp4_matmul_wmma.hip -- Bridges FP4 dequant path into WMMA tensor cores on gfx1151
//
// Loads ROCmFP4 Codebook10 weights from GPU-visible memory, dequantizes nibbles to FP16 via a
// register LUT (dequant_code), scales by per-16-element UE4M3 factors, then feeds the FP16 WMMA
// tensor core (__builtin_amdgcn_wmma_f16_16x16x16_f16_w32). c_frag is flushed into an FP32 shared accumulator after every strip (bounded rounding).
//
// Data contract (fp4_common.h): A = M packed lines of K values; B = N packed lines of K values
// (line j = column j of B); UE4M3 scale byte per 16-element half-block.
//
// Fragment conventions match wmma_kernel.hip (CPU-verified there): lane L in 0..15 supplies row L
// of A / column L of B; OPSEL=false results are read back by pairs of threads (rows even/odd).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void fp4_matmul_wmma(
    const unsigned char* A,        // M packed lines, K values each (row i of A)
    const unsigned char* B,        // N packed lines, K values each (line j = column j of B)
    const unsigned char* A_scales, // M x (K/16) UE4M3 bytes
    const unsigned char* B_scales, // N x (K/16) UE4M3 bytes
    float* C,                      // M x N FP32 output, row-major
    int M, int N, int K)          // dimensions; K assumed multiple of 32
{
    const int lIdx = threadIdx.x;             // 0..31
    const int lane = lIdx % 16;               // unique row/column per lane pair

    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    // FP32 tile accumulator: c_frag is flushed here every strip so FP16 rounding stays bounded
    // by one strip's magnitude instead of growing over the whole K dimension.
    __shared__ float accum[16][16];
    for (int i = lIdx; i < 256; i += 32) accum[i / 16][i % 16] = 0.0f;
    __syncthreads();

    int realRow = baseRow + lane;
    bool rowOK  = realRow < M;
    bool colOK  = (baseCol + lane) < N;

    const unsigned char* Arow = A + (size_t)realRow * (K / NIBBLES_BYTE);   // only dereferenced if rowOK
    const unsigned char* Bcol = B + (size_t)(baseCol + lane) * (K / NIBBLES_BYTE);

    int numStrips = K / HALF_BLOCK;           // one 16-wide strip per scale byte

    half16 c_frag = {};                       // WMMA accumulator fragment persists across strips

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * HALF_BLOCK;

        float aScale = rowOK ? ue4m3_to_float(A_scales[(size_t)realRow * numStrips + s]) : 0.0f;
        float bScale = colOK ? ue4m3_to_float(B_scales[(size_t)(baseCol + lane) * numStrips + s]) : 0.0f;

        half16 a_frag, b_frag;

        #pragma unroll
        for (int ele = 0; ele < HALF_BLOCK; ele += 2) {
            int k = kStart + ele;                          // even
            uint8_t ax = rowOK ? *(Arow + (k >> 1)) : 0x00;   // values at k, k+1
            uint8_t bx = colOK ? *(Bcol + (k >> 1)) : 0x00;

            int nlo_a = ax & 0x0F, nhi_a = ax >> 4;
            int nlo_b = bx & 0x0F, nhi_b = bx >> 4;

            _Float16 v[4];
            v[0] = (_Float16)(dequant_code(nlo_a) * aScale);
            v[1] = (_Float16)(dequant_code(nhi_a) * aScale);
            v[2] = (_Float16)(dequant_code(nlo_b) * bScale);
            v[3] = (_Float16)(dequant_code(nhi_b) * bScale);

            a_frag[ele]     = v[0];
            a_frag[ele + 1] = v[1];
            b_frag[ele]     = v[2];
            b_frag[ele + 1] = v[3];
        }

        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

        // Flush this strip's contribution into FP32 accumulation and reset the fragment.
        #pragma unroll
        for (int ele = 0; ele < 8; ele++) {
            accum[ele * 2 + (lIdx / 16)][lane] += (_Float16)c_frag[ele * 2];
        }
        c_frag = {};
        __syncthreads();   // flush is complete before the next strip reads/writes accum
    }

    // Store final results: thread t owns cells {row = 2*ele + (t/16), col = t%16}.
    for (int ele = 0; ele < 8; ele++) {
        int r = ele * 2 + (lIdx / 16);
        int tr = baseRow + r;
        int tc = baseCol + lane;
        if (tr < M && tc < N) C[tr * N + tc] = accum[r][lane];
    }
}
