#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

// FP4 -> FP16 WMMA bench kernel: dequantize nibbles through a 32-byte LDS LUT, then feed the
// FP16 WMMA tensor core (__builtin_amdgcn_wmma_f16_16x16x16_f16_w32). This is the best-known
// gfx1151 path for FP4 matmul (see PLAN.md / FP4_FP6_FP8_Analysis.md).
//
// Bench data contract: A = M x K nibbles row-major; B = K x N nibbles row-major (B[k][j] is
// column j); C = M x N FP16; one global float scale. No UE4M3 per-block scales here -- the
// single scale is baked into the A-side LUT so C = sum_k (a*scale)*b, matching the CPU ref.
//
// Fragment conventions match wmma_kernel.hip / fp4_matmul_wmma.hip (CPU- and hardware-verified):
// lane L in 0..15 supplies row L of A / column L of B; OPSEL=false results are read back by the
// 32-thread pairs owning rows {2e + lIdx/16} at column lIdx%16. c_frag is flushed into an FP32
// shared accumulator after every strip so FP16 rounding stays bounded by one strip's magnitude.
extern "C" __global__ void wmma_bench_fp4_lut(
    const unsigned char* A,       // M x K FP4, row-major (1 nibble per byte)
    const unsigned char* B,       // K x N FP4, row-major (1 nibble per byte)
    __half* C,                    // M x N FP16, row-major
    int M, int N, int K,
    float scale)
{
    const int lIdx = threadIdx.x;             // 0..31
    const int lane = lIdx % 16;               // unique row/column index within the tile

    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    // LDS LUTs: entry[nibble] = Codebook10 value as FP16. A-side bakes in the global scale.
    __shared__ half lds_lut_a[16];   // code * scale
    __shared__ half lds_lut_b[16];   // code

    for (int i = lIdx; i < 16; i += 32) {
        float code = dequant_code(i);
        lds_lut_a[i] = __half(code * scale);
        lds_lut_b[i] = __half(code);
    }
    __syncthreads();

    // FP32 tile accumulator: c_frag is flushed here every strip so FP16 rounding stays bounded.
    __shared__ float accum[16][16];
    for (int i = lIdx; i < 256; i += 32) accum[i / 16][i % 16] = 0.0f;
    __syncthreads();

    int realRow = baseRow + lane;
    bool rowOK  = realRow < M;
    bool colOK  = (baseCol + lane) < N;

    const unsigned char* Arow = A + (size_t)realRow * K;          // only dereferenced if rowOK
    const unsigned char* Bcol = B + (baseCol + lane);             // only dereferenced if colOK

    half16 c_frag = {};                       // WMMA accumulator fragment persists across strips

    const int numStrips = (K + HALF_BLOCK - 1) / HALF_BLOCK;

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * HALF_BLOCK;

        half16 a_frag, b_frag;

        #pragma unroll
        for (int ele = 0; ele < HALF_BLOCK; ele++) {
            int k = kStart + ele;
            uint8_t na = rowOK && k < K ? Arow[k] : 0x00;         // zero-padded partial strip
            uint8_t nb = colOK && k < K ? Bcol[(size_t)k * N] : 0x00;

            a_frag[ele] = lds_lut_a[na & 0x0F];
            b_frag[ele] = lds_lut_b[nb & 0x0F];
        }

        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

        // Flush this strip's contribution into FP32 accumulation and reset the fragment.
        #pragma unroll
        for (int ele = 0; ele < 8; ele++) {
            accum[ele * 2 + (lIdx / 16)][lane] += (_Float16)c_frag[ele * 2];
        }
        c_frag = {};
        __syncthreads();   // flush is complete before the next strip touches accum
    }

    // Store final results: thread t owns cells {row = 2*ele + (t/16), col = t%16}.
    #pragma unroll
    for (int ele = 0; ele < 8; ele++) {
        int r = ele * 2 + (lIdx / 16);
        int tr = baseRow + r;
        int tc = baseCol + lane;
        if (tr < M && tc < N) C[tr * N + tc] = (__half)accum[r][lane];
    }
}
