// gemm_act_fp4.hip -- Phase 3: activation(FP16) x weight(ROCmFP4 Codebook10) -> FP32 WMMA GEMM.
//
// The production projection path for attention (Q/K/V/O): activations stay FP16 in mapped memory;
// weights stay packed Codebook10 nibbles with UE4M3 per-16-element scales (llama.cpp block layout);
// dequantization happens inline in the kernel at compute time -- no conversion pass, no copies.
//
//   C[m][n] = sum_k A[m][k] * deq(W[n][k])
//   deq(n,k) = codebook[nib(W[n][k])] * ue4m3_to_float(W_scales[n][k/16])
//
// A is [M x K] FP16 row-major; W is N lines of K values packed 2-per-byte (low nibble first);
// W_scales is N x (K/16) bytes; C is [M x N] FP32 row-major. All operands live in pinned pages.
// Tiling reuses the hardware-verified conventions: 16x16 tile / 32 threads, K in 16-wide strips
// (one UE4M3 half-block each), OPSEL=false c_frag flushed to an FP32 shared accumulator per strip.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void gemm_act_fp4(const __half* A, const unsigned char* W,
                                        const unsigned char* W_scales, float* C,
                                        int M, int N, int K /* multiple of 16 */)
{
    const int lIdx = threadIdx.x;             // 0..31
    const int lane = lIdx % 16;               // unique row/column index within the tile

    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    const int aRow = baseRow + lane;          // this thread's A row (duplicated across t/16 halves)
    const int wLine = baseCol + lane;         // this thread's weight line (= output column)

    // LDS LUT: Codebook10 nibble -> FP32 magnitude, so scaled values need only one multiply.
    __shared__ float lds_code[16];
    for (int i = lIdx; i < 16; i += 32) lds_code[i] = dequant_code(i);
    __syncthreads();

    __shared__ float accum[16][16];
    for (int i = lIdx; i < 256; i += 32) accum[i / 16][i % 16] = 0.0f;
    __syncthreads();

    half16 c_frag = {};                       // per-strip WMMA accumulator fragment

    const int numStrips = K / HALF_BLOCK;     // exactly one UE4M3 half-block per strip
    const size_t w_row_bytes = (size_t)K / NIBBLES_BYTE;

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * HALF_BLOCK;

        half16 a_frag, b_frag;

        #pragma unroll
        for (int ele = 0; ele < HALF_BLOCK; ele += 2) {
            int k = kStart + ele;            // values at k and k+1 share one packed weight byte
            bool aOK = (aRow < M);           // rows beyond M get zero fragments (guarded)

            a_frag[ele]     = aOK ? ((const __half*)A)[(size_t)aRow * K + k] : __half(0.0f);
            a_frag[ele + 1] = aOK ? ((const __half*)A)[(size_t)aRow * K + k + 1] : __half(0.0f);

            // Weight line: one byte holds nibbles for k, k+1; scale covers this whole strip.
            // Lines beyond N are zero-padded (guarded so edge tiles never read past the last line).
            bool wOK = (wLine < N);
            float wScale = wOK ? ue4m3_to_float(W_scales[(size_t)wLine * numStrips + s]) : 0.0f;
            uint8_t wb = wOK ? W[(size_t)wLine * w_row_bytes + (k >> 1)] : 0x00u;
            float c_lo = lds_code[wb & 0x0F];
            float c_hi = lds_code[wb >> 4];

            b_frag[ele]     = (__half)(c_lo * wScale);
            b_frag[ele + 1] = (__half)(c_hi * wScale);
        }

        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {   // flush the strip into FP32 accumulation
            accum[ele * 2 + (lIdx / 16)][lane] += (_Float16)c_frag[ele * 2];
        }
        c_frag = {};
        __syncthreads();
    }

    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {       // thread t owns cells {row=2*ele+(t/16), col=t%16}
        int r = ele * 2 + (lIdx / 16);
        int tr = baseRow + r;
        int tc = baseCol + lane;
        if (tr < M && tc < N) C[(size_t)tr * N + tc] = accum[r][lane];
    }
}
