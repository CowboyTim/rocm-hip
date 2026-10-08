// gemm_act_fp4.hip -- Phase 3: activation(FP16) x weight(quantized) -> FP32 WMMA GEMM.
//
// Supports multiple quantization formats:
//   - ROCmFP4/TYPE100/TYPE13: UE4M3 per-16-element scales (llama.cpp block layout)
//   - Q4_0/Q8_0: FP16 delta per-32 values
//   - Q4_1: FP16 scale + min per-32 values
// Dequantization happens inline in the kernel at compute time -- no conversion pass, no copies.
//
// For UE4M3 types:
//   C[m][n] = sum_k A[m][k] * deq(W[n][k])
//   deq(n,k) = codebook[nib(W[n][k])] * ue4m3_to_float(W_scales[n][k/16])
// For non-UE4M3 types (Q4_0/Q4_1/Q8_0):
//   Q4_0/Q8_0: value = codebook[nib] * delta (one FP16 delta per 32 values)
//   Q4_1: value = scale * codebook[nib] + min (scale+min per 32 values)
//
// A is [M x K] FP16 row-major; W is N lines of K values packed 2-per-byte;
// W_scales/W_scales_f provide scales; C is [M x N] FP32 row-major. All operands live in pinned pages.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void gemm_act_fp4(const __half* A, const unsigned char* W,
                                        const unsigned char* W_scales, const float* W_scales_f,
                                        float* C, int M, int N, int K /* multiple of 16 */)
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

    const size_t w_row_bytes = (size_t)K / NIBBLES_BYTE;

    // Determine scale format: W_scales_f != nullptr means non-UE4M3 types (Q4_0/Q4_1/Q8_0)
    bool use_float_scales = (W_scales_f != nullptr);

    if (!use_float_scales) {
        // UE4M3-encoded scales (TYPE100/TYPE13): one byte per 16 elements
        const int numStrips = K / HALF_BLOCK;

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
    } else {
        // Non-UE4M3 types (Q4_0/Q4_1/Q8_0): pre-decoded floats from W_scales_f
        // Layout depends on type:
        //   Q4_0/Q8_0: one decoded FP16 delta per block of 32 values -> K/32 floats per row
        //   Q4_1: two decoded floats per block (scale + min) -> K/16 floats per row
        // For compatibility, we always read using Q4_1 layout (K/16 floats) since it covers all cases.
        // For Q4_0/Q8_0 where only delta exists, scale = delta and min = 0.

        const size_t blocks_per_row = K / 32;

        for (size_t b = 0; b < blocks_per_row; b++) {
            int kStart = (int)b * 32;

            half16 a_frag, b_frag;

            #pragma unroll
            for (int ele = 0; ele < 32; ele += 2) {
                int k = kStart + ele;            // values at k and k+1 share one packed weight byte
                bool aOK = (aRow < M);

                a_frag[ele]     = aOK ? ((const __half*)A)[(size_t)aRow * K + k] : __half(0.0f);
                a_frag[ele + 1] = aOK ? ((const __half*)A)[(size_t)aRow * K + k + 1] : __half(0.0f);

                bool wOK = (wLine < N);
                uint8_t wb = wOK ? W[(size_t)wLine * w_row_bytes + (k >> 1)] : 0x00u;
                float c_lo_code = lds_code[wb & 0x0F];
                float c_hi_code = lds_code[wb >> 4];

                if (wOK) {
                    // Q4_1 layout: scale and min per block -> K/16 floats total
                    size_t fi = (size_t)wLine * blocks_per_row * 2 + b * 2;
                    float scale_val = W_scales_f[fi];
                    float min_val   = W_scales_f[fi + 1];

                    // Use Q4_1 formula: value = scale × codebook[nib] + min
                    // For Q4_0/Q8_0, caller sets scale=delta, min=0.0f
                    b_frag[ele]     = (__half)(c_lo_code * scale_val + min_val);
                    b_frag[ele + 1] = (__half)(c_hi_code * scale_val + min_val);
                } else {
                    b_frag[ele]     = __half(0.0f);
                    b_frag[ele + 1] = __half(0.0f);
                }
            }

            c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

            #pragma unroll
            for (int ele = 0; ele < 16; ++ele) {   // flush the strip into FP32 accumulation
                accum[ele / 2 + (lIdx % 2)][lane] += (_Float16)c_frag[ele];
            }
            c_frag = {};
            __syncthreads();
        }
    }

    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {       // thread t owns cells {row=2*ele+(t/16), col=t%16}
        int r = ele * 2 + (lIdx / 16);
        int tr = baseRow + r;
        int tc = baseCol + lane;
        if (tr < M && tc < N) C[(size_t)tr * N + tc] = accum[r][lane];
    }
}
