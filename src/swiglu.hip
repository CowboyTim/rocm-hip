// swiglu.hip -- Phase 4: SwiGLU activation for Qwen 3 MLP layers.
// SwiGLU(x) = SiLU(gate) * up  where SiLU(z) = z * sigmoid(z)
// Both gate and up projections are FP4-quantized (ROCmFP4 Codebook10).
// This kernel fuses: gate_proj (A x W_gate), up_proj (A x W_up), SiLU(gate) * up -> output.
// A: [M x K] FP16 activations; W_gate, W_up: [N x K] ROCmFP4 packed; out: [M x N] FP32.
// FIXED: Accepts pre-decoded float scale arrays instead of UE4M3 byte arrays.
// This avoids GPU pointer dereference issues on gfx1151 architecture.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void swiglu_fp4(const __half* A, 
                                        const unsigned char* W_gate,
                                        const unsigned char* W_gate_scale_b,
                                        const unsigned char* W_up,
                                        const unsigned char* W_up_scale_b,
                                        float* out,
                                        int M, int N, int K /* multiple of 16 */)
{
    const int lIdx = threadIdx.x;             // 0..31
    const int lane = lIdx % 16;               // unique row/column index within the tile

    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    const int aRow = baseRow + lane;          // this thread's A row (duplicated across t/16 halves)
    const int wLine = baseCol + lane;         // this thread's weight line (= output column)

    // LDS LUT: Codebook10 nibble -> FP32 magnitude (shared with gemm_act_fp4)
    __shared__ float lds_code[16];
    for (int i = lIdx; i < 16; i += 32) lds_code[i] = dequant_code(i);
    __syncthreads();

    __shared__ float accum_gate[16][16];
    __shared__ float accum_up[16][16];
    for (int i = lIdx; i < 256; i += 32) {
        accum_gate[i / 16][i % 16] = 0.0f;
        accum_up[i / 16][i % 16] = 0.0f;
    }
    __syncthreads();

    half16 c_frag_gate = {};
    half16 c_frag_up = {};

    const int numStrips = K / HALF_BLOCK;
    const size_t w_row_bytes = (size_t)K / NIBBLES_BYTE;

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * HALF_BLOCK;

        half16 a_frag, b_frag_gate, b_frag_up;

        #pragma unroll
        for (int ele = 0; ele < HALF_BLOCK; ele += 2) {
            int k = kStart + ele;
            bool aOK = (aRow < M);

            a_frag[ele]     = aOK ? ((const __half*)A)[(size_t)aRow * K + k] : __half(0.0f);
            a_frag[ele + 1] = aOK ? ((const __half*)A)[(size_t)aRow * K + k + 1] : __half(0.0f);

            bool wOK = (wLine < N);
            // Decode UE4M3 scale bytes inline - avoids dereferencing float* through kernarg on gfx1151
            float wScale_gate = wOK ? ue4m3_to_float(W_gate_scale_b[(size_t)wLine * numStrips + s]) : 0.0f;
            float wScale_up   = wOK ? ue4m3_to_float(W_up_scale_b[(size_t)wLine * numStrips + s])   : 0.0f;
            uint8_t wb_gate = wOK ? W_gate[(size_t)wLine * w_row_bytes + (k >> 1)] : 0x00u;
            uint8_t wb_up   = wOK ? W_up[(size_t)wLine * w_row_bytes + (k >> 1)]   : 0x00u;

            float c_lo_gate = lds_code[wb_gate & 0x0F];
            float c_hi_gate = lds_code[wb_gate >> 4];
            float c_lo_up   = lds_code[wb_up & 0x0F];
            float c_hi_up   = lds_code[wb_up >> 4];

            b_frag_gate[ele]     = (__half)(c_lo_gate * wScale_gate);
            b_frag_gate[ele + 1] = (__half)(c_hi_gate * wScale_gate);
            b_frag_up[ele]       = (__half)(c_lo_up * wScale_up);
            b_frag_up[ele + 1]   = (__half)(c_hi_up * wScale_up);
        }

        c_frag_gate = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag_gate, c_frag_gate, false);
        c_frag_up   = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag_up,   c_frag_up,   false);

        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {
            accum_gate[ele * 2 + (lIdx / 16)][lane] += (_Float16)c_frag_gate[ele * 2];
            accum_up[ele * 2 + (lIdx / 16)][lane]   += (_Float16)c_frag_up[ele * 2];
        }
        c_frag_gate = {};
        c_frag_up = {};
        __syncthreads();
    }

    // Store final results and apply SwiGLU: out = SiLU(gate) * up
    // SiLU(z) = z * sigmoid(z)
    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        int r = ele * 2 + (lIdx / 16);
        int tr = baseRow + r;
        int tc = baseCol + lane;
        if (tr < M && tc < N) {
            float gate_val = accum_gate[r][lane];
            float up_val   = accum_up[r][lane];
            float silu = gate_val / (1.0f + expf(-gate_val));  // SiLU(z) = z * sigmoid(z)
            out[(size_t)tr * N + tc] = silu * up_val;
        }
    }
}
