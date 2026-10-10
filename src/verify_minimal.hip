// Minimal HIP kernel for verifying ROCm 7.10 launch capability
// Uses UE4M3 path from gemm_act_fp4.hip without conditionals
#include <stdio.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "fp4_common.h"

using namespace rocmfp4;

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void verify_minimal(const __half* A, const unsigned char* W,
                                           const unsigned char* W_scales, const float* W_scales_f,
                                           float* C, int M, int N, int K)
{
    // Just test that the kernel loads and the WMMA intrinsic compiles/runs
    // Uses UE4M3 path (W_scales_f == nullptr)
    const int lIdx = threadIdx.x;
    const int lane = lIdx % 16;

    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    const int aRow = baseRow + lane;
    const int wLine = baseCol + lane;

    // LDS LUT initialization
    __shared__ float lds_code[16];
    for (int i = lIdx; i < 16; i += 32) lds_code[i] = dequant_code(i);
    __syncthreads();

    __shared__ float accum[16][16];
    for (int i = lIdx; i < 256; i += 32) accum[i / 16][i % 16] = 0.0f;
    __syncthreads();

    half16 c_frag = {};

    const size_t w_row_bytes = (size_t)K / NIBBLES_BYTE;

    // UE4M3 path (W_scales_f == nullptr)
    const int numStrips = K / HALF_BLOCK;

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * HALF_BLOCK;

        half16 a_frag, b_frag;

        #pragma unroll
        for (int ele = 0; ele < HALF_BLOCK; ele += 2) {
            int k = kStart + ele;
            bool aOK = (aRow < M);

            a_frag[ele]     = aOK ? ((const __half*)A)[(size_t)aRow * K + k] : __half(0.0f);
            a_frag[ele + 1] = aOK ? ((const __half*)A)[(size_t)aRow * K + k + 1] : __half(0.0f);

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
        for (int ele = 0; ele < 8; ++ele) {
            accum[ele * 2 + (lIdx / 16)][lane] += (_Float16)c_frag[ele * 2];
        }
        c_frag = {};
        __syncthreads();
    }

    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        int r = ele * 2 + (lIdx / 16);
        int tr = baseRow + r;
        int tc = baseCol + lane;
        if (tr < M && tc < N) C[(size_t)tr * N + tc] = accum[r][lane];
    }
}
