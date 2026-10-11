// layernorm.hip -- Phase 3: RMSNorm on unified-memory activations (Qwen3-style).
//
//   y[t][c] = x[t][c] * rsqrtf( sum_c(x[t][c]^2) / n_embd + eps ) * gamma[c]
//
// x is [n_tokens x n_embd] FP16, gamma is [n_embd] FP16, y is [n_tokens x n_embd] FP16 -- all in
// hipHostRegister'ed pages so nothing is copied. One 32-thread block handles one token row: a
// strided pass-1 sums squares per thread, a warp-wide shfl butterfly folds it to every lane, then
// pass-2 normalizes and applies the per-channel scale. Partial rows are guard-clipped; arbitrary
// n_embd works (any multiple or not). Launch with blockDim == 32 (one wavefront on gfx1151).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#pragma GCC diagnostic ignored "-Wpass-failed"

extern "C" __global__ void rmsnorm(const __half* x, const __half* gamma, __half* y,
                                   int n_tokens, int n_embd, float eps)
{
    const int t = blockIdx.x;                  // token row index
    if (t >= n_tokens) return;                 // uniform per block

    const __half* xr = x + (size_t)t * n_embd;
    __half* yr      = y + (size_t)t * n_embd;

    // Pass 1: sum of squares over this thread's strided slice.
    float ssq = 0.0f;
    for (int c = threadIdx.x; c < n_embd; c += blockDim.x) {
        float v = (float)xr[c];
        ssq += v * v;
    }

    // Warp butterfly reduction -- block is exactly one wavefront, so every lane ends up with the full sum.
    // This toolchain (-mno-wavefrontsize64) takes a 64-bit active mask for shfl_sync.
    #pragma unroll
    for (int off = blockDim.x / 2; off > 0; off >>= 1)
        ssq += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, ssq, off);

    const float inv_rms = rsqrtf(ssq / (float)n_embd + eps);

    // Pass 2: normalize and apply the per-channel scale.
    for (int c = threadIdx.x; c < n_embd; c += blockDim.x) {
        float v = (float)xr[c];
        float g = gamma ? (float)gamma[c] : 1.0f;  // Use unity scale if gamma is null
        yr[c] = (__half)(v * inv_rms * g);
    }
}
