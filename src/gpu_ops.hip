// gpu_ops.hip -- small per-token vector kernels of the Qwen3.5 forward pass (Phase 20 speed work).
//
// add_rmsnorm  one workgroup: x += add (if add != NULL), then y = x / sqrt(mean(x^2) + eps) * w. Every
//              residual add of the forward pass is followed by an RMSNorm (the next FFN's or mixer's, or
//              output_norm), so one fused kernel covers each layer boundary.
// swiglu       out = SiLU(g) * u, element-wise.
// Same math as qwen_rmsnorm / the CPU SwiGLU in qwen_ops (the reference); the sum of squares is a float
// tree reduction here, a double loop there.

#include <hip/hip_runtime.h>

#define NORM_THREADS 1024

extern "C" __global__ void __launch_bounds__(NORM_THREADS)
add_rmsnorm(float* x, const float* add, const float* w, float* y, int n, float eps) {
    __shared__ float part[NORM_THREADS / 32];
    const int t = threadIdx.x;
    float ss = 0.0f;
    for (int i = t; i < n; i += NORM_THREADS) {
        float v = x[i];
        if (add) {
            v += add[i];
            x[i] = v;
        }
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((t & 31) == 0) part[t >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < NORM_THREADS / 32; ++i) tot += part[i];
    const float inv = rsqrtf(tot / n + eps);
    for (int i = t; i < n; i += NORM_THREADS) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

extern "C" __global__ void swiglu(const float* g, const float* u, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float a = g[i];
        out[i] = a / (1.0f + __expf(-a)) * u[i];
    }
}
