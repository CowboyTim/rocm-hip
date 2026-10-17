// gpu_ops.hip -- small per-token vector kernels of the Qwen3.5 forward pass (Phase 20 speed work).
//
// add_rmsnorm  x_out = x_in + add (if add != NULL), then y = x_out / sqrt(mean(x_out^2) + eps) * w. Every
//              residual add of the forward pass is followed by an RMSNorm (the next FFN's or mixer's, or
//              output_norm), so one fused kernel covers each layer boundary.
// swiglu       out = SiLU(g) * u, element-wise; one workgroup per 256 values.
// Both are the producers of matvec inputs, so they also quantize their output for the int8 matvecs
// (q8_device.h) when o.qK != NULL, which saves a launch each; n must then be a multiple of 256.
// Same math as qwen_rmsnorm / the CPU SwiGLU in qwen_ops (the reference); the sum of squares is a float
// tree reduction here, a double loop there.

#include <hip/hip_runtime.h>

#include "q8_device.h"

// add_rmsnorm: grid ceil(n / 256), block 256. Every workgroup computes the full sum of squares itself (n is
// small, ~10 KB, and the redundant reads hit the cache), then normalizes and quantizes only its own 256-value
// block: the whole GPU works on it instead of one workgroup (that took ~9.5 us, 65 times per token). The
// residual is written to x_out, not back to x_in, so workgroups never read a value another one has already
// updated; the caller ping-pongs the two buffers. Every workgroup sums in the same order, so they all get
// the same scale.
extern "C" __global__ void __launch_bounds__(256)
add_rmsnorm(const float* x_in, const float* add, float* x_out, const float* w, float* y, int n, float eps, Q8Out o) {
    __shared__ float part[8];
    __shared__ Q8Shared sh;
    const int t = threadIdx.x, i = blockIdx.x * 256 + t;
    float ss = 0.0f;
    for (int k = t; k < n; k += 256) {
        const float v = add ? x_in[k] + add[k] : x_in[k];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((t & 31) == 0) part[t >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) tot += part[k];
    const float inv = rsqrtf(tot / n + eps);

    const bool active = i < n;
    float v = 0.0f;
    if (active) {
        v = add ? x_in[i] + add[i] : x_in[i];
        if (add) x_out[i] = v;
        v = v * inv * (w ? w[i] : 1.0f);
        y[i] = v;
    }
    if (o.qK) q8_block_value(v, active, blockIdx.x, t, sh, o);  // uniform; n % 256 == 0 when set
}

extern "C" __global__ void __launch_bounds__(256)
swiglu(const float* g, const float* u, float* out, int n, Q8Out o) {
    __shared__ Q8Shared sh;
    const int i = blockIdx.x * 256 + threadIdx.x;
    float v = 0.0f;
    if (i < n) {
        const float a = g[i];
        v = a / (1.0f + __expf(-a)) * u[i];
        out[i] = v;
    }
    if (o.qK) q8_block_value(v, i < n, blockIdx.x, threadIdx.x, sh, o);  // uniform; n % 256 == 0 when set
}
