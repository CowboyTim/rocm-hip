// deltanet.hip -- Gated DeltaNet token mixer kernels, one token (Phase 20 speed work).
//
// Two kernels, queued back to back between the input projections (qkv, z, alpha, beta) and ssm_out, so a
// DeltaNet layer needs no CPU round trip. Same math as the CPU reference in qwen_ops.cpp
// (qwen_ssm_conv_step, qwen_l2norm, qwen_gated_delta_step), which follows llama.cpp's qwen35 graph.
//
// deltanet_pre   grid conv_dim / DS, block DS. Workgroup b covers channels [b*DS, (b+1)*DS) of [q | k | v],
//                which is exactly one head. Per channel: causal conv1d with the persistent conv state,
//                then SiLU. q and k heads (the first n_qk workgroups) are then L2-normed. Workgroup 0 also
//                computes the per-V-head decay g = softplus(alpha + dt_bias) * ssm_a and beta = sigmoid.
// deltanet_rule  grid n_v_heads, block DS. Thread j owns column j of its head's DS x DS state S, stored
//                row-major (S[i][j] at i*DS + j), so the loads coalesce across the wave. Every step of the
//                delta rule for column j is local to thread j:
//                  S[:, j] *= exp(g);  delta_j = (v_j - S[:, j] . k) * beta;  S[:, j] += k * delta_j;
//                  o_j = S[:, j] . q / sqrt(DS)
//                Then the gated RMSNorm over the head: o * rsqrt(mean(o^2) + eps) * ssm_norm * SiLU(z).
//                V head h reads Q/K head h % n_k_heads, as ggml broadcasts.
//
// The state layout is the transpose of ggml's (and of qwen_gated_delta_step's), chosen for coalescing;
// tests transpose when they compare with the reference new_state.

#include <hip/hip_runtime.h>

#define DS 128          // head size (ssm.state_size); the host checks it
#define MAX_DCONV 8

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// Sum over the DS threads of a workgroup (wave32)
__device__ float block_sum(float v) {
    __shared__ float part[DS / 32];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) part[w] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < DS / 32; ++i) s += part[i];
    __syncthreads();
    return s;
}

// Both kernels take T consecutive tokens (prefill; T = 1 when generating) and loop over them in order: the
// recurrence is sequential, but the conv history and each state column stay in registers for the whole batch,
// so the state is read and written once per batch instead of once per token. Per-token rows: qkv / conv_out
// [conv_dim], alpha / beta / g [n_v], z / out [n_v * DS].

extern "C" __global__ void __launch_bounds__(DS)
deltanet_pre(const float* qkv, float* conv_state, const float* conv_w, int d_conv, float* conv_out, int n_qk,
             const float* alpha, const float* beta_in, const float* dt_bias, const float* ssm_a, float* g_out,
             float* beta_out, int n_v, float eps, int T) {
    const int conv_dim = gridDim.x * DS;
    const int c = blockIdx.x * DS + threadIdx.x;
    const int ns = d_conv - 1;
    float* st = conv_state + (size_t)c * ns;
    const float* w = conv_w + (size_t)c * d_conv;
    float hist[MAX_DCONV], wc[MAX_DCONV];
    for (int i = 0; i < ns; ++i) hist[i] = st[i];
    for (int i = 0; i < d_conv; ++i) wc[i] = w[i];

    for (int t = 0; t < T; ++t) {
        const float x = qkv[(size_t)t * conv_dim + c];
        float s = 0.0f;
        for (int i = 0; i < ns; ++i) s += hist[i] * wc[i];
        s += x * wc[ns];
        for (int i = 0; i + 1 < ns; ++i) hist[i] = hist[i + 1];
        hist[ns - 1] = x;

        float y = silu(s);
        if ((int)blockIdx.x < n_qk) {  // uniform per workgroup
            const float sum = block_sum(y * y);
            y *= 1.0f / fmaxf(sqrtf(sum), eps);
        }
        conv_out[(size_t)t * conv_dim + c] = y;
    }
    for (int i = 0; i < ns; ++i) st[i] = hist[i];

    if (blockIdx.x == 0)
        for (int k = threadIdx.x; k < T * n_v; k += DS) {
            const int h = k % n_v;
            const float a = alpha[k] + dt_bias[h];
            const float sp = a > 20.0f ? a : logf(1.0f + expf(a));
            g_out[k] = sp * ssm_a[h];
            beta_out[k] = 1.0f / (1.0f + expf(-beta_in[k]));
        }
}

extern "C" __global__ void __launch_bounds__(DS)
deltanet_rule(float* state, const float* conv_out, const float* g, const float* beta, const float* z,
              const float* norm_w, float* out, int n_k, float eps, int T) {
    __shared__ float sq[DS], sk[DS];
    const int h = blockIdx.x, j = threadIdx.x, hk = h % n_k, n_v = gridDim.x;
    const int conv_dim = 2 * n_k * DS + n_v * DS;
    float* S = state + (size_t)h * DS * DS;
    float col[DS];
#pragma unroll
    for (int i = 0; i < DS; ++i) col[i] = S[i * DS + j];
    const float nw = norm_w[j];

    for (int t = 0; t < T; ++t) {
        const float* co = conv_out + (size_t)t * conv_dim;
        __syncthreads();  // the previous token's reads of sq / sk are done
        sq[j] = co[hk * DS + j];
        sk[j] = co[n_k * DS + hk * DS + j];
        const float v = co[2 * n_k * DS + h * DS + j];
        __syncthreads();

        const float decay = expf(g[t * n_v + h]);
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < DS; ++i) {
            col[i] *= decay;
            kv += col[i] * sk[i];
        }
        const float delta = (v - kv) * beta[t * n_v + h];
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < DS; ++i) {
            col[i] += sk[i] * delta;
            o += col[i] * sq[i];
        }
        o *= 1.0f / sqrtf((float)DS);

        const float ss = block_sum(o * o);
        const float inv = rsqrtf(ss / DS + eps);
        const size_t r = (size_t)t * n_v * DS + h * DS + j;
        out[r] = o * inv * nw * silu(z[r]);
    }
#pragma unroll
    for (int i = 0; i < DS; ++i) S[i * DS + j] = col[i];
}
