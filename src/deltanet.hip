// deltanet.hip -- Gated DeltaNet token mixer kernels (Phase 20 speed work).
//
// Queued back to back between the input projections (qkv, z, alpha, beta) and ssm_out, so a DeltaNet layer
// needs no CPU round trip. Same math as the CPU reference in qwen_ops.cpp (qwen_ssm_conv_step, qwen_l2norm,
// qwen_gated_delta_step), which follows llama.cpp's qwen35 graph.
//
// deltanet_pre   grid conv_dim / DS, block DS. Workgroup b covers channels [b*DS, (b+1)*DS) of [q | k | v],
//                which is exactly one head. Per channel: causal conv1d with the persistent conv state,
//                then SiLU. q and k heads (the first n_qk workgroups) are then L2-normed. Workgroup 0 also
//                computes the per-V-head decay g = softplus(alpha + dt_bias) * ssm_a and beta = sigmoid.
//                Loops over the T tokens; used for small T (generation).
// deltanet_conv  The same for many tokens (prefill): grid (conv_dim / DS, T), every token in parallel. Each
//                token's conv window is already in qkv (or, for the first d_conv - 1 tokens, in the conv
//                state), so the arithmetic is the same as deltanet_pre's, bit for bit. The conv state is
//                then advanced by deltanet_conv_state (a separate launch: deltanet_conv still reads it).
// deltanet_rule  grid (n_v_heads, DS / CW), block 128: each workgroup owns CW = 128 / RP columns of its
//                head's DS x DS state S. RP adjacent lanes share
//                column j, each holding R = DS / RP of its rows in registers; the column's two dot products
//                are reduced across those lanes with shuffles, so the waves never wait on each other:
//                  S[:, j] *= exp(g);  delta_j = (v_j - S[:, j] . k) * beta;  S[:, j] += k * delta_j;
//                  o_j = S[:, j] . q / sqrt(DS)
//                V head h reads Q/K head h % n_k_heads, as ggml broadcasts. Writes o (not yet normed).
// deltanet_norm  The gated RMSNorm over each head: o * rsqrt(mean(o^2) + eps) * ssm_norm * SiLU(z), in place,
//                grid T * n_v * DS / 256 (two heads per workgroup). Also quantizes the result for the int8
//                ssm_out matmul (q8_device.h) when o.qK != NULL.
//
// The state is stored in deltanet_rule's thread order (deltanet_state_index in deltanet.h), chosen for
// coalescing; tests map it when they compare with the reference new_state.

#include <hip/hip_runtime.h>

#include "q8_device.h"

#define DS 128          // head size (ssm.state_size); the host checks it
#define MAX_DCONV 8
#define RULE_RP 8       // lanes per state column in deltanet_rule
#define RULE_CW (128 / RULE_RP)

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

// Per-token rows: qkv / conv_out [conv_dim], alpha / beta / g [n_v], z / out [n_v * DS].

// One channel of one token: conv1d over the window hist (oldest first) and x, then SiLU, then the L2 norm
// for the q and k heads
__device__ __forceinline__ float conv_channel(const float* hist, const float* wc, int ns, float x, bool qk,
                                              float eps) {
    float s = 0.0f;
    for (int i = 0; i < ns; ++i) s += hist[i] * wc[i];
    s += x * wc[ns];
    float y = silu(s);
    if (qk) {  // uniform per workgroup
        const float sum = block_sum(y * y);
        y *= 1.0f / fmaxf(sqrtf(sum), eps);
    }
    return y;
}

// The decay gate and beta of token rows [k0, k1) * n_v, by one workgroup
__device__ __forceinline__ void gates(const float* alpha, const float* beta_in, const float* dt_bias,
                                      const float* ssm_a, float* g_out, float* beta_out, int n_v, int k0, int k1) {
    for (int k = k0 + threadIdx.x; k < k1; k += DS) {
        const int h = k % n_v;
        const float a = alpha[k] + dt_bias[h];
        const float sp = a > 20.0f ? a : logf(1.0f + expf(a));
        g_out[k] = sp * ssm_a[h];
        beta_out[k] = 1.0f / (1.0f + expf(-beta_in[k]));
    }
}

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
        conv_out[(size_t)t * conv_dim + c] = conv_channel(hist, wc, ns, x, (int)blockIdx.x < n_qk, eps);
        for (int i = 0; i + 1 < ns; ++i) hist[i] = hist[i + 1];
        hist[ns - 1] = x;
    }
    for (int i = 0; i < ns; ++i) st[i] = hist[i];

    if (blockIdx.x == 0) gates(alpha, beta_in, dt_bias, ssm_a, g_out, beta_out, n_v, 0, T * n_v);
}

extern "C" __global__ void __launch_bounds__(DS)
deltanet_conv(const float* qkv, const float* conv_state, const float* conv_w, int d_conv, float* conv_out,
              int n_qk, const float* alpha, const float* beta_in, const float* dt_bias, const float* ssm_a,
              float* g_out, float* beta_out, int n_v, float eps) {
    const int conv_dim = gridDim.x * DS, t = blockIdx.y;
    const int c = blockIdx.x * DS + threadIdx.x;
    const int ns = d_conv - 1;
    const float* w = conv_w + (size_t)c * d_conv;
    float hist[MAX_DCONV], wc[MAX_DCONV];
    for (int i = 0; i < ns; ++i) {
        const int tt = t - ns + i;  // the window's token i; before the batch: from the conv state
        hist[i] = tt >= 0 ? qkv[(size_t)tt * conv_dim + c] : conv_state[(size_t)c * ns + ns + tt];
    }
    for (int i = 0; i < d_conv; ++i) wc[i] = w[i];
    const float x = qkv[(size_t)t * conv_dim + c];
    conv_out[(size_t)t * conv_dim + c] = conv_channel(hist, wc, ns, x, (int)blockIdx.x < n_qk, eps);

    if (blockIdx.x == 0) gates(alpha, beta_in, dt_bias, ssm_a, g_out, beta_out, n_v, t * n_v, (t + 1) * n_v);
}

// Conv state after T tokens: the last d_conv - 1 inputs. grid conv_dim / DS, block DS.
extern "C" __global__ void __launch_bounds__(DS)
deltanet_conv_state(const float* qkv, float* conv_state, int d_conv, int T) {
    const int conv_dim = gridDim.x * DS;
    const int c = blockIdx.x * DS + threadIdx.x;
    const int ns = d_conv - 1;
    float* st = conv_state + (size_t)c * ns;
    float h[MAX_DCONV];
    for (int i = 0; i < ns; ++i) {
        const int tt = T - ns + i;
        h[i] = tt >= 0 ? qkv[(size_t)tt * conv_dim + c] : st[i + T];
    }
    for (int i = 0; i < ns; ++i) st[i] = h[i];
}

extern "C" __global__ void __launch_bounds__(128)
deltanet_rule(float* state, const float* conv_out, const float* g, const float* beta, float* out, int n_k, int T) {
    constexpr int R = DS / RULE_RP;
    const int h = blockIdx.x, n_v = gridDim.x, hk = h % n_k;
    const int p = threadIdx.x % RULE_RP, j = blockIdx.y * RULE_CW + threadIdx.x / RULE_RP;
    const int conv_dim = 2 * n_k * DS + n_v * DS;
    // Rows p*R .. p*R + R - 1 of column j, stored in this thread order (deltanet_state_index in deltanet.h):
    // value r of thread t at ((h * (DS / RULE_CW) + blockIdx.y) * R + r) * 128 + t, coalesced over the wave
    float* S = state + ((size_t)h * (DS / RULE_CW) + blockIdx.y) * R * 128 + threadIdx.x;
    float col[R];
#pragma unroll
    for (int r = 0; r < R; ++r) col[r] = S[r * 128];
    const float scale = 1.0f / sqrtf((float)DS);

    for (int t = 0; t < T; ++t) {
        const float* co = conv_out + (size_t)t * conv_dim;
        const float4* q4 = (const float4*)(co + hk * DS + p * R);
        const float4* k4 = (const float4*)(co + n_k * DS + hk * DS + p * R);
        float q[R], k[R];
#pragma unroll
        for (int r = 0; r < R / 4; ++r) {
            const float4 a = q4[r], b = k4[r];
            q[4 * r] = a.x, q[4 * r + 1] = a.y, q[4 * r + 2] = a.z, q[4 * r + 3] = a.w;
            k[4 * r] = b.x, k[4 * r + 1] = b.y, k[4 * r + 2] = b.z, k[4 * r + 3] = b.w;
        }
        const float v = co[2 * n_k * DS + h * DS + j];
        const float decay = expf(g[t * n_v + h]), bt = beta[t * n_v + h];

        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            col[r] *= decay;
            kv += col[r] * k[r];
        }
#pragma unroll
        for (int off = 1; off < RULE_RP; off <<= 1) kv += __shfl_xor(kv, off, 32);
        const float delta = (v - kv) * bt;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            col[r] += k[r] * delta;
            o += col[r] * q[r];
        }
#pragma unroll
        for (int off = 1; off < RULE_RP; off <<= 1) o += __shfl_xor(o, off, 32);
        if (p == 0) out[(size_t)t * n_v * DS + h * DS + j] = o * scale;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) S[r * 128] = col[r];
}

// Two heads per 256-thread workgroup: waves 0-3 hold head 2 b, waves 4-7 head 2 b + 1
extern "C" __global__ void __launch_bounds__(256)
deltanet_norm(float* out, const float* z, const float* norm_w, float eps, Q8Out o) {
    __shared__ float part[8];
    __shared__ Q8Shared sh;
    const int t = threadIdx.x, w = t >> 5;
    const size_t i = (size_t)blockIdx.x * 256 + t;
    const float v = out[i];
    float ss = v * v;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((t & 31) == 0) part[w] = ss;
    __syncthreads();
    const int w0 = w & 4;
    ss = part[w0] + part[w0 + 1] + part[w0 + 2] + part[w0 + 3];
    const float y = v * rsqrtf(ss / DS + eps) * norm_w[t & (DS - 1)] * silu(z[i]);
    out[i] = y;
    if (o.qK) q8_block_value(y, true, blockIdx.x, t, sh, o);
}
