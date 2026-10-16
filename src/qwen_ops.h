// qwen_ops.h -- CPU reference ops for the Qwen3.5 single-token forward pass (Phase 20 Step 3c).
//
// Small per-token vector ops that are not worth a kernel yet. The forward pass (qwen_model.cpp) and
// tests/forward_verify call the same functions, so the test checks the code the engine runs.
// Semantics follow ggml: RMSNorm is x / sqrt(mean(x^2) + eps) * w; RoPE is NeoX-style (pairs i and
// i + n_rot/2) over the first n_rot dims, which is what Qwen3.5's M-RoPE reduces to for text tokens
// (all position sections equal the token position).

#ifndef QWEN_OPS_H
#define QWEN_OPS_H

#include <hip/hip_fp16.h>

#include <math.h>

// y[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]. w may be NULL (no gamma). x and y may alias.
void qwen_rmsnorm(const float* x, const float* w, float* y, int n, float eps);

// In place NeoX RoPE on one head of head_dim values; dims >= n_rot are left unchanged.
void qwen_rope_neox(float* v, int n_rot, int pos, float freq_base);

// Causal single-query GQA attention over n_kv cached positions (0..n_kv-1).
// q: [n_head * head_dim]; K, V: row p at p * kv_stride, head g at g * head_dim; out: [n_head * head_dim].
// Query head h reads KV head h / (n_head / n_head_kv), as ggml's broadcast does.
void qwen_attention(const float* q, const __half* K, const __half* V, int kv_stride, int n_kv,
                    int n_head, int n_head_kv, int head_dim, float scale, float* out);

// ---- Gated DeltaNet (Phase 20 Step 5), following ggml's SSM_CONV, L2_NORM and GATED_DELTA_NET CPU ops

// One token of the causal depthwise conv1d. state: [channels][d_conv - 1], oldest first, updated in place
// (shifted by one, x appended). w: [channels][d_conv] (ssm_conv1d, d_conv contiguous). y: raw conv output
// [channels], before SiLU. y must not alias state or x.
void qwen_ssm_conv_step(float* state, const float* x, const float* w, float* y, int channels, int d_conv);

// y = x / max(||x||_2, eps). x and y may alias.
void qwen_l2norm(const float* x, float* y, int n, float eps);

// One token of the gated delta rule for all V heads. Per V head h (K/Q head h % n_k_heads, as ggml
// broadcasts), with S the d x d state, S_h[j * d + i] = S[i][j] (ggml's transposed layout, so it diffs
// directly against the reference new_state):
//   S *= exp(g[h]);  delta = (v - S^T k) * beta[h];  S += k delta^T;  out = S^T q / sqrt(d)
// q, k: [n_k_heads * d] (already L2-normed); v, out: [n_v_heads * d]; g, beta: [n_v_heads].
void qwen_gated_delta_step(float* S, const float* q, const float* k, const float* v, const float* g,
                           const float* beta, int n_v_heads, int n_k_heads, int d, float* out);

static inline float qwen_softplus(float x) { return x > 20.0f ? x : logf(1.0f + expf(x)); }
static inline float qwen_silu(float x) { return x / (1.0f + __builtin_expf(-x)); }
static inline float qwen_sigmoid(float x) { return 1.0f / (1.0f + __builtin_expf(-x)); }

#endif // QWEN_OPS_H
