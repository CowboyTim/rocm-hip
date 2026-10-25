// gpu_ops.h -- host launchers for gpu_ops.hip (per-token vector kernels). Pointers must be device-accessible
// (pinned host memory on UMA); launches are asynchronous on `stream`.

#ifndef GPU_OPS_H
#define GPU_OPS_H

#include <hip/hip_runtime.h>

#include "qmatvec.h"
#include "deltanet.h"  // deltanet_mid_states

#ifdef __cplusplus
extern "C" {
#endif

// Load gpu_ops.co. Returns 0 on success.
int gpu_ops_init(const char* co_path);
void gpu_ops_free(void);

// For each of T rows of n values: x_out = x_in + add, then y = RMSNorm(x_out) * w (w NULL: no gamma). With
// add NULL, x_out is not written and y = RMSNorm(x_in) * w. x_out must not alias x_in when add is set
// (ping-pong the residual buffers). If q8 is not NULL, the T rows of y are also quantized into it as one
// flattened [T * n] vector for the int8 matvecs (n % 256 == 0, T * n <= q8->max_k).
hipError_t gpu_add_rmsnorm(const float* x_in, const float* add, float* x_out, const float* w, float* y, int n, int T,
                           float eps, const qmatvec_q8* q8, hipStream_t stream);

// out = SiLU(g) * u, quantized into q8 as well when it is not NULL (n % 32 == 0: whole Q8_0 blocks; the Q8_K
// side is complete only for rows of whole 256-value blocks)
hipError_t gpu_swiglu(const float* g, const float* u, float* out, int n, const qmatvec_q8* q8, hipStream_t stream);

// MoE router for T tokens: logits [T][n_expert] (ffn_gate_inp output), x [T][n] (the FFN input) and the shared
// expert's gate vector w_shexp [n] -> ids / wts [T][n_used] (selected experts, descending, and their
// renormalized softmax weights) and sg [T] (sigmoid shared-expert gate; not written without w_shexp). n_expert <= 1024, n_used <= 64.
// ids_stride (0: n_used) > n_used: ids rows of ids_stride entries, the one after the selection set to -1 (the
// shared expert as one more slot of the expert-indexed matvecs, qmatvec_launch_q8_id_x).
hipError_t gpu_moe_router(const float* logits, const float* x, const float* w_shexp, int* ids, float* wts, float* sg,
                          int n_expert, int n_used, int n, int T, hipStream_t stream, int ids_stride);

// out [T][n] = sum_j wts[t][j] * down[t][j] + sg[t] * shared[t]; down [T][n_used][n]
hipError_t gpu_moe_combine(const float* down, const float* wts, const float* shared, const float* sg, float* out,
                           int n_used, int n, int T, hipStream_t stream);
// The same with each expert's output scaled by exp_scale[ids[t][j]] first (exp_scale NULL: none) and, with shared
// NULL, no shared expert (Gemma-4: per-expert scales, no shared expert; w_shexp NULL in gpu_moe_router)
hipError_t gpu_moe_combine_scaled(const float* down, const float* wts, const float* shared, const float* sg,
                                  const int* ids, const float* exp_scale, float* out, int n_used, int n, int T,
                                  hipStream_t stream);

// Gemma-4 (gpu_ops.hip). gpu_add_normed: out = (base + RMSNorm(a) * w) * (*scale, or 1 if NULL) for T rows of n
// values (n % 256 == 0, n <= 4096; out may be base, not a). gpu_geglu: out [rows][h] = GELU_tanh(g) * u, g / u row
// r at gate + r * gate_stride / up + r * up_stride (the experts' fused [gate | up] rows: up = gate + h, strides
// 2 h), quantized into q8 if set. gpu_softcap: x = cap * tanh(x / cap).
hipError_t gpu_add_normed(const float* base, const float* a, const float* w, const float* scale, float* out, int n,
                          int T, float eps, hipStream_t stream);
// Gemma-4's norm chains, fused (the same bits as the separate kernels; n % 256 == 0, n <= 4096):
// gpu_gemma_attn_post: x_out = x + RMSNorm(a) * w_a; y1 = RMSNorm(x_out) * w1 (quantized into q8 if set); y2 =
// RMSNorm(x_out) * w2 if w2. gpu_gemma_ffn_post: the experts' outputs combined as gpu_moe_combine_scaled (down
// [T * n_used][n], wts, ids, exp_scale), h = RMSNorm(mlp) * w1 + RMSNorm(combined) * w2, x_out = (x + RMSNorm(h) *
// w_post) * (*scale, or 1), y = RMSNorm(x_out) * w_next (quantized into q8 if set).
hipError_t gpu_gemma_attn_post(const float* x, const float* a, const float* w_a, float* x_out, const float* w1,
                               float* y1, const qmatvec_q8* q8, const float* w2, float* y2, int n, int T, float eps,
                               hipStream_t stream);
hipError_t gpu_gemma_ffn_post(const float* x, const float* mlp, const float* down, const float* wts, const int* ids,
                              const float* exp_scale, int n_used, const float* w1, const float* w2, const float* w_post,
                              const float* scale, float* x_out, const float* w_next, float* y, const qmatvec_q8* q8,
                              int n, int T, float eps, hipStream_t stream);
hipError_t gpu_geglu(const float* gate, const float* up, float* out, int rows, int h, int gate_stride, int up_stride,
                     const qmatvec_q8* q8, hipStream_t stream);
hipError_t gpu_softcap(float* x, int n, float cap, hipStream_t stream);

// Group T tokens' selected experts (ids [T][n_used]) by expert for the expert-tiled matmuls: see moe_group in
// gpu_ops.hip. tok_rows / pair_rows [T * n_used]; tile_e / tile_start / tile_n need gpu_moe_max_tiles entries.
static inline int gpu_moe_max_tiles(int T, int n_used, int n_expert, int tile) {
    const int n = T * n_used, bound = n_expert + n / tile;
    return n < bound ? n : bound;
}
hipError_t gpu_moe_group(const int* ids, int T, int n_used, int n_expert, int tile, int* tok_rows, int* pair_rows,
                         int* n_tiles, int* tile_e, int* tile_start, int* tile_n, hipStream_t stream);

// Hyper-connections (qwen4exp; gpu_ops.hip): res holds T tokens x hc streams x n values.
// gpu_hc_norm: if cur [T][n] is set, res_out[t][c] = res_in[t][c] + cur[t] * 2 * sigmoid(inject[t][c] / hc)
// first (res_out unused without cur); then y[t][c] = RMSNorm(that) * w[c * n ..], quantized into q8 (if set) as
// one flattened [T * hc * n] vector. n % 256 == 0, n <= 4096. res_out may be res_in, but a different buffer lets a
// few tokens spread over many workgroups (same results either way).
hipError_t gpu_hc_norm(const float* res_in, float* res_out, const float* cur, const float* inject, const float* w,
                       float* y, int n, int hc, int T, float eps, const qmatvec_q8* q8, hipStream_t stream);
// out[t] = SiLU(lo[t] / hc) for T rows of n values (lo [T][n]), zero-padded to rows of np >= n (out [T][np]);
// quantized into q8 if set (np % 32 == 0)
hipError_t gpu_hc_lowrank(const float* lo, float* out, int n, int np, int T, int hc, const qmatvec_q8* q8,
                          hipStream_t stream);
// out[t] = mean over the streams c of xn[t][c] * sigmoid(gate[t][c]), quantized into q8 if set
hipError_t gpu_hc_mix(const float* xn, const float* gate, float* out, int n, int hc, int T, const qmatvec_q8* q8,
                      hipStream_t stream);
// PLE gate (see ple_gate in gpu_ops.hip): key, res, gated, normed [T][hc][n]; value [T][n]; norm gammas [hc * n]
hipError_t gpu_ple_gate(const float* key, const float* value, const float* res, const float* wk, const float* wq,
                        const float* wc, float* gated, float* normed, int n, int hc, int T, float eps,
                        hipStream_t stream);
// PLE conv: res += gated + SiLU(dilated causal depthwise conv of normed), m channels, K taps (F16 w [m][K]),
// dilation dil; state [(K - 1) * dil][m] holds the inputs before this batch and is advanced. state_mid (optional,
// NULL): also the states after the first tokens (deltanet_mid_states)
hipError_t gpu_ple_conv(const float* normed, const float* gated, float* res, float* state, const void* w, int m, int T,
                        int K, int dil, const deltanet_mid_states* state_mid, hipStream_t stream);

// MTP head input: out[t][c] = [e[t] | h[t][c]] for T tokens, hc streams of n values (e [T][n], h [T][hc][n],
// out [T][hc][2n])
hipError_t gpu_mtp_concat(const float* e, const float* h, float* out, int n, int hc, int T, hipStream_t stream);

// Nemotron-H (gpu_ops.hip): the Mamba-2 mixer on T tokens of zx (ssm_in's rows, stride ld: [z | xBC | dt]).
// gpu_mamba_conv: xBC (at x_off) through the causal conv1d (+ bias, SiLU) into out [T][conv_dim], then the conv state
// [conv_dim][d_conv - 1] advanced. gpu_mamba_scan: the selective scan (state [n_head][64][128], d_state 128, 64 dims
// per head) on out's x / B / C and zx's dt (at dt_off): y [T][d_inner] = scan + D x. gpu_mamba_gnorm: out =
// RMSNorm(y * SiLU(z)) * w per group of d_inner / n_group values (z at zx's row start), quantized into q8 if set.
hipError_t gpu_mamba_conv(const float* zx, long long ld, int x_off, float* conv_state, const float* w, const float* bias,
                          float* out, int conv_dim, int d_conv, int T, hipStream_t stream);
hipError_t gpu_mamba_scan(float* state, const float* xbc, const float* zx, long long ld, int dt_off, const float* dt_b,
                          const float* A, const float* D, float* y, int n_head, int d_inner, int n_group, int T,
                          hipStream_t stream);
hipError_t gpu_mamba_gnorm(const float* y, const float* zx, long long ld, const float* w, float* out, int d_inner,
                           int n_group, int T, float eps, const qmatvec_q8* q8, hipStream_t stream);
// The sigmoid router: the n_used largest sigmoid(logits) + bias per token; ids, and weights sigmoid(logits)
// (normalized to sum 1 if norm) * scale. gpu_relu2: out [rows][h] = max(x, 0)^2, row r of x at in + r * stride,
// quantized into q8 if set.
hipError_t gpu_moe_router_sigmoid(const float* logits, const float* bias, int* ids, float* wts, int n_expert,
                                  int n_used, int norm, float scale, int T, hipStream_t stream);
hipError_t gpu_relu2(const float* in, float* out, int rows, int h, int stride, const qmatvec_q8* q8,
                     hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GPU_OPS_H
