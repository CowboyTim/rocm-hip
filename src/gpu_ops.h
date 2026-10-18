// gpu_ops.h -- host launchers for gpu_ops.hip (per-token vector kernels). Pointers must be device-accessible
// (pinned host memory on UMA); launches are asynchronous on `stream`.

#ifndef GPU_OPS_H
#define GPU_OPS_H

#include <hip/hip_runtime.h>

#include "qmatvec.h"

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
// renormalized softmax weights) and sg [T] (sigmoid shared-expert gate). n_expert <= 1024, n_used <= 64.
// ids_stride (0: n_used) > n_used: ids rows of ids_stride entries, the one after the selection set to -1 (the
// shared expert as one more slot of the expert-indexed matvecs, qmatvec_launch_q8_id_x).
hipError_t gpu_moe_router(const float* logits, const float* x, const float* w_shexp, int* ids, float* wts, float* sg,
                          int n_expert, int n_used, int n, int T, hipStream_t stream, int ids_stride);

// out [T][n] = sum_j wts[t][j] * down[t][j] + sg[t] * shared[t]; down [T][n_used][n]
hipError_t gpu_moe_combine(const float* down, const float* wts, const float* shared, const float* sg, float* out,
                           int n_used, int n, int T, hipStream_t stream);

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
// dilation dil; state [(K - 1) * dil][m] holds the inputs before this batch and is advanced
hipError_t gpu_ple_conv(const float* normed, const float* gated, float* res, float* state, const void* w, int m, int T,
                        int K, int dil, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GPU_OPS_H
