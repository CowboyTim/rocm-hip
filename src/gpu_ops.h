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

// out = SiLU(g) * u, quantized into q8 as well when it is not NULL (n % 256 == 0)
hipError_t gpu_swiglu(const float* g, const float* u, float* out, int n, const qmatvec_q8* q8, hipStream_t stream);

// MoE router for T tokens: logits [T][n_expert] (ffn_gate_inp output), x [T][n] (the FFN input) and the shared
// expert's gate vector w_shexp [n] -> ids / wts [T][n_used] (selected experts, descending, and their
// renormalized softmax weights) and sg [T] (sigmoid shared-expert gate). n_expert <= 1024, n_used <= 64.
hipError_t gpu_moe_router(const float* logits, const float* x, const float* w_shexp, int* ids, float* wts, float* sg,
                          int n_expert, int n_used, int n, int T, hipStream_t stream);

// out [T][n] = sum_j wts[t][j] * down[t][j] + sg[t] * shared[t]; down [T][n_used][n]
hipError_t gpu_moe_combine(const float* down, const float* wts, const float* shared, const float* sg, float* out,
                           int n_used, int n, int T, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GPU_OPS_H
