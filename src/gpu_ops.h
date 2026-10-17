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

// x_out = x_in + add, then y = RMSNorm(x_out) * w (w NULL: no gamma). With add NULL, x_out is not written
// and y = RMSNorm(x_in) * w. x_out must not alias x_in when add is set (ping-pong the residual buffers).
// If q8 is not NULL, y is also quantized into it for the int8 matvecs (n % 256 == 0, n <= q8->max_k).
hipError_t gpu_add_rmsnorm(const float* x_in, const float* add, float* x_out, const float* w, float* y, int n,
                           float eps, const qmatvec_q8* q8, hipStream_t stream);

// out = SiLU(g) * u, quantized into q8 as well when it is not NULL (n % 256 == 0)
hipError_t gpu_swiglu(const float* g, const float* u, float* out, int n, const qmatvec_q8* q8, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GPU_OPS_H
