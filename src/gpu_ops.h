// gpu_ops.h -- host launchers for gpu_ops.hip (per-token vector kernels). Pointers must be device-accessible
// (pinned host memory on UMA); launches are asynchronous on `stream`.

#ifndef GPU_OPS_H
#define GPU_OPS_H

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load gpu_ops.co. Returns 0 on success.
int gpu_ops_init(const char* co_path);
void gpu_ops_free(void);

// x += add (skipped when add is NULL), then y = RMSNorm(x) * w (w NULL: no gamma)
hipError_t gpu_add_rmsnorm(float* x, const float* add, const float* w, float* y, int n, float eps, hipStream_t stream);

// out = SiLU(g) * u
hipError_t gpu_swiglu(const float* g, const float* u, float* out, int n, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GPU_OPS_H
