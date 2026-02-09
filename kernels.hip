#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

extern "C" __global__ void halo_kernel(float* data, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1.0f;
}
