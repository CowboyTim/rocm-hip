// kv_cache_kernel.hip -- Phase 3: KV cache zero-copy self-test kernel
//
// Proves the GPU can read AND write pages pinned via hipHostRegister inside the KV-cache mapping:
// for every (layer, pos, elem) slot it reads the CPU-written FP16 K/V patterns from the mapped
// file, computes check = K + 2*V + pos in FP32, and writes it into a second registered buffer.
// The host then verifies every element through its own mapping of the same pages.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void kv_cache_selftest(
    const __half* k_base,             // start of all K regions (inside the KV mapping)
    const __half* v_base,             // start of all V regions (inside the KV mapping)
    float* check_out,                 // [n_layers * n_ctx * d_kv] FP32, separate registered buffer
    int n_layers, int n_ctx, int d_kv,
    size_t k_region_halfs,            // full per-layer K region size in halfs (row stride across layers)
    size_t v_region_halfs)           // full per-layer V region size in halfs
{
    const size_t total = (size_t)n_layers * n_ctx * d_kv;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        const size_t rem  = idx % ((size_t)n_ctx * d_kv);
        const int l       = idx / ((size_t)n_ctx * d_kv);
        const size_t rem2 = rem / d_kv;          // pos within layer
        const int e       = rem % d_kv;

        const __half* k_row = k_base + ((size_t)l * k_region_halfs + rem2 * d_kv + e);
        const __half* v_row = v_base + ((size_t)l * v_region_halfs + rem2 * d_kv + e);

        check_out[idx] = (float)(*k_row) + 2.0f * (float)(*v_row) + (float)((int)rem2);
    }
}
