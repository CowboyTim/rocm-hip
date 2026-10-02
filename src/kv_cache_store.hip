// kv_cache_store.hip -- Phase 3: write a batch of roped K/V rows into cache slots (zero-copy store).
//
//   k_dst[pos_start + t][d] = k_src[t][d]   for t in [0,toks), d in [0,d_kv)
//   v_dst[pos_start + t][d] = v_src[t][d]
//
// k_src/v_src are the projection outputs staged contiguously as [toks x d_kv]; dst bases point at the
// layer's full K/V regions inside the managed cache mapping, so this is one flat grid-stride copy per
// operand over pinned pages. The engine calls this after RoPE on each decode/prefill step.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void kv_store(const __half* k_src, const __half* v_src,
                                    __half* k_dst_base, __half* v_dst_base,
                                    int toks, int d_kv, int pos_start, int n_ctx) {
    const size_t half_elems = (size_t)toks * d_kv;
    const size_t total = half_elems * 2;

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        if (idx < half_elems) {
            int t = idx / d_kv;
            if (pos_start + t < n_ctx) k_dst_base[(size_t)(pos_start + t) * d_kv + (idx % d_kv)] = k_src[idx];
        } else {
            size_t j = idx - half_elems;
            int t = j / d_kv;
            if (pos_start + t < n_ctx) v_dst_base[(size_t)(pos_start + t) * d_kv + (j % d_kv)] = v_src[j];
        }
    }
}
