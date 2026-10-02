// softmax_masked.hip -- Phase 3: in-place masked row softmax over attention scores in unified memory.
//
//   s is [n_rows x n_cols] FP32 (mapped). For each row r:
//     valid columns c: c < n_valid_kv AND (qpos == NULL OR c <= qpos[r])   // causal + history mask
//     m    = max of scaled scores over valid columns
//     out[c] = exp(scaled - m) / sum(exp(...))          for valid c
//     out[c] = 0                                       otherwise
//   The scale is folded into the stored values during pass 1 (scores become score*scale in place), so
//   pass 2 only exponentiates. One 32-thread block per row; shfl butterfly folds max and sum. A row
//   with no valid columns is zeroed (guarded against -inf - -inf NaN).
//
// qpos (optional, CPU-mapped int32[n_rows]) holds absolute query positions: decode passes the current
// position (sees full history incl. itself); a prefill chunk passes its absolute indices.

#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_masked(float* s, int n_rows, int n_cols, float scale,
                                          const int* qpos /* NULL disables causal masking */, int n_valid_kv)
{
    const int r = blockIdx.x;                       // row index
    if (r >= n_rows) return;                        // uniform per block

    float* srow = s + (size_t)r * n_cols;
    const int qmax = (qpos != nullptr) ? qpos[r] : n_cols;   // causal ceiling for this row

    // Pass 1: fold in the scale, track the masked max.
    float m = -INFINITY;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        if (c >= n_valid_kv || c > qmax) continue;  // masked out entirely
        float v = srow[c] * scale;
        srow[c] = v;                                // scaled value stays in place for pass 2
        if (v > m) m = v;
    }

    #pragma unroll
    for (int off = blockDim.x / 2; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, m, off));   // every lane gets the row max

    if (m == -INFINITY) {                            // no valid columns: zero the whole row
        for (int c = threadIdx.x; c < n_cols; c += blockDim.x) srow[c] = 0.0f;
        return;
    }

    // Pass 2: exponentiate and sum.
    float e_sum = 0.0f;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        if (c >= n_valid_kv || c > qmax) { srow[c] = 0.0f; continue; }
        float e = expf(srow[c] - m);
        srow[c] = e;                                 // staged probabilities pre-normalization
        e_sum += e;
    }

    #pragma unroll
    for (int off = blockDim.x / 2; off > 0; off >>= 1)   // 64-bit active mask required by -mno-wavefrontsize64
        e_sum += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, e_sum, off);

    const float inv_sum = 1.0f / e_sum;
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        if (c < n_valid_kv && c <= qmax) srow[c] *= inv_sum;   // masked columns already zeroed above
    }
}
