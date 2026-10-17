// attention.h -- host launchers for attention.hip (gated full attention on the GPU, one token).
//
// All pointers must be device-accessible (pinned host memory on UMA). The three launches go on `stream`
// in order and need no sync between them; see attention.hip for the math.

#ifndef ATTENTION_H
#define ATTENTION_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ATTN_HEAD_DIM 256       // the kernels are compiled for head_dim = 256
#define ATTN_CHUNK 256          // KV positions per attn_partial workgroup

// Load attention.co. Returns 0 on success.
int attention_init(const char* co_path);
void attention_free(void);

// Partial-result buffer sizes, in floats, for up to n_ctx positions
static inline size_t attention_part_ml_floats(int n_q, int n_ctx) {
    return (size_t)2 * n_q * ((n_ctx + ATTN_CHUNK - 1) / ATTN_CHUNK);
}
static inline size_t attention_part_acc_floats(int n_q, int n_ctx) {
    return (size_t)n_q * ((n_ctx + ATTN_CHUNK - 1) / ATTN_CHUNK) * ATTN_HEAD_DIM;
}

// One token at position pos (attends to positions 0..pos):
//   qfull [n_q][2 * 256] (attn_q output, [Q | gate] per head), k, v [n_kv * 256] (attn_k / attn_v outputs),
//   q_norm, k_norm [256]; K, V: the layer's cache (row p at p * kv_stride halves, kv_stride = n_kv * 256);
//   q: scratch [n_q * 256]; part_ml / part_acc: scratch sized by the helpers above; out [n_q * 256] (gated).
hipError_t attention_forward(const float* qfull, const float* k, const float* v, const float* q_norm,
                             const float* k_norm, __half* K, __half* V, int n_q, int n_kv, int n_rot, int pos,
                             float freq_base, float eps, float* q, float* part_ml, float* part_acc, float* out,
                             hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // ATTENTION_H
