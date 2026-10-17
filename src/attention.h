// attention.h -- host launchers for attention.hip (gated full attention on the GPU, one token).
//
// All pointers must be device-accessible (pinned host memory on UMA). The three launches go on `stream`
// in order and need no sync between them; see attention.hip for the math.

#ifndef ATTENTION_H
#define ATTENTION_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "qmatvec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ATTN_HEAD_DIM 256       // the kernels are compiled for head_dim = 256
#define ATTN_CHUNK 256          // KV positions per attn_partial workgroup

// Load attention.co. Returns 0 on success.
int attention_init(const char* co_path);
void attention_free(void);

// Partial results are kept per (token, head, slot); a slot is one range of positions. Up to ATTN_MAX_SPLITS
// ranges per token in a batch (attn_flash), and a single token may need one 256-position chunk per slot
// (attn_partial). Batches that use attn_partial run in token groups that fit, so the buffers do not grow
// with n_ctx times T.
#define ATTN_MAX_SPLITS 8

static inline size_t attention_part_slots(int n_ctx, int T) {
    const size_t c = (size_t)(n_ctx + ATTN_CHUNK - 1) / ATTN_CHUNK;
    const size_t per_token = c < ATTN_MAX_SPLITS ? c : ATTN_MAX_SPLITS;
    return c > (size_t)T * per_token ? c : (size_t)T * per_token;
}

// Partial-result buffer sizes, in floats, for batches of up to T tokens and up to n_ctx positions
static inline size_t attention_part_ml_floats(int n_q, int n_ctx, int T) {
    return 2 * attention_part_slots(n_ctx, T) * (size_t)n_q;
}
static inline size_t attention_part_acc_floats(int n_q, int n_ctx, int T) {
    return attention_part_slots(n_ctx, T) * (size_t)n_q * ATTN_HEAD_DIM;
}

// T tokens at positions pos0 .. pos0 + T - 1; token t attends to positions 0 .. pos0 + t. Per token row t:
//   qfull [n_q][2 * 256] (attn_q output, [Q | gate] per head), k, v [n_kv * 256] (attn_k / attn_v outputs),
//   q: scratch [n_q * 256], out [n_q * 256] (gated). q_norm, k_norm [256]; K, V: the layer's cache (row p at
//   p * kv_stride halves, kv_stride = n_kv * 256); part_ml / part_acc: scratch sized by the helpers above.
//   q8: if not NULL, out is also quantized into it (flattened [T * n_q * 256]) for the int8 attn_output matmul.
hipError_t attention_forward(const float* qfull, const float* k, const float* v, const float* q_norm,
                             const float* k_norm, __half* K, __half* V, int n_q, int n_kv, int n_rot, int pos0, int T,
                             float freq_base, float eps, float* q, float* part_ml, float* part_acc, float* out,
                             const qmatvec_q8* q8, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // ATTENTION_H
