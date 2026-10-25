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

#define ATTN_HEAD_DIM 256       // head_dim of attention_part_acc_floats (the kernels: 128, 256 and 512)
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
static inline size_t attention_part_acc_floats_d(int n_q, int n_ctx, int T, int head_dim) {
    return attention_part_slots(n_ctx, T) * (size_t)n_q * (size_t)head_dim;
}
static inline size_t attention_part_acc_floats(int n_q, int n_ctx, int T) {
    return attention_part_acc_floats_d(n_q, n_ctx, T, ATTN_HEAD_DIM);
}

// T tokens at positions pos0 .. pos0 + T - 1; token t attends to positions 0 .. pos0 + t. Per token row t:
//   head_dim D = 128 or 256; gated: qfull [n_q][2 * D] (attn_q output, [Q | gate] per head; Qwen3.5 and later),
//   else qfull [n_q][D] (Qwen3) and no output gate. k, v [n_kv * D] (attn_k / attn_v outputs), q: scratch
//   [n_q * D], out [n_q * D]. q_norm, k_norm [D]; K, V: the layer's cache (row p at
//   (p >> kv_shift) * kv_bstride + (p & (2^kv_shift - 1)) * kv_stride halves from row 0, kv_stride = n_kv * D:
//   kv_cache.h's blocks; a contiguous cache passes kv_shift 30, kv_bstride 0); part_ml / part_acc: scratch sized
//   by the helpers above.
//   q8: if not NULL, out is also quantized into it (flattened [T * n_q * D]) for the int8 attn_output matmul.
//   n_q must be a multiple of 256 / D.
hipError_t attention_forward(const float* qfull, const float* k, const float* v, const float* q_norm,
                             const float* k_norm, __half* K, __half* V, int kv_shift, long long kv_bstride, int n_q,
                             int n_kv, int n_rot, int pos0, int T,
                             float freq_base, float eps, float* q, float* part_ml, float* part_acc, float* out,
                             int head_dim, int gated, const qmatvec_q8* q8, hipStream_t stream);

// What attention_forward_opts adds (Gemma-4); zeros give attention_forward's behaviour
typedef struct {
    int head_dim;    // 128, 256 or 512 (512: no attn_flash, every batch takes attn_partial)
    int gated;       // attn_q holds [Q | gate] per head
    int kv_stride;   // halves between the cache rows of two positions; 0: n_kv * head_dim (a layer that uses only
                     // part of a wider cache row, as Gemma-4's global layers do, passes the row width)
    int rot_pairs;   // NeoX pairs (of n_rot / 2) that rotate; 0: all. Gemma-4's global layers: 64 of 256, the
                     // rest have frequency factor 1e30 (rope_freqs), which leaves them unrotated
    int v_norm;      // RMSNorm (no gamma) of each V head before it is cached
    int window;      // sliding window: a token at p attends to positions > p - window only; 0: all before it
    float scale;     // the scores' scale; 0: 1 / sqrt(head_dim) (Gemma-4: 1, its q / k norms replace it)
    int no_qk_norm;  // Q and K as they come (no RMSNorm; Nemotron-H), q_norm / k_norm unused
} attention_opts;

hipError_t attention_forward_opts(const float* qfull, const float* k, const float* v, const float* q_norm,
                                  const float* k_norm, __half* K, __half* V, int kv_shift, long long kv_bstride,
                                  int n_q, int n_kv, int n_rot, int pos0, int T, float freq_base, float eps, float* q,
                                  float* part_ml, float* part_acc, float* out, const attention_opts* opts,
                                  const qmatvec_q8* q8, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // ATTENTION_H
