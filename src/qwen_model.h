// qwen_model.h -- Qwen3.5 forward pass host API.

#ifndef QWEN_MODEL_H
#define QWEN_MODEL_H

#include <stddef.h>
#include <stdint.h>
#include "inference.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct QwenModel qwen_model_t;

// Create model from an initialized inference_model_t (engine model with loaded weights)
qwen_model_t* qwen_model_create(inference_model_t* engine_model);

// Get the qwen model from an inference model (created during kv_cache_init)
qwen_model_t* inference_get_qwen_model(inference_model_t* model);

// Free model resources
void qwen_model_free(qwen_model_t* model);

// Run single-token forward pass
// token_id: input token
// pos: absolute position in sequence (for RoPE + KV cache indexing). Tokens of one sequence must come in
//      order; pos 0 starts a new sequence and zeroes the DeltaNet recurrent state.
// logits: output buffer [vocab_size] FP32 (pre-allocated by caller), or NULL to skip the copy and read them
//         in place with qwen_model_logits()
// Returns 0 on success, negative on error
int qwen_model_forward(qwen_model_t* model, int token_id, int pos, float* logits);

// Logits of the last qwen_model_forward, [vocab_size] FP32, in the pinned scratch buffer (no copy). Valid
// until the next forward pass.
const float* qwen_model_logits(qwen_model_t* model);

// Called after each decoder layer with the residual stream x[n] (n = n_embd); for tests and tracing.
// cb == NULL removes it.
typedef void (*qwen_layer_cb)(int layer, int pos, const float* x, int n, void* user);
void qwen_model_set_layer_callback(qwen_model_t* model, qwen_layer_cb cb, void* user);

// Wall-time breakdown of the forward passes since the last reset (CPU ops, GPU matvecs, weight bandwidth),
// printed to stderr under the given title
void qwen_model_profile_reset(qwen_model_t* model);
void qwen_model_profile_print(qwen_model_t* model, const char* title);

// Batched forward pass for multiple sequences
// tokens: array of token IDs, length batch_size
// positions: array of positions (absolute) for each sequence, length batch_size
// seq_lens: array of sequence lengths for each batch item (for masking), length batch_size
// batch_size: number of sequences in batch
// seq_len: maximum sequence length (for kernel launch configuration)
// logits_out: output buffer [batch_size * vocab_size] FP32 (pre-allocated by caller)
// Returns 0 on success, negative on error
int qwen_model_forward_batch(qwen_model_t* model,
                             const int32_t* tokens,
                             const int* positions,
                             const int* seq_lens,
                             int batch_size,
                             int seq_len,
                             float* logits_out);

#ifdef __cplusplus
}
#endif

#endif // QWEN_MODEL_H