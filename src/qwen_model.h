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

// Prefill: n tokens of the sequence at positions pos0 .. pos0 + n - 1 (pos0 = 0 starts a new sequence),
// processed in micro-batches of qwen_model_max_batch tokens: each weight block is read once per 16-token tile
// instead of once per token. Same results as n qwen_model_forward calls. logits (optional, [vocab_size]) gets
// the last token's logits; qwen_model_logits() has them too. Returns 0 on success.
int qwen_model_prefill(qwen_model_t* model, const int* tokens, int n, int pos0, float* logits);

// Tokens per prefill micro-batch (scratch rows)
int qwen_model_max_batch(const qwen_model_t* model);

// Logits of the last qwen_model_forward / qwen_model_prefill, [vocab_size] FP32, in the pinned scratch buffer (no copy). Valid
// until the next forward pass.
const float* qwen_model_logits(qwen_model_t* model);

// Called after each decoder layer with the residual stream x[n] (n = n_embd); for tests and tracing.
// cb == NULL removes it.
typedef void (*qwen_layer_cb)(int layer, int pos, const float* x, int n, void* user);
void qwen_model_set_layer_callback(qwen_model_t* model, qwen_layer_cb cb, void* user);

// Prefix reuse support. A model with recurrent layers (DeltaNet) cannot rewind its state, so a sequence can only
// be restarted from a saved position: qwen_model_state_save snapshots the whole recurrent state into slot
// (allocated on first use, same size as the state), qwen_model_state_restore puts it back. Both are GPU copies
// queued in order with the forward passes; the KV cache needs nothing (later positions are just overwritten).
// Return 0 on success, -1 on error. qwen_model_is_recurrent: 1 if there is recurrent state at all.
int qwen_model_is_recurrent(const qwen_model_t* model);
int qwen_model_state_save(qwen_model_t* model, int slot);
int qwen_model_state_restore(qwen_model_t* model, int slot);

// Opt-in int8 (Q8_0) copy of a float LM head (F32 / F16 / BF16; the 35B's 1 GB BF16 one): built on the GPU on
// first use, about half the bytes per token; logits differ slightly from the float head's. enable = 0
// returns to the original head. Returns 0, 1 if the head is already quantized (nothing to do), -1 on error.
int qwen_model_set_lm_head_q8(qwen_model_t* model, int enable);

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