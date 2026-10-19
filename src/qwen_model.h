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

// The last error of a qwen_model call on this thread (empty if none)
const char* qwen_model_last_error(void);

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

// Tests, hyper-connection models only: run layers [l0, l1) on n tokens from position 0 (fresh state), starting from
// the residual streams res_in [n][n_hc * n_embd] instead of the embeddings; the layer callback sees each layer's
// output. No LM head. n <= qwen_model_max_batch. Returns 0 on success.
int qwen_model_run_layers(qwen_model_t* model, const int* tokens, int n, const float* res_in, int l0, int l1);

// MTP draft head (inference_set_draft_model; qwen4exp). While one is loaded, every forward pass also runs the head
// on its tokens (its own KV cache follows the sequence). qwen_model_mtp_drafts: the head's drafts for positions
// pos + 1 .. pos + k (k <= SPEC_MAX_DRAFTS) given the token at pos, which is not decoded yet and directly follows
// the last forward pass: its greedy choice, then the head again on that draft with its own output standing in
// for the model's; returns how many (0: no draft). qwen_model_mtp_draft: one (-1: none). qwen_model_mtp_reset:
// forget the head's state (a new or unloaded head).
#define SPEC_MAX_DRAFTS 3
int qwen_model_mtp_drafts(qwen_model_t* model, int token, int pos, int* drafts, int k);
int qwen_model_mtp_draft(qwen_model_t* model, int token, int pos);
void qwen_model_mtp_reset(qwen_model_t* model);
// Speculative decoding: tokens[0] at pos (the next token) and n - 1 drafts after it (2 <= n <= SPEC_MAX_DRAFTS + 1)
// in one forward pass. qwen_model_verify_logits(i): the logits after tokens[i] ([vocab], valid until the next
// forward; i = n - 1 is qwen_model_logits()), exactly what n qwen_model_forward calls give. If only tokens[0 ..
// keep - 1] are the sequence, qwen_model_rollback(keep) returns the model to the state after them (the next
// forward is then at pos + keep).
int qwen_model_forward_verify(qwen_model_t* model, const int* tokens, int n, int pos);
const float* qwen_model_verify_logits(const qwen_model_t* model, int i);
int qwen_model_rollback(qwen_model_t* model, int keep);

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
// The whole recurrent state in host memory (pinned, coarse-grained; *bytes its size, 0 / NULL without recurrent
// layers), for writing it to a file and reading it back (the KV block store). Only after qwen_model_sync: the GPU
// may still be writing it; a CPU write is seen by the next forward pass.
void* qwen_model_state_memory(qwen_model_t* model, size_t* bytes);
// Wait for the model's queued GPU work. Returns 0, -1 on error.
int qwen_model_sync(qwen_model_t* model);

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