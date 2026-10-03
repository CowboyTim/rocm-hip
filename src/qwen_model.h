// qwen_model.h -- Phase 4: Qwen 3 dense transformer model host API.

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

// Free model resources
void qwen_model_free(qwen_model_t* model);

// Run single-token forward pass
// token_id: input token
// pos: absolute position in sequence (for RoPE + KV cache indexing)
// logits: output buffer [vocab_size] FP32 (pre-allocated by caller)
// Returns 0 on success, negative on error
int qwen_model_forward(qwen_model_t* model, int token_id, int pos, float* logits);

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