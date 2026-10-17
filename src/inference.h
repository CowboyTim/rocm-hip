// inference.h -- Phase 4/10/11: C API for zero-copy ROCmFP4 inference engine on Strix Halo (gfx1151).
// All functions are extern "C" with opaque handles. No C++ or HIP types exposed.
// Designed for eventual Perl XS CPAN module consumption.

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stddef.h>
#include <stdint.h>
#include "tokenizer.h"
#include "sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error codes (0 = success, negative = error)
typedef enum {
    INFERENCE_OK = 0,
    INFERENCE_ERR_INVALID_ARG = -1,
    INFERENCE_ERR_OUT_OF_MEMORY = -2,
    INFERENCE_ERR_MODEL_LOAD = -3,
    INFERENCE_ERR_KV_CACHE = -4,
    INFERENCE_ERR_TOKENIZER = -5,
    INFERENCE_ERR_KERNEL_LAUNCH = -6,
    INFERENCE_ERR_INVALID_STATE = -7,
    INFERENCE_ERR_NOT_IMPLEMENTED = -8,
    INFERENCE_ERR_CONTEXT_FULL = -9,
} inference_error_t;

// Convert error code to human readable string.
const char* inference_strerror(inference_error_t err);

// Opaque handle for the inference engine model
typedef struct inference_model inference_model_t;

// Model loading: mmap() the GGUF file, parse metadata, pin weight pages via hipHostRegister.
// Returns NULL on failure (check inference_get_last_error()).
inference_model_t* inference_load_model(const char* gguf_path);

// Reference counting for shared ownership
void inference_model_retain(inference_model_t* model);
void inference_model_release(inference_model_t* model);

// Free model: unpin all pages, unmap GGUF file, free metadata structures.
void inference_free_model(inference_model_t* model);

// Tokenization (CPU-side, uses tokenizer from GGUF metadata)
// Returns allocated int32 array of token IDs (length written to out_len), or NULL on failure.
// Caller must free with inference_free_tokens().
int32_t* inference_tokenize(inference_model_t* model, const char* text, int* out_len);
void inference_free_tokens(int32_t* tokens, int len);

// KV cache initialization: allocates /dev/shm buffer for all layers, pins with hipHostRegister.
// n_ctx: maximum sequence length (context window).
// Returns 0 on success, negative error code on failure.
int inference_kv_cache_init(inference_model_t* model, int n_ctx);
void inference_kv_cache_free(inference_model_t* model);

// Inference control
// Seed for inference_generate's sampler (default 0).
void inference_set_seed(inference_model_t* model, unsigned int seed);

// The model holds one sequence: tokens are decoded at positions 0, 1, 2, ... (n_past counts them).
// inference_reset starts a new sequence (position 0, which also clears the DeltaNet recurrent state).
void inference_reset(inference_model_t* model);
int inference_n_past(const inference_model_t* model);

// Single-token decode step: runs one forward pass for token_id at position n_past, appends it to the
// KV cache and recurrent state, and returns the greedy next token. Use inference_get_logits (or
// qwen_model_logits) to sample differently.
// Returns 0 on success, INFERENCE_ERR_CONTEXT_FULL when n_past reached n_ctx, negative on error.
int inference_decode_token(inference_model_t* model, int token_id, int* next_token_out);

// Batched forward pass (prefill or decode) for a batch of sequences.
// tokens: array of token IDs of length batch_size * seq_len (row-major).
// positions: array of sequence start positions in KV cache (length batch_size).
// seq_lens: array of actual sequence lengths (length batch_size).
// logits_out: optional output buffer for logits, size batch_size * vocab_size floats. If NULL, logits are not returned.
// Returns 0 on success, negative on error.
int inference_forward_batch(inference_model_t* model,
                            const int32_t* tokens,
                            const int* positions,
                            const int* seq_lens,
                            int batch_size,
                            int seq_len,
                            float* logits_out);

// Generation: starts a new sequence, prefills prompt_ids, then samples up to max_new_tokens tokens with
// the sampler chain of sampler.h. Stops early at EOS (not written to out_ids) or when the context is full.
// out_ids: buffer of max_new_tokens; out_len: number of tokens generated.
// Returns 0 on success, negative on error.
int inference_generate_ex(inference_model_t* model, const int32_t* prompt_ids, int prompt_len, int max_new_tokens,
                          const sampler_params* params, int32_t* out_ids, int* out_len);

// Same with temperature / top_k / top_p only (min_p and repeat penalty off) and the inference_set_seed seed.
// temperature <= 0 is greedy.
int inference_generate(inference_model_t* model,
                       const int32_t* prompt_ids,
                       int prompt_len,
                       int max_new_tokens,
                       float temperature,
                       int top_k,
                       float top_p,
                       int32_t* out_ids,
                       int* out_len);

// Forward declaration of Qwen model handle (defined in qwen_model.h)
struct QwenModel;
typedef struct QwenModel qwen_model_t;

// Get the internal Qwen model handle (created during kv_cache_init)
// Use this instead of creating a new qwen model — ensures proper initialization
qwen_model_t* inference_get_qwen_model(inference_model_t* model);

// Copy of the logits of the last decoded token (no forward pass), for custom sampling.
// vocab_ids: pre-allocated int32[vocab_size], filled with 0..vocab_size-1
// logits: pre-allocated float[vocab_size]
// Returns 0 on success, negative on error (e.g. nothing decoded yet).
int inference_get_logits(inference_model_t* model, int* vocab_ids, float* logits);

// Model introspection
int inference_n_ctx(const inference_model_t* model);
int inference_batch_size(const inference_model_t* model);  // always 1 for now
const char* inference_model_name(const inference_model_t* model);
int inference_vocab_size(const inference_model_t* model);

// Direct tensor access for debugging/analysis
// Returns pointer to tensor data in mapped memory (valid while model is alive)
// out_bytes: receives tensor size in bytes
const void* inference_get_tensor_ptr(const inference_model_t* model, const char* tensor_name, size_t* out_bytes);

// Tokenizer access for decoding tokens back to text
// Returns opaque tokenizer handle (valid while model is alive)
// Can be passed to tokenizer_decode() for text output
tokenizer_t* inference_get_tokenizer(const inference_model_t* model);

// Error handling: returns last error message (thread-local static buffer), or NULL if none.
const char* inference_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H