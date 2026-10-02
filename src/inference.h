// inference.h -- Phase 4: C API for zero-copy ROCmFP4 inference engine on Strix Halo (gfx1151).
// All functions are extern "C" with opaque handles. No C++ or HIP types exposed.
// Designed for eventual Perl XS CPAN module consumption.

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the inference engine model
typedef struct inference_model inference_model_t;

// Model loading: mmap() the GGUF file, parse metadata, pin weight pages via hipHostRegister.
// Returns NULL on failure (check inference_get_last_error()).
inference_model_t* inference_load_model(const char* gguf_path);

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
void inference_set_seed(inference_model_t* model, unsigned int seed);

// Single-token decode step: runs one forward pass for the given token_id,
// appends to KV cache, returns next predicted token ID (greedy).
// Returns 0 on success, negative on error. next_token_out is only valid on success.
int inference_decode_token(inference_model_t* model, int token_id, int* next_token_out);

// Logits access for sampling (temperature, top-k, top-p)
// vocab_ids: pre-allocated int32[vocab_size] array
// logits: pre-allocated float[vocab_size] array
// Returns 0 on success, negative on error.
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

// Error handling: returns last error message (thread-local static buffer), or NULL if none.
const char* inference_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H