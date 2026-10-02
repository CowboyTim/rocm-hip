#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque tokenizer handle
typedef struct tokenizer tokenizer_t;

// Create tokenizer from vocab and merges arrays (as stored in GGUF metadata)
// vocab: array of token strings, vocab_size elements
// merges: array of merge rule strings "token1 token2", merges_size elements
// Returns NULL on failure
tokenizer_t* tokenizer_create(const char** vocab, size_t vocab_size,
                              const char** merges, size_t merges_size);

// Load tokenizer directly from a GGUF model (reads vocab, merges, added_tokens from KV metadata)
// Returns NULL on failure.
tokenizer_t* tokenizer_load_from_gguf(void* gguf_model);

// Free tokenizer
void tokenizer_free(tokenizer_t* tok);

// Encode text to token IDs (BPE). Output buffer must be allocated by caller (max len = strlen(text)+2).
// Returns number of tokens written, or -1 on error.
int tokenizer_encode(tokenizer_t* tok, const char* text, int* out_ids, int max_ids);

// Decode token IDs to text. Output buffer allocated by caller.
// Returns length of written string (excluding null), or -1 on error.
int tokenizer_decode(tokenizer_t* tok, const int* ids, int n_ids, char* out_str, int max_len);

// Get vocab size
size_t tokenizer_vocab_size(tokenizer_t* tok);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_H