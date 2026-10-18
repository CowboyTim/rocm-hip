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

// The same, but returns the full text length (it wrote min(length, max_len - 1) bytes plus a NUL), so a
// caller can retry with a bigger buffer; max_len 0 only measures. Returns -1 on error.
int tokenizer_detokenize(tokenizer_t* tok, const int* ids, int n_ids, char* out_str, int max_len);

// Get vocab size
size_t tokenizer_vocab_size(tokenizer_t* tok);

// Get special token IDs (return -1 if not set)
int tokenizer_bos_id(tokenizer_t* tok);
int tokenizer_eos_id(tokenizer_t* tok);
int tokenizer_pad_id(tokenizer_t* tok);
int tokenizer_unk_id(tokenizer_t* tok);

// 1 if id ends generation: the GGUF's EOS / EOT / EOM and FIM pad / rep / sep ids, and tokens whose text is an
// end-of-turn marker such as <|im_end|>, <|endoftext|> or <end_of_turn> (llama.cpp's rule)
int tokenizer_is_eog(tokenizer_t* tok, int id);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_H