// tokenizer_verify.cpp -- Phase 9 acceptance test: BPE tokenizer encode/decode round‑trip.

#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char* vocab[] = {
    "a", "b", "c", "ab", "bc", "abc", "Ġ", "Ġa", "Ġb", "Ġc"
};

static const char* merges[] = {
    "a b",
    "b c",
    "ab c",
    "a bc"
};

// Helper to set up special tokens on a tokenizer for testing
static void setup_special_tokens(tokenizer_t* tok) {
    Tokenizer* t = (Tokenizer*)tok;
    // Use existing vocab entries for special tokens if present
    // For test, we'll add special tokens to vocab if not present
    // Ensure special tokens exist in vocab
    const char* specials[] = {"<|endoftext|>", "<|pad|>", "<|unk|>", "<|bos|>"};
    for (const char* s : specials) {
        if (t->vocab_map.find(s) == t->vocab_map.end()) {
            int id = (int)t->vocab.size();
            t->vocab.push_back(s);
            t->vocab_map[s] = id;
        }
    }
    // Set special token IDs
    t->bos_id = t->vocab_map["<|bos|>"];
    t->eos_id = t->vocab_map["<|endoftext|>"];
    t->pad_id = t->vocab_map["<|pad|>"];
    t->unk_id = t->vocab_map["<|unk|>"];
    // Populate special token maps
    t->special_str_to_id["<|bos|>"] = t->bos_id;
    t->id_to_special_str[t->bos_id] = "<|bos|>";
    t->special_str_to_id["<|endoftext|>"] = t->eos_id;
    t->id_to_special_str[t->eos_id] = "<|endoftext|>";
    t->special_str_to_id["<|pad|>"] = t->pad_id;
    t->id_to_special_str[t->pad_id] = "<|pad|>";
    t->special_str_to_id["<|unk|>"] = t->unk_id;
    t->id_to_special_str[t->unk_id] = "<|unk|>";
}

// Helper to set up special tokens on a tokenizer for testing
static void setup_special_tokens(tokenizer_t* tok) {
    Tokenizer* t = (Tokenizer*)tok;
    // Ensure special tokens exist in vocab
    const char* specials[] = {"<|endoftext|>", "<|pad|>", "<|unk|>", "<|bos|>"};
    for (const char* s : specials) {
        if (t->vocab_map.find(s) == t->vocab_map.end()) {
            int id = (int)t->vocab.size();
            t->vocab.push_back(s);
            t->vocab_map[s] = id;
        }
    }
    // Set special token IDs
    t->bos_id = t->vocab_map["<|bos|>"];
    t->eos_id = t->vocab_map["<|endoftext|>"];
    t->pad_id = t->vocab_map["<|pad|>"];
    t->unk_id = t->vocab_map["<|unk|>"];
    // Populate special token maps
    t->special_str_to_id["<|bos|>"] = t->bos_id;
    t->id_to_special_str[t->bos_id] = "<|bos|>";
    t->special_str_to_id["<|endoftext|>"] = t->eos_id;
    t->id_to_special_str[t->eos_id] = "<|endoftext|>";
    t->special_str_to_id["<|pad|>"] = t->pad_id;
    t->id_to_special_str[t->pad_id] = "<|pad|>";
    t->special_str_to_id["<|unk|>"] = t->unk_id;
    t->id_to_special_str[t->unk_id] = "<|unk|>";
}

int main() {
    printf("=== Tokenizer Verification ===\n");

    // Test special token IDs (should be -1 as not set)
    printf("bos_id=%d eos_id=%d pad_id=%d unk_id=%d\n",
           tokenizer_bos_id(nullptr), tokenizer_eos_id(nullptr),
           tokenizer_pad_id(nullptr), tokenizer_unk_id(nullptr));

    // Create a real tokenizer
    tokenizer_t* tok = tokenizer_create(vocab, 10, merges, 4);
    if (!tok) {
        fprintf(stderr, "tokenizer_create failed\n");
        return 1;
    }

    // Setup special tokens for testing
    setup_special_tokens(tok);

    // Test special token getters
    printf("bos_id=%d eos_id=%d pad_id=%d unk_id=%d\n",
           tokenizer_bos_id(tok), tokenizer_eos_id(tok),
           tokenizer_pad_id(tok), tokenizer_unk_id(tok));

    // Test vocab size
    size_t vs = tokenizer_vocab_size(tok);
    printf("vocab size = %zu\n", vs);

    // Test encode of regular text
    int out_ids[16];
    int n = tokenizer_encode(tok, "abc", out_ids, 16);
    if (n < 0) {
        fprintf(stderr, "encode failed\n");
        tokenizer_free(tok);
        return 1;
    }
    printf("encode 'abc' -> %d tokens: ", n);
    for (int i = 0; i < n; ++i) printf("%d ", out_ids[i]);
    printf("\n");

    // Decode back
    char decoded[64];
    int dec_len = tokenizer_decode(tok, out_ids, n, decoded, sizeof(decoded));
    if (dec_len < 0) {
        fprintf(stderr, "decode failed\n");
        tokenizer_free(tok);
        return 1;
    }
    decoded[dec_len] = '\0';
    printf("decode -> '%s'\n", decoded);

    // Test special token encode/decode
    int special_ids[4];
    n = tokenizer_encode(tok, "<|endoftext|>", special_ids, 4);
    if (n != 1 || special_ids[0] != tokenizer_eos_id(tok)) {
        fprintf(stderr, "special token encode failed\n");
        tokenizer_free(tok);
        return 1;
    }
    printf("encode '<|endoftext|>' -> id %d\n", special_ids[0]);

    char decoded[64];
    int dec_len = tokenizer_decode(tok, special_ids, n, decoded, sizeof(decoded));
    if (dec_len < 0 || strcmp(decoded, "<|endoftext|>") != 0) {
        fprintf(stderr, "special token decode failed: got '%s'\n", decoded);
        tokenizer_free(tok);
        return 1;
    }
    printf("decode special -> '%s'\n", decoded);

    // Clean up
    tokenizer_free(tok);
    printf("All tests passed.\n");
    return 0;
}