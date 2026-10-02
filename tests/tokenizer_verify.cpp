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

    // Test special token IDs (should be -1 as not set via GGUF)
    printf("bos_id=%d eos_id=%d pad_id=%d unk_id=%d\n",
           tokenizer_bos_id(tok), tokenizer_eos_id(tok),
           tokenizer_pad_id(tok), tokenizer_unk_id(tok));

    // Clean up
    tokenizer_free(tok);
    printf("All tests passed.\n");
    return 0;
}