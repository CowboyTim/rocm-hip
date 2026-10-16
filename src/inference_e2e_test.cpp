// End-to-end inference test: loads real Qwen3 GGUF weights, runs full forward pass.
// Uses the complete inference engine (same code path as qwen_demo) with zero-copy mmap loading.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "inference.h"
#include "tokenizer.h"
#include "qwen_model.h"

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP ERROR: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model.gguf> \"prompt text\" [num_tokens]\n", argv[0]);
        printf("\nEnd-to-end inference test for Qwen3 models on ROCm.\n");
        printf("Loads real GGUF weights via zero-copy mmap, runs full forward pass.\n");
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = argv[2];
    int num_gen_tokens = (argc > 3) ? atoi(argv[3]) : 5;

    printf("=== ROCmFP4 Inference E2E Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Prompt: \"%s\"\n", prompt);
    printf("Tokens to generate: %d\n\n", num_gen_tokens);

    // --- Step 1: Load model from GGUF via zero-copy mmap ---
    fprintf(stderr, "[Step 1] Loading model via inference_load_model()...\n"); fflush(stderr);

    inference_model_t* engine = inference_load_model(model_path);
    if (!engine) {
        fprintf(stderr, "FAILED: Could not load model: %s\n", inference_get_last_error());
        return 1;
    }

    // Print loaded metadata
    printf("[METADATA]\n");
    printf("  Architecture:     %s\n", inference_model_name(engine));
    printf("  Vocab size:       %d\n", inference_vocab_size(engine));
    printf("  Context length:   %d\n", inference_n_ctx(engine));
    fprintf(stderr, "[Step 1] Model loaded successfully.\n\n"); fflush(stderr);

    // --- Step 2: Initialize KV cache ---
    fprintf(stderr, "[Step 2] Initializing KV cache (n_ctx=4096)...\n"); fflush(stderr);

    if (inference_kv_cache_init(engine, 4096) != 0) {
        fprintf(stderr, "FAILED: KV cache init failed: %s\n", inference_get_last_error());
        inference_free_model(engine);
        return 1;
    }
    fprintf(stderr, "[Step 2] KV cache initialized.\n\n"); fflush(stderr);

    // --- Step 3: Get Qwen model handle and tokenize prompt ---
    qwen_model_t* model = inference_get_qwen_model(engine);
    if (!model) {
        fprintf(stderr, "FAILED: Could not get Qwen model handle\n");
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }

    int n_tokens = 0;
    int32_t* tokens = inference_tokenize(engine, prompt, &n_tokens);
    if (!tokens) {
        fprintf(stderr, "FAILED: Tokenize failed: %s\n", inference_get_last_error());
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }

    printf("[TOKENIZATION]\n");
    printf("  Prompt tokens (%d): ", n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        printf("%d ", tokens[i]);
    }
    printf("\n");

    // Decode back to text for verification
    char decoded[4096];
    tokenizer_decode(inference_get_tokenizer(engine), tokens, n_tokens, decoded, sizeof(decoded));
    printf("  Decoded back: \"%s\"\n", decoded);
    printf("\n");

    // --- Step 4: Prefill - run forward pass for each prompt token ---
    printf("[PREFILL]\n");
    fprintf(stderr, "[Step 4] Running prefill (%d tokens)...\n", n_tokens); fflush(stderr);

    bool success = true;
    int vocab_size = inference_vocab_size(engine);

    for (int i = 0; i < n_tokens; ++i) {
        float* logits = new float[vocab_size];

        if (qwen_model_forward(model, tokens[i], i, logits) != 0) {
            fprintf(stderr, "\nFAILED: Forward pass failed at prefill token %d\n", i); fflush(stderr);
            delete[] logits;
            success = false;
            break;
        }

        // Print top-5 tokens by logit value as diagnostic
        if (i == 0 || i == n_tokens - 1) {
            std::vector<std::pair<float,int>> top5;
            for (int v = 0; v < vocab_size && (int)top5.size() < 5; ++v) {
                top5.emplace_back(logits[v], v);
            }
            // Partial sort to find top 5
            for (int j = 0; j < 5 && j < (int)top5.size(); ++j) {
                int max_idx = j;
                for (int k = j + 1; k < (int)top5.size(); ++k) {
                    if (top5[k].first > top5[max_idx].first) max_idx = k;
                }
                std::swap(top5[j], top5[max_idx]);
            }

            printf("  Token %d logits (top-5): ", i);
            for (int j = 0; j < 5 && j < (int)top5.size(); ++j) {
                char tok_buf[256];
                tokenizer_decode(inference_get_tokenizer(engine), &top5[j].second, 1, tok_buf, sizeof(tok_buf));
                printf("[%d %.3f \"%s\"] ", top5[j].second, top5[j].first, tok_buf);
            }
            printf("\n");
        }

        delete[] logits;
    }

    if (!success) {
        inference_free_tokens(tokens, n_tokens);
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }
    fprintf(stderr, "[Step 4] Prefill complete.\n\n"); fflush(stderr);

    // --- Step 5: Generate tokens iteratively ---
    printf("[GENERATION]\n");
    fprintf(stderr, "[Step 5] Generating %d tokens...\n", num_gen_tokens); fflush(stderr);

    int next_token = tokens[n_tokens - 1]; // Start from last prefill token

    for (int g = 0; g < num_gen_tokens; ++g) {
        float* logits = new float[vocab_size];

        int pos = n_tokens + g;
        if (qwen_model_forward(model, next_token, pos, logits) != 0) {
            fprintf(stderr, "\nFAILED: Generation failed at step %d\n", g); fflush(stderr);
            success = false;
            break;
        }

        // Greedy sampling: pick highest logit
        int best = 0;
        float best_logit = logits[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best = v;
            }
        }

        char tok_buf[256];
        tokenizer_decode(inference_get_tokenizer(engine), &best, 1, tok_buf, sizeof(tok_buf));

        printf("  Gen %2d: token=%d (%.4f) \"%s\"\n", g, best, best_logit, tok_buf);

        delete[] logits;

        if (best == tokenizer_eos_id(inference_get_tokenizer(engine))) {
            printf("  [EOS detected, stopping generation]\n");
            break;
        }

        next_token = best;
    }

    if (!success) {
        inference_free_tokens(tokens, n_tokens);
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }

    fprintf(stderr, "[Step 5] Generation complete.\n\n"); fflush(stderr);

    // --- Cleanup ---
    printf("[DONE]\n");
    printf("Full sequence (%d tokens):\n", n_tokens + num_gen_tokens);

    // Reconstruct full token list for decode

    printf("  Prompt: %d tokens\n", n_tokens);
    printf("  Generated: %d tokens\n", num_gen_tokens);

    inference_free_tokens(tokens, n_tokens);
    inference_kv_cache_free(engine);
    inference_free_model(engine);

    printf("\n=== E2E Test Complete ===\n");
    return 0;
}
