// demo_inference.cpp -- Phase 4: minimal demo of the inference C API.
// Usage: ./inference_demo <model.gguf> "prompt text"

#include "inference.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model.gguf> \"prompt text\"\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = argv[2];

    printf("Loading model: %s\n", model_path);
    inference_model_t* model = inference_load_model(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", inference_get_last_error());
        return 1;
    }

    printf("Model: %s (n_ctx=%d, n_layer=%d, n_embd=%d)\n",
           inference_model_name(model),
           inference_n_ctx(model), 28, 4096);  // placeholder layer/embd

    // Initialize KV cache
    if (inference_kv_cache_init(model, 4096) != 0) {
        fprintf(stderr, "KV cache init failed: %s\n", inference_get_last_error());
        inference_free_model(model);
        return 1;
    }

    // Tokenize prompt
    int n_tokens = 0;
    int32_t* tokens = inference_tokenize(model, prompt, &n_tokens);
    if (!tokens) {
        fprintf(stderr, "Tokenize failed: %s\n", inference_get_last_error());
        inference_kv_cache_free(model);
        inference_free_model(model);
        return 1;
    }

    printf("Prompt tokens (%d): ", n_tokens);
    for (int i = 0; i < n_tokens; ++i) printf("%d ", tokens[i]);
    printf("\n");

    // Run decode for each prompt token (prefill)
    for (int i = 0; i < n_tokens; ++i) {
        int next = 0;
        if (inference_decode_token(model, tokens[i], &next) != 0) {
            fprintf(stderr, "Decode failed at token %d: %s\n", i, inference_get_last_error());
            inference_free_tokens(tokens, n_tokens);
            inference_kv_cache_free(model);
            inference_free_model(model);
            return 1;
        }
    }
    inference_free_tokens(tokens, n_tokens);

    // Generate a few continuation tokens (placeholder)
    int next = 0;
    for (int i = 0; i < 5; ++i) {
        if (inference_decode_token(model, next, &next) != 0) {
            fprintf(stderr, "Generation failed: %s\n", inference_get_last_error());
            break;
        }
        printf("Generated token: %d\n", next);
    }

    inference_kv_cache_free(model);
    inference_free_model(model);
    printf("Done.\n");
    return 0;
}