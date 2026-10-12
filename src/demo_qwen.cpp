// demo_qwen.cpp -- Phase 4: minimal demo of Qwen 3 forward pass.
// Usage: ./qwen_demo <model.gguf> "prompt text"

#include "qwen_model.h"
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
    inference_model_t* engine = inference_load_model(model_path);
    if (!engine) {
        fprintf(stderr, "Failed to load model: %s\n", inference_get_last_error());
        return 1;
    }

    printf("Initializing KV cache...\n");
    if (inference_kv_cache_init(engine, 4096) != 0) {
        fprintf(stderr, "KV cache init failed: %s\n", inference_get_last_error());
        inference_free_model(engine);
        return 1;
    }
    fprintf(stderr, "[DEMO] KV cache init succeeded\n");

    // Get the Qwen model created during kv_cache_init (already configured with embedding table)
    qwen_model_t* model = inference_get_qwen_model(engine);
    if (!model) {
        fprintf(stderr, "Failed to get Qwen model\n");
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }

    // Tokenize prompt
    int n_tokens = 0;
    int32_t* tokens = inference_tokenize(engine, prompt, &n_tokens);
    if (!tokens) {
        fprintf(stderr, "Tokenize failed: %s\n", inference_get_last_error());
        qwen_model_free(model);
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }

    printf("Prompt tokens (%d): ", n_tokens);
    for (int i = 0; i < n_tokens; ++i) printf("%d ", tokens[i]);
    printf("\n");

    // Prefill
    int vocab_size = inference_vocab_size(engine);
    for (int i = 0; i < n_tokens; ++i) {
        float* logits = new float[vocab_size];
        if (qwen_model_forward(model, tokens[i], i, logits) != 0) {
            fprintf(stderr, "Forward failed at token %d: %s\n", i, inference_get_last_error());
            delete[] logits;
            inference_free_tokens(tokens, n_tokens);
            qwen_model_free(model);
            inference_kv_cache_free(engine);
            inference_free_model(engine);
            return 1;
        }
        // Greedy next token (for prefill, we just run forward to populate KV cache)
        delete[] logits;
    }
    inference_free_tokens(tokens, n_tokens);

    // Generate a few continuation tokens
    int next = 0;
    for (int i = 0; i < 5; ++i) {
        float* logits = new float[vocab_size];
        if (qwen_model_forward(model, next, n_tokens + i, logits) != 0) {
            fprintf(stderr, "Generation failed: %s\n", inference_get_last_error());
            delete[] logits;
            break;
        }
        // Greedy sampling
        next = 0;
        for (int v = 1; v < vocab_size; ++v) {
            if (logits[v] > logits[next]) next = v;
        }
        printf("Generated token %d: %d\n", i, next);
        delete[] logits;
    }

    inference_kv_cache_free(engine);
    inference_free_model(engine);
    printf("Done.\n");
    return 0;
}
