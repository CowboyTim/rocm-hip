// demo_qwen.cpp -- greedy generation with the Qwen3.5 forward pass.
// Usage: ./qwen_demo <model.gguf> "prompt text" [n_gen]   (run from build/co)

#include "qwen_model.h"
#include "inference.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model.gguf> \"prompt text\" [n_gen]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = argv[2];
    const int n_gen = argc > 3 ? atoi(argv[3]) : 16;

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
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }

    printf("Prompt tokens (%d): ", n_tokens);
    for (int i = 0; i < n_tokens; ++i) printf("%d ", tokens[i]);
    printf("\n");

    // Greedy pick over the logits, read in place from the pinned scratch buffer
    const int vocab_size = inference_vocab_size(engine);
    auto argmax = [&]() {
        const float* logits = qwen_model_logits(model);
        int best = 0;
        for (int v = 1; v < vocab_size; ++v)
            if (logits[v] > logits[best]) best = v;
        return best;
    };
    const bool profile = getenv("QWEN_PROFILE") != nullptr;

    // Prefill; the last prompt token's logits give the first generated token
    qwen_model_profile_reset(model);
    for (int i = 0; i < n_tokens; ++i) {
        if (qwen_model_forward(model, tokens[i], i, nullptr) != 0) {
            fprintf(stderr, "Forward failed at token %d: %s\n", i, inference_get_last_error());
            inference_free_tokens(tokens, n_tokens);
            inference_kv_cache_free(engine);
            inference_free_model(engine);
            return 1;
        }
    }
    if (profile) qwen_model_profile_print(model, "prefill");
    int next = argmax();
    inference_free_tokens(tokens, n_tokens);

    // Greedy generation; stops at n_gen tokens or EOS
    qwen_model_profile_reset(model);
    tokenizer_t* tok = inference_get_tokenizer(engine);
    const int eos = tokenizer_eos_id(tok);
    std::string text;
    char piece[256];
    for (int i = 0; i < n_gen; ++i) {
        tokenizer_decode(tok, &next, 1, piece, sizeof(piece));
        text += piece;
        printf("Generated token %d: %d '%s'\n", i, next, piece);
        if (next == eos || i + 1 == n_gen) break;
        if (qwen_model_forward(model, next, n_tokens + i, nullptr) != 0) {
            fprintf(stderr, "Generation failed: %s\n", inference_get_last_error());
            break;
        }
        next = argmax();
    }
    if (profile) qwen_model_profile_print(model, "generation");
    printf("Prompt + generation: %s%s\n", prompt, text.c_str());

    inference_kv_cache_free(engine);
    inference_free_model(engine);
    printf("Done.\n");
    return 0;
}
