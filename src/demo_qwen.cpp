// demo_qwen.cpp -- text generation with the Qwen3.5 forward pass.
// Usage: ./qwen_demo <model.gguf> "prompt text" [n_gen] [--temp T] [--top-k K] [--top-p P] [--min-p P]
//                    [--repeat-penalty R] [--seed S]      (run from build/co)
// Greedy by default (--temp 0). QWEN_PROFILE=1 prints the time breakdown.

#include "qwen_model.h"
#include "inference.h"
#include "sampler.h"
#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model.gguf> \"prompt text\" [n_gen] [--temp T] [--top-k K] [--top-p P] [--min-p P] "
               "[--repeat-penalty R] [--seed S]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = argv[2];
    int n_gen = 16;
    sampler_params sp = sampler_default_params();
    sp.temperature = 0.0f;  // greedy unless asked
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : "0";
        if (a == "--temp") { sp.temperature = (float)atof(v); ++i; }
        else if (a == "--top-k") { sp.top_k = atoi(v); ++i; }
        else if (a == "--top-p") { sp.top_p = (float)atof(v); ++i; }
        else if (a == "--min-p") { sp.min_p = (float)atof(v); ++i; }
        else if (a == "--repeat-penalty") { sp.repeat_penalty = (float)atof(v); ++i; }
        else if (a == "--seed") { sp.seed = strtoull(v, nullptr, 10); ++i; }
        else n_gen = atoi(argv[i]);
    }
    printf("Sampling: temp %g top_k %d top_p %g min_p %g repeat_penalty %g seed %llu\n", sp.temperature, sp.top_k,
           sp.top_p, sp.min_p, sp.repeat_penalty, (unsigned long long)sp.seed);

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

    // Sample from the logits read in place from the pinned scratch buffer
    const int vocab_size = inference_vocab_size(engine);
    sampler_t* smp = sampler_create(&sp);
    auto pick = [&]() { return sampler_sample(smp, qwen_model_logits(model), vocab_size); };
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
    for (int i = 0; i < n_tokens; ++i) sampler_accept(smp, tokens[i]);
    int next = pick();
    inference_free_tokens(tokens, n_tokens);

    // Generation; stops at n_gen tokens or EOS
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
        sampler_accept(smp, next);
        next = pick();
    }
    sampler_free(smp);
    if (profile) qwen_model_profile_print(model, "generation");
    printf("Prompt + generation: %s%s\n", prompt, text.c_str());

    inference_kv_cache_free(engine);
    inference_free_model(engine);
    printf("Done.\n");
    return 0;
}
