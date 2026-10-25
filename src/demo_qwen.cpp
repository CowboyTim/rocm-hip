// demo_qwen.cpp -- text generation with the Qwen3.5 forward pass.
// Usage: ./qwen_demo <model.gguf> "prompt text" | @prompt_file [n_gen] [--temp T] [--top-k K] [--top-p P] [--min-p P]
//                    [--repeat-penalty R] [--seed S] [--lm-head-q8] [--ctx N (default 4096)]
//                    [--draft mtp.gguf [--drafts 0..3]] [--batch N]
//
// --draft: an MTP draft head for speculative decoding; generation then runs through inference_generate_stream
// (ignoring end-of-generation tokens, as this demo does not stop at them either); --drafts: drafts per pass.
// --batch: tokens per prefill micro-batch (16 .. 8192; default 1024).
// (0: adaptive, the default).
// Greedy by default (--temp 0). QWEN_PROFILE=1 prints the time breakdown.

#include "qwen_model.h"
#include "inference.h"
#include "sampler.h"
#include "tokenizer.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model.gguf> \"prompt text\" [n_gen] [--temp T] [--top-k K] [--top-p P] [--min-p P] "
               "[--repeat-penalty R] [--seed S] [--lm-head-q8] [--ctx N] [--draft mtp.gguf [--drafts 0..3]]\n"
               "       [--batch N]\n",
               argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    // "@path": read the prompt from a file (a single argument is limited to 128 KB)
    std::string prompt_buf = argv[2];
    if (prompt_buf.size() > 1 && prompt_buf[0] == '@') {
        FILE* pf = fopen(prompt_buf.c_str() + 1, "rb");
        if (!pf) {
            fprintf(stderr, "cannot read prompt file %s\n", prompt_buf.c_str() + 1);
            return 1;
        }
        prompt_buf.clear();
        char chunk[65536];
        for (size_t got; (got = fread(chunk, 1, sizeof(chunk), pf)) > 0;) prompt_buf.append(chunk, got);
        fclose(pf);
    }
    const char* prompt = prompt_buf.c_str();
    int n_gen = 16;
    sampler_params sp = sampler_default_params();
    sp.temperature = 0.0f;  // greedy unless asked
    bool head_q8 = false;
    int n_ctx = 4096;
    const char* draft = nullptr;
    int drafts = -1;  // drafts per pass (0: adaptive; -1: the library's default)
    int batch = 0;    // tokens per prefill micro-batch (0: the library's default, 1024)
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : "0";
        if (a == "--temp") { sp.temperature = (float)atof(v); ++i; }
        else if (a == "--top-k") { sp.top_k = atoi(v); ++i; }
        else if (a == "--top-p") { sp.top_p = (float)atof(v); ++i; }
        else if (a == "--min-p") { sp.min_p = (float)atof(v); ++i; }
        else if (a == "--repeat-penalty") { sp.repeat_penalty = (float)atof(v); ++i; }
        else if (a == "--seed") { sp.seed = strtoull(v, nullptr, 10); ++i; }
        else if (a == "--lm-head-q8") head_q8 = true;
        else if (a == "--ctx") { n_ctx = atoi(v); ++i; }
        else if (a == "--draft") { draft = v; ++i; }
        else if (a == "--drafts") { drafts = atoi(v); ++i; }
        else if (a == "--batch") { batch = atoi(v); ++i; }
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
    if (batch > 0 && inference_set_prefill_batch(engine, batch) != 0) {
        fprintf(stderr, "--batch: %s\n", inference_get_last_error());
        return 1;
    }
    if (inference_kv_cache_init(engine, n_ctx) != 0) {
        fprintf(stderr, "KV cache init failed: %s\n", inference_get_last_error());
        inference_free_model(engine);
        return 1;
    }
    fprintf(stderr, "[DEMO] KV cache init succeeded\n");
    if (head_q8) {
        const int rc = inference_set_lm_head_q8(engine, 1);
        if (rc < 0) {
            fprintf(stderr, "LM head Q8_0 copy failed: %s\n", inference_get_last_error());
            inference_free_model(engine);
            return 1;
        }
        printf("LM head: %s\n", rc == 0 ? "int8 (Q8_0) copy" : "already quantized, unchanged");
    }

    if (draft) {
        if (inference_set_draft_model(engine, draft) != 0) {
            fprintf(stderr, "Draft model: %s\n", inference_get_last_error());
            inference_free_model(engine);
            return 1;
        }
        if (drafts >= 0 && inference_set_draft_tokens(engine, drafts) != 0) {
            fprintf(stderr, "--drafts: %s\n", inference_get_last_error());
            return 1;
        }
        printf("Draft head: %s (speculative decoding)\n", draft);
    }

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

    printf("Prompt tokens (%d): ", n_tokens);  // the first 32 for long prompts
    for (int i = 0; i < n_tokens && i < 32; ++i) printf("%d ", tokens[i]);
    printf("%s\n", n_tokens > 32 ? "..." : "");

    auto now = []() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    if (draft) {  // speculative decoding lives in the library's generation loop
        inference_gen_options o = inference_gen_default_options();
        o.sampler = sp;
        o.max_new_tokens = n_gen;
        o.ignore_eog = 1;
        std::vector<int32_t> out((size_t)std::max(n_gen, 1));
        int n_out = 0;
        inference_stop_reason why;
        const double t0 = now();
        const int rc = inference_generate_stream(engine, tokens, n_tokens, &o, out.data(), n_gen, &n_out, &why);
        const double t = now() - t0;
        inference_free_tokens(tokens, n_tokens);
        if (rc != 0) fprintf(stderr, "Generation failed: %s\n", inference_get_last_error());
        std::vector<char> text((size_t)n_out * 64 + 16);
        inference_detokenize(engine, out.data(), n_out, text.data(), (int)text.size());
        printf("Timing: prompt of %d tokens + %d generated in %.1f ms (%.2f tok/s including the prompt)\n", n_tokens,
               n_out, 1e3 * t, n_out / t);
        printf("Prompt + generation: %s%s\n", prompt, text.data());
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        printf("Done.\n");
        return rc == 0 ? 0 : 1;
    }

    // Sample from the logits read in place from the pinned scratch buffer
    const int vocab_size = inference_vocab_size(engine);
    sampler_t* smp = sampler_create(&sp);
    auto pick = [&]() { return sampler_sample(smp, qwen_model_logits(model), vocab_size); };
    const bool profile = getenv("QWEN_PROFILE") != nullptr;

    // Prefill; the last prompt token's logits give the first generated token
    qwen_model_profile_reset(model);
    double t0 = now();
    if (qwen_model_prefill(model, (const int*)tokens, n_tokens, 0, nullptr) != 0) {
        fprintf(stderr, "Prefill failed: %s\n", inference_get_last_error());
        inference_free_tokens(tokens, n_tokens);
        inference_kv_cache_free(engine);
        inference_free_model(engine);
        return 1;
    }
    const double t_prefill = now() - t0;
    if (profile) qwen_model_profile_print(model, "prefill");
    for (int i = 0; i < n_tokens; ++i) sampler_accept(smp, tokens[i]);
    int next = pick();
    inference_free_tokens(tokens, n_tokens);

    // Generation; stops at n_gen tokens or an end-of-generation token (EOS, or e.g. Gemma-4's <turn|>)
    qwen_model_profile_reset(model);
    t0 = now();
    int n_fwd = 0;
    tokenizer_t* tok = inference_get_tokenizer(engine);
    std::string text;
    char piece[256];
    for (int i = 0; i < n_gen; ++i) {
        tokenizer_decode(tok, &next, 1, piece, sizeof(piece));
        text += piece;
        printf("Generated token %d: %d '%s'\n", i, next, piece);
        if (tokenizer_is_eog(tok, next) || i + 1 == n_gen) break;
        if (qwen_model_forward(model, next, n_tokens + i, nullptr) != 0) {
            fprintf(stderr, "Generation failed: %s\n", inference_get_last_error());
            break;
        }
        ++n_fwd;
        sampler_accept(smp, next);
        next = pick();
    }
    sampler_free(smp);
    const double t_gen = now() - t0;
    if (profile) qwen_model_profile_print(model, "generation");
    printf("Timing%s: prefill %d tokens %.1f ms (%.2f tok/s), generation %d tokens %.1f ms (%.2f tok/s)\n",
           profile ? " (profiled, extra syncs)" : "", n_tokens, 1e3 * t_prefill, n_tokens / t_prefill, n_fwd,
           1e3 * t_gen, n_fwd > 0 ? n_fwd / t_gen : 0.0);
    printf("Prompt + generation: %s%s\n", prompt, text.c_str());

    inference_kv_cache_free(engine);
    inference_free_model(engine);
    printf("Done.\n");
    return 0;
}
