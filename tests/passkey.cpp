// passkey: long-context retrieval. A 5-digit pass key is hidden in a long text at a given depth; after the text the
// model is asked for it and must reproduce it (greedy). Tests that attention finds one fact at any position of a
// long context, with each KV cache type (perplexity at 16k does not show that).
//
//   passkey <model.gguf> <text file> [--ctx 8192,32768] [--depth 0.1,0.5,0.9] [--kv f16|q8_0|q4_0] [--n 12]
//
// The prompt (raw text, no chat template): an instruction, the text file's tokens from its start (wikitext-2 test:
// real prose, not a repeated filler) with "The pass key is K. Remember it. K is the pass key." inserted at the depth,
// then "What is the pass key? The pass key is"; together --ctx - 4 - --n tokens. One line per trial, then "passkey: F of
// N found"; exit 0 when all are. --n: tokens generated (a model that
// reasons first needs more). Run from build/co.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "inference.h"

static std::vector<double> parse_list(const char* s) {
    std::vector<double> v;
    for (const char* p = s; *p;) {
        v.push_back(atof(p));
        const char* c = strchr(p, ',');
        if (!c) break;
        p = c + 1;
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <text file> [--ctx 8192,32768] [--depth 0.1,0.5,0.9] [--kv f16|q8_0|q4_0] [--n 12]\n",
                argv[0]);
        return 2;
    }
    std::vector<double> ctxs = {8192, 32768}, depths = {0.1, 0.5, 0.9};
    int kv = INFERENCE_KV_F16, n_gen = 12;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--ctx") ctxs = parse_list(v.c_str());
        else if (a == "--depth") depths = parse_list(v.c_str());
        else if (a == "--n") n_gen = atoi(v.c_str());
        else if (a == "--kv") kv = v == "q8_0" ? INFERENCE_KV_Q8_0 : v == "q4_0" ? INFERENCE_KV_Q4_0 : INFERENCE_KV_F16;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    int max_ctx = 0;
    for (double c : ctxs) max_ctx = std::max(max_ctx, (int)c);

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    std::string text;
    char buf[65536];
    for (size_t got; (got = fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, got);
    fclose(f);

    inference_model_t* m = inference_load_model(argv[1]);
    if (!m) { fprintf(stderr, "load: %s\n", inference_get_last_error()); return 1; }
    if ((kv != INFERENCE_KV_F16 && inference_set_kv_type(m, kv) != 0) || inference_kv_cache_init(m, max_ctx) != 0) {
        fprintf(stderr, "setup: %s\n", inference_get_last_error());
        return 1;
    }
    // Pieces tokenized apart and joined (BOS, when the tokenizer adds one, only at the start)
    const int bos = inference_bos_id(m);
    auto tok = [&](const std::string& s, bool first) {
        int n = 0;
        int32_t* t = inference_tokenize(m, s.c_str(), &n);
        std::vector<int32_t> v(t, t + n);
        inference_free_tokens(t, n);
        if (!first && !v.empty() && v[0] == bos) v.erase(v.begin());
        return v;
    };
    const std::vector<int32_t> head = tok("There is an important piece of information hidden inside a lot of irrelevant "
                                          "text. Find it and memorize it. I will quiz you about it.\n\n", true);
    const std::vector<int32_t> filler = tok(text, false);
    const std::vector<int32_t> tail = tok("\n\nWhat is the pass key? The pass key is", false);

    int found = 0, trials = 0;
    unsigned lcg = 12345;
    for (double cd : ctxs) {
        const int ctx = (int)cd;
        for (double depth : depths) {
            lcg = lcg * 1664525u + 1013904223u;
            const int key = 10000 + (int)((lcg >> 8) % 90000);
            const std::vector<int32_t> needle = tok("\nThe pass key is " + std::to_string(key) + ". Remember it. " +
                                                    std::to_string(key) + " is the pass key.\n", false);
            const int n_fill = ctx - 4 - n_gen - (int)(head.size() + needle.size() + tail.size());
            if (n_fill < 0 || n_fill > (int)filler.size()) {
                fprintf(stderr, "ctx %d: the text has %zu tokens, %d needed\n", ctx, filler.size(), n_fill);
                return 1;
            }
            const int at = (int)(depth * n_fill);
            std::vector<int32_t> p(head);
            p.insert(p.end(), filler.begin(), filler.begin() + at);
            p.insert(p.end(), needle.begin(), needle.end());
            p.insert(p.end(), filler.begin() + at, filler.begin() + n_fill);
            p.insert(p.end(), tail.begin(), tail.end());

            inference_reset(m);
            const auto t0 = std::chrono::steady_clock::now();
            if (inference_prefill(m, p.data(), (int)p.size()) != 0) {
                fprintf(stderr, "prefill: %s\n", inference_get_last_error());
                return 1;
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // greedy: the prefill's argmax, then up to n_gen - 1 more
            const int V = inference_vocab_size(m);
            std::vector<int> ids(V);
            std::vector<float> lg(V);
            inference_get_logits(m, ids.data(), lg.data());
            int next = 0;
            for (int v = 1; v < V; ++v) if (lg[v] > lg[next]) next = v;
            std::vector<int32_t> out;
            for (int i = 0; i < n_gen && !inference_token_is_eog(m, next); ++i) {
                out.push_back(next);
                if (i + 1 < n_gen && inference_decode_token(m, next, &next) != 0) break;
            }
            char reply[4096] = "";
            inference_detokenize(m, out.data(), (int)out.size(), reply, sizeof reply);
            for (char* c = reply; *c; ++c) if (*c == '\n') *c = ' ';
            const bool ok = strstr(reply, std::to_string(key).c_str()) != nullptr;
            found += ok;
            ++trials;
            printf("%s ctx %6d depth %.2f: key %d, reply \"%s\" (prefill %zu tokens, %.1f s)\n", ok ? "PASS" : "FAIL", ctx,
                   depth, key, reply, p.size(), s);
            fflush(stdout);
        }
    }
    printf("passkey: %d of %d found\n", found, trials);
    inference_free_model(m);
    return found == trials ? 0 : 1;
}
