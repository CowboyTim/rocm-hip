// generate_verify -- the engine's generation API on a real model (run from build/co).
//
//   generate_verify <model.gguf> [expected first token]
//
// Checks that inference_generate is greedy at temperature 0 and agrees with a manual
// inference_decode_token loop, that every call starts a fresh sequence (repeat calls give the same
// tokens; this caught position state leaking between calls), that seeded sampling is reproducible,
// and that inference_get_logits returns the last token's logits without running the model again.

#include "inference.h"
#include "qwen_model.h"
#include "sampler.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static std::vector<int> gen(inference_model_t* m, const std::vector<int32_t>& prompt, int n, const sampler_params& p) {
    std::vector<int32_t> out(n);
    int len = 0;
    if (inference_generate_ex(m, prompt.data(), (int)prompt.size(), n, &p, out.data(), &len) != 0) return {};
    return std::vector<int>(out.begin(), out.begin() + len);
}

static void print(const char* tag, const std::vector<int>& v) {
    printf("      %s:", tag);
    for (int t : v) printf(" %d", t);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [expected first token]\n", argv[0]);
        return 2;
    }
    inference_model_t* m = inference_load_model(argv[1]);
    if (!m || inference_kv_cache_init(m, 256) != 0) {
        fprintf(stderr, "load failed: %s\n", inference_get_last_error());
        return 1;
    }
    int n_prompt = 0;
    int32_t* toks = inference_tokenize(m, "The capital of France is", &n_prompt);
    std::vector<int32_t> prompt(toks, toks + n_prompt);
    inference_free_tokens(toks, n_prompt);
    const int N = 8;
    printf("generate_verify: %s, prompt %d tokens\n", argv[1], n_prompt);

    sampler_params greedy = sampler_default_params();
    greedy.temperature = 0.0f;
    const std::vector<int> g1 = gen(m, prompt, N, greedy);
    const std::vector<int> g2 = gen(m, prompt, N, greedy);
    print("greedy", g1);
    check((int)g1.size() == N && g1 == g2, "greedy generate is repeatable (each call starts a new sequence)");
    if (argc > 2) check(!g1.empty() && g1[0] == atoi(argv[2]), "first greedy token matches the expected one");

    // Manual loop with inference_decode_token (greedy) must give the same tokens
    std::vector<int> manual;
    inference_reset(m);
    int next = -1;
    for (int32_t t : prompt) inference_decode_token(m, t, &next);
    for (int i = 0; i < N; ++i) {
        manual.push_back(next);
        if (i + 1 < N) inference_decode_token(m, next, &next);
    }
    check(manual == g1 && inference_n_past(m) == n_prompt + N - 1,
          "inference_decode_token loop agrees with greedy generate; n_past counts tokens");

    // inference_get_logits: the last token's logits, unchanged by the call itself
    {
        const int V = inference_vocab_size(m);
        std::vector<float> a(V), b(V);
        std::vector<int> ids(V);
        inference_get_logits(m, ids.data(), a.data());
        inference_get_logits(m, ids.data(), b.data());
        int best = 0;
        for (int i = 1; i < V; ++i)
            if (a[i] > a[best]) best = i;
        check(a == b && best == next && memcmp(a.data(), qwen_model_logits(inference_get_qwen_model(m)), V * 4) == 0,
              "inference_get_logits returns the last logits without another forward pass");
    }

    sampler_params p = sampler_default_params();  // temperature 0.8, top_k 40, top_p 0.95, min_p 0.05
    p.seed = 1234;
    const std::vector<int> s1 = gen(m, prompt, N, p), s2 = gen(m, prompt, N, p);
    p.temperature = 1.5f;
    p.top_k = 0;
    p.min_p = 0.0f;
    std::vector<int> s3;
    for (uint64_t seed = 1; seed <= 4 && (s3.empty() || s3 == s1); ++seed) {
        p.seed = seed;
        s3 = gen(m, prompt, N, p);
    }
    print("sampled", s1);
    print("hot", s3);
    check(!s1.empty() && s1 == s2, "seeded sampling is reproducible");
    check(!s3.empty() && s3 != s1, "a hotter distribution or another seed gives different tokens");

    inference_kv_cache_free(m);
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
