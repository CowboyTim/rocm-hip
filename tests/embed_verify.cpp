// embed_verify -- embedding models (inference_embed; EmbeddingGemma) vs llama.cpp's llama-embedding.
//
//   embed_verify <model.gguf> <ref.tsv>        (run from build/co)
//
// ref.tsv is tools/gen_embed_ref.py's: per line a text (hex) and its normalized embedding. Each text's embedding
// must have cosine >= 0.999 with the reference's; and the two code snippets that mean the same must be closer to
// each other than to the weather sentence (a sanity check of the embedding itself). Prints the speed too.

#include "inference.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    g_fail += !ok;
}

static std::string unhex(const std::string& h) {
    std::string s;
    for (size_t i = 0; i + 1 < h.size(); i += 2) s += (char)strtol(h.substr(i, 2).c_str(), nullptr, 16);
    return s;
}

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) ab += (double)a[i] * b[i], aa += (double)a[i] * a[i], bb += (double)b[i] * b[i];
    return ab / std::sqrt(aa * bb + 1e-30);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <ref.tsv>\n", argv[0]); return 2; }
    inference_model_t* m = inference_load_model(argv[1]);
    if (!m || inference_kv_cache_init(m, 2048) != 0) { fprintf(stderr, "load: %s\n", inference_get_last_error()); return 1; }
    const int D = inference_embedding_dim(m);
    printf("embed_verify: %s, embedding %d\n", argv[1], D);
    check(D > 0, "an embedding model");
    std::ifstream f(argv[2]);
    std::string line;
    std::vector<std::vector<float>> got;
    int tokens = 0;
    double secs = 0;
    while (std::getline(f, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string text = unhex(line.substr(0, tab));
        std::vector<float> ref;
        std::istringstream ss(line.substr(tab + 1));
        for (float v; ss >> v;) ref.push_back(v);
        int n = 0;
        int32_t* ids = inference_tokenize(m, text.c_str(), &n);
        std::vector<float> e((size_t)D);
        const auto t0 = std::chrono::steady_clock::now();
        const int rc = inference_embed(m, ids, n, e.data(), 1);
        secs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        tokens += n;
        inference_free_tokens(ids, n);
        const double c = rc == 0 ? cosine(e, ref) : 0.0;
        char w[200];
        snprintf(w, sizeof w, "%-40.40s %4d tokens: cosine %.6f vs llama.cpp", text.c_str(), n, c);
        for (char* p = w; *p; ++p)
            if (*p == '\n') *p = ' ';
        check(rc == 0 && (int)ref.size() == D && c >= 0.999, w);
        got.push_back(e);
    }
    if (got.size() >= 4) {
        const double same = cosine(got[0], got[1]), other = cosine(got[0], got[3]);
        char w[160];
        snprintf(w, sizeof w, "two snippets that mean the same: %.3f, vs a weather sentence: %.3f", same, other);
        check(same > other + 0.1, w);
    }
    std::vector<int32_t> big(4096, 2);
    std::vector<float> e((size_t)D);
    check(inference_embed(m, big.data(), (int)big.size(), e.data(), 1) == INFERENCE_ERR_CONTEXT_FULL,
          "an input longer than the context: INFERENCE_ERR_CONTEXT_FULL");
    printf("  %d tokens in %.1f ms (%.0f tokens/s)\n", tokens, 1e3 * secs, tokens / secs);
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
