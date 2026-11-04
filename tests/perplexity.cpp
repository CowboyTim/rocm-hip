// perplexity: a model's perplexity on a text, computed as llama.cpp's llama-perplexity does, so the two can be
// compared (and kept as a regression baseline: tests/regress.sh).
//
//   perplexity <model.gguf> <text file> [--ctx 512] [--chunks N] [--kv f16|q8_0|q4_0] [--batch N]
//
// The text is tokenized once and cut into chunks of --ctx tokens (the first replaced by BOS when the model's
// tokenizer adds one). Each chunk is a new sequence, prefilled with the logits of every position
// (inference_prefill_logits); the second half of its tokens is scored, each given at least --ctx / 2 tokens before
// it: nll += -log softmax(logits[i])[tokens[i + 1]] for i = ctx / 2 .. ctx - 2 (the scored half prefilled in pieces,
// with logits; the first half without). Prints the running perplexity per
// chunk and, last, "perplexity: P +/- E (N tokens, C chunks)" (E: the standard error of the mean log-likelihood,
// carried through exp). Run from build/co.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "inference.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <text file> [--ctx 512] [--chunks N] [--kv f16|q8_0|q4_0] [--batch N]\n",
                argv[0]);
        return 2;
    }
    int n_ctx = 512, max_chunks = 0, batch = 0, kv = INFERENCE_KV_F16;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--ctx") n_ctx = atoi(v.c_str());
        else if (a == "--chunks") max_chunks = atoi(v.c_str());
        else if (a == "--batch") batch = atoi(v.c_str());
        else if (a == "--kv") kv = v == "q8_0" ? INFERENCE_KV_Q8_0 : v == "q4_0" ? INFERENCE_KV_Q4_0 : INFERENCE_KV_F16;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (n_ctx < 16) { fprintf(stderr, "--ctx must be at least 16\n"); return 2; }

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    std::string text;
    char buf[65536];
    for (size_t got; (got = fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, got);
    fclose(f);

    inference_model_t* m = inference_load_model(argv[1]);
    if (!m) { fprintf(stderr, "load: %s\n", inference_get_last_error()); return 1; }
    if ((kv != INFERENCE_KV_F16 && inference_set_kv_type(m, kv) != 0) ||
        // the logits of a whole prefill batch are kept (batch x vocab floats): 256 rows unless asked
        inference_set_prefill_batch(m, batch > 0 ? batch : std::min(n_ctx, 256)) != 0 ||
        inference_kv_cache_init(m, n_ctx) != 0) {
        fprintf(stderr, "setup: %s\n", inference_get_last_error());
        return 1;
    }
    int n_tok = 0;
    int32_t* toks = inference_tokenize(m, text.c_str(), &n_tok);
    if (!toks || n_tok < 2 * n_ctx) {
        fprintf(stderr, "the text has %d tokens: at least 2 chunks of %d needed\n", n_tok, n_ctx);
        return 1;
    }
    const int bos = inference_bos_id(m);
    const bool add_bos = bos >= 0 && toks[0] == bos;  // the tokenizer prepends it: so does every chunk
    int n_chunks = n_tok / n_ctx;
    if (max_chunks > 0 && max_chunks < n_chunks) n_chunks = max_chunks;
    const int V = inference_vocab_size(m), first = n_ctx / 2;
    printf("perplexity: %s, %d tokens, %d chunks of %d (scoring %d each), vocab %d%s\n", argv[1], n_tok, n_chunks,
           n_ctx, n_ctx - 1 - first, V, add_bos ? ", BOS per chunk" : "");

    // The first half of a chunk is prefilled without logits, the scored half in pieces of PIECE positions with them,
    // so that the logits held stay at PIECE x vocab floats at any --ctx (16k x 248k would be 16 GB)
    const int PIECE = 512;
    std::vector<float> logits((size_t)PIECE * V);
    std::vector<int32_t> chunk(n_ctx);
    double nll = 0.0, nll2 = 0.0;
    long long count = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < n_chunks; ++c) {
        memcpy(chunk.data(), toks + (size_t)c * n_ctx, sizeof(int32_t) * n_ctx);
        if (add_bos) chunk[0] = bos;
        inference_reset(m);
        if (inference_prefill(m, chunk.data(), first) != 0) {
            fprintf(stderr, "chunk %d: %s\n", c, inference_get_last_error());
            return 1;
        }
        for (int p0 = first; p0 < n_ctx - 1; p0 += PIECE) {  // (token n_ctx - 1 predicts nothing scored)
            const int np = std::min(PIECE, n_ctx - 1 - p0);
            if (inference_prefill_logits(m, chunk.data() + p0, np, logits.data()) != 0) {
                fprintf(stderr, "chunk %d: %s\n", c, inference_get_last_error());
                return 1;
            }
            // (rows on all CPU cores: the softmax over the vocabulary is the slow part)
            double cn = 0.0, cn2 = 0.0;
#pragma omp parallel for reduction(+ : cn, cn2) schedule(static)
            for (int i = 0; i < np; ++i) {  // logits row i: after token p0 + i, scoring token p0 + i + 1
                const float* lg = &logits[(size_t)i * V];
                float mx = lg[0];
                for (int v = 1; v < V; ++v) mx = std::max(mx, lg[v]);
                double sum = 0.0;
                for (int v = 0; v < V; ++v) sum += exp((double)(lg[v] - mx));
                const double l = -((double)(lg[chunk[p0 + i + 1]] - mx) - log(sum));
                cn += l;
                cn2 += l * l;
            }
            nll += cn;
            nll2 += cn2;
            count += np;
        }
        printf("[%d]%.4f%s", c + 1, exp(nll / count), (c + 1) % 10 == 0 || c + 1 == n_chunks ? "\n" : " ");
        fflush(stdout);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double mean = nll / count, var = nll2 / count - mean * mean;
    const double err = sqrt(var > 0.0 ? var / (count - 1) : 0.0);
    printf("perplexity: %.4f +/- %.4f (%lld tokens, %d chunks, %.1f s, %.0f tok/s)\n", exp(mean), exp(mean) * err, count,
           n_chunks, s, (double)n_chunks * n_ctx / s);
    inference_free_tokens(toks, n_tok);
    inference_free_model(m);
    return 0;
}
