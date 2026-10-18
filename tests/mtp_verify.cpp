// mtp_verify -- the MTP draft head (qwen4exp nextn sidecar) and speculative decoding.
//
//   mtp_verify <model.gguf> <mtp-head.gguf> [n_tokens]
//
// 1. Acceptance: plain greedy decoding, and before each token is decoded the head's draft for the token after
//    it; the fraction of drafts that equal what the model then picks. The head is only ever a guess, so this
//    measures it (the reference reports 0.71 on code to 0.94 on reasoning with its Q8_0 head) rather than
//    checking exact values; a broken head gives ~0.
// 2. Speculative greedy decoding (qwen_model_mtp_draft, qwen_model_forward_verify, qwen_model_rollback) gives
//    exactly the tokens plain greedy decoding gives, and how much faster. MTP_VERIFY_SPEC_ONLY=1: the
//    speculative run alone (profiling), against nothing.
// 3. The library: inference_generate_stream with the head loaded gives the tokens it gives without (greedy, and
//    sampling at temperature 0.8 with a fixed seed), ending at the end of the turn.
// Prompts: a coding and a reasoning one, chat-formatted.

#include "inference.h"
#include "qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int argmax(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <mtp-head.gguf> [n_tokens]\n", argv[0]);
        return 2;
    }
    const int n_gen = argc > 3 ? atoi(argv[3]) : 128;
    inference_model_t* m = inference_load_model(argv[1]);
    if (!m || inference_set_draft_model(m, argv[2]) != 0 || inference_kv_cache_init(m, 4096) != 0) {
        fprintf(stderr, "setup failed: %s\n", inference_get_last_error());
        return 1;
    }
    qwen_model_t* q = inference_get_qwen_model(m);
    const int V = inference_vocab_size(m);
    const char* prompts[] = {
        "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively, with a "
        "docstring and type hints.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>user\nA train leaves at 3 pm and travels 120 km at 80 km/h. When does it arrive? Explain "
        "step by step.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    };
    char what[256];
    for (const char* text : prompts) {
        int n = 0;
        int32_t* t = inference_tokenize(m, text, &n);
        std::vector<int> toks(t, t + n);
        inference_free_tokens(t, n);
        std::vector<float> logits((size_t)V);
        if (qwen_model_prefill(q, toks.data(), n, 0, logits.data()) != 0) {
            fprintf(stderr, "prefill: %s\n", qwen_model_last_error());
            return 1;
        }
        int pos = n, drafts = 0, hits = 0;
        std::string out;
        int next = argmax(logits.data(), V);
        std::vector<int> plain{next};
        double t0 = now();
        const bool spec_only = getenv("MTP_VERIFY_SPEC_ONLY") != nullptr;
        for (int i = 0; i < n_gen && !spec_only; ++i) {
            const int d = qwen_model_mtp_draft(q, next, pos);  // the head's guess for position pos + 1
            if (qwen_model_forward(q, next, pos, logits.data()) != 0) {
                fprintf(stderr, "forward: %s\n", qwen_model_last_error());
                return 1;
            }
            int len = 0;
            char piece[256];
            len = inference_detokenize(m, &next, 1, piece, sizeof(piece));
            if (len > 0 && len < (int)sizeof(piece)) out.append(piece, (size_t)len);
            ++pos;
            next = argmax(logits.data(), V);
            plain.push_back(next);
            if (d >= 0) {
                ++drafts;
                hits += d == next;
            }
        }
        const double t_plain = now() - t0;
        printf("  prompt of %d tokens, %d generated: %.60s...\n", n, n_gen, out.c_str());
        snprintf(what, sizeof(what), "greedy drafts accepted: %d of %d (%.2f)", hits, drafts,
                 drafts ? (double)hits / drafts : 0.0);
        if (!spec_only) check(drafts == n_gen && hits > drafts / 3, what);
        if (spec_only) plain.resize((size_t)n_gen + 1, -1);

        // Speculative: x is emitted but not decoded yet; verify [x, draft] in one pass
        if (qwen_model_prefill(q, toks.data(), n, 0, logits.data()) != 0) return 1;
        std::vector<int> spec{argmax(logits.data(), V)};
        std::vector<float> l0((size_t)V), l1((size_t)V);
        int passes = 0, accepted = 0;
        pos = n;
        t0 = now();
        while ((int)spec.size() < (int)plain.size()) {
            const int x = spec.back();
            const int d = qwen_model_mtp_draft(q, x, pos);
            const int pair[2] = {x, d};
            if (d < 0 || qwen_model_forward_verify(q, pair, pos, l0.data(), l1.data()) != 0) {
                fprintf(stderr, "speculative step at %d failed: %s\n", pos, qwen_model_last_error());
                return 1;
            }
            ++passes;
            const int y = argmax(l0.data(), V);
            spec.push_back(y);
            if (y == d) {
                ++accepted;
                spec.push_back(argmax(l1.data(), V));
                pos += 2;
            } else {
                if (qwen_model_rollback(q) != 0) {
                    fprintf(stderr, "rollback: %s\n", qwen_model_last_error());
                    return 1;
                }
                pos += 1;
            }
        }
        const double t_spec = now() - t0;
        spec.resize(plain.size());
        int first_diff = -1;
        for (size_t i = 0; i < plain.size() && first_diff < 0; ++i)
            if (plain[i] != spec[i]) first_diff = (int)i;
        snprintf(what, sizeof(what), "speculative greedy = plain greedy over %zu tokens%s (first difference at %d)",
                 plain.size(), first_diff < 0 ? "" : " -- DIFFERS", first_diff);
        if (!spec_only) check(first_diff < 0, what);
        printf("    plain %.1f tok/s; speculative %.1f tok/s (%d passes, %d drafts accepted, %.2f tokens per pass)\n",
               n_gen / t_plain, n_gen / t_spec, passes, accepted, (double)plain.size() / std::max(1, passes));
    }
    // 3. The library, with and without the head
    for (const char* text : prompts) {
        for (float temp : {0.0f, 0.8f}) {
            int n = 0;
            int32_t* t = inference_tokenize(m, text, &n);
            std::vector<int32_t> toks(t, t + n);
            inference_free_tokens(t, n);
            std::vector<int32_t> out[2];
            double secs[2];
            int nout[2];
            for (int with = 1; with >= 0; --with) {
                if (with == 0 && inference_set_draft_model(m, nullptr) != 0) return 1;
                if (with == 1 && inference_set_draft_model(m, argv[2]) != 0) return 1;
                inference_reset(m);
                inference_gen_options o = inference_gen_default_options();
                o.sampler.temperature = temp;
                o.sampler.seed = 7;
                o.max_new_tokens = 384;
                out[with].resize(384);
                inference_stop_reason why;
                const double t0 = now();
                if (inference_generate_stream(m, toks.data(), n, &o, out[with].data(), 384, &nout[with], &why) != 0) {
                    fprintf(stderr, "generate: %s\n", inference_get_last_error());
                    return 1;
                }
                secs[with] = now() - t0;
                out[with].resize((size_t)nout[with]);
            }
            snprintf(what, sizeof(what), "inference_generate_stream, temperature %.1f: %d tokens with the head = %d "
                     "without%s (%.1f vs %.1f tok/s incl. the prompt)", temp, nout[1], nout[0],
                     out[0] == out[1] ? "" : " -- DIFFER", nout[1] / secs[1], nout[0] / secs[0]);
            check(out[0] == out[1] && nout[0] > 16, what);
        }
    }
    inference_kv_cache_free(m);
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
