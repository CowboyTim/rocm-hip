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
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
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

// Streaming generation from a new sequence; collects the callback's text chunks
struct Stream {
    std::vector<int32_t> ids;
    std::vector<std::string> chunks;
    std::string text;
    int n = 0, rc = 0, stop_after = -1;  // stop_after: on_token returns 1 at that token count
    inference_stop_reason reason = INFERENCE_STOP_NONE;
};

static int on_token(void* user, int32_t token, const char* text, int len) {
    Stream* st = (Stream*)user;
    st->chunks.emplace_back(text, (size_t)len);
    st->text.append(text, (size_t)len);
    const int tokens = (int)st->chunks.size() - (token < 0 ? 1 : 0);
    return st->stop_after >= 0 && tokens >= st->stop_after;
}

static Stream stream(inference_model_t* m, const std::vector<int32_t>& prompt, int max_new, const sampler_params& sp,
                     const std::vector<const char*>& stops = {}, int stop_after = -1, bool reset = true) {
    Stream st;
    st.stop_after = stop_after;
    inference_gen_options o = inference_gen_default_options();
    o.sampler = sp;
    o.max_new_tokens = max_new;
    o.stop_strings = stops.empty() ? nullptr : stops.data();
    o.n_stop_strings = (int)stops.size();
    o.on_token = on_token;
    o.user = &st;
    st.ids.resize((size_t)max_new);
    if (reset) inference_reset(m);
    st.rc = inference_generate_stream(m, prompt.data(), (int)prompt.size(), &o, st.ids.data(), max_new, &st.n,
                                      &st.reason);
    st.ids.resize((size_t)std::min(st.n, max_new));
    return st;
}

static bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (n == 0 || i + n > s.size()) return false;
        for (size_t k = 1; k < n; ++k)
            if (((unsigned char)s[i + k] & 0xC0) != 0x80) return false;
        i += n;
    }
    return true;
}

static std::string detok(inference_model_t* m, const std::vector<int32_t>& ids) {
    const int len = inference_detokenize(m, ids.data(), (int)ids.size(), nullptr, 0);
    std::string s((size_t)std::max(len, 0) + 1, '\0');
    inference_detokenize(m, ids.data(), (int)ids.size(), &s[0], (int)s.size());
    s.resize((size_t)std::max(len, 0));
    return s;
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
    if (!m || inference_kv_cache_init(m, 1024) != 0) {
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

    // Long prompt: prefill (in one call, and in three calls of 200, 200 and the rest, so that batches start at
    // pos0 > 0 inside attention's 256-position chunks whatever the micro-batch size) must leave exactly the
    // logits that feeding the same tokens one at a time does
    {
        std::string text;
        for (int i = 0; i < 40; ++i) text += "The quick brown fox jumps over the lazy dog, again and again. ";
        int n = 0;
        int32_t* lt = inference_tokenize(m, text.c_str(), &n);
        const int V = inference_vocab_size(m);
        std::vector<float> a(V), b(V), c(V);
        std::vector<int> ids(V);
        inference_reset(m);
        const int rc1 = inference_prefill(m, lt, n);
        inference_get_logits(m, ids.data(), a.data());
        inference_reset(m);
        int rc3 = 0;
        for (int i0 = 0; i0 < n && rc3 == 0; i0 += 200) rc3 = inference_prefill(m, lt + i0, std::min(200, n - i0));
        inference_get_logits(m, ids.data(), c.data());
        inference_reset(m);
        int nx = 0, rc2 = 0;
        for (int i = 0; i < n && rc2 == 0; ++i) rc2 = inference_decode_token(m, lt[i], &nx);
        inference_get_logits(m, ids.data(), b.data());
        inference_free_tokens(lt, n);
        // The dot4 kernels give each output row the same arithmetic in a batch as alone, so prefill is
        // bit-identical to token by token. The WMMA kernels (batches of 32+ tokens) sum in a different float
        // order, and tiny differences flip int8 activation roundings that then propagate (see forward_verify):
        // then the bound is logit cosine >= 0.995 (measured on 561 tokens: dense 4B 0.9991, MoE 35B ~0.997;
        // MoE routing near-ties make it the more sensitive one) and the same top-1 token unless token by token
        // itself has a near-tie: its logits for the two choices within 0.5 (the 35B's logits differ by up to
        // ~1.5 between the two paths; on this prompt its top two are 0.16 apart).
        double d = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
        int ta = 0, tb = 0;
        for (int i = 0; i < V; ++i) {
            d = std::max(d, (double)std::fabs(a[i] - b[i]));
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
            if (a[i] > a[ta]) ta = i;
            if (b[i] > b[tb]) tb = i;
        }
        const double cs = dot / std::sqrt(na * nb);
        const bool exact = getenv("QMATVEC_NO_WMMA") != nullptr;
        char what[256];
        snprintf(what, sizeof(what),
                 "prefill of %d tokens vs token by token (batch %d): max |d| logits %.3g, cos %.7f, top-1 %d / %d "
                 "(token by token: %d leads %d by %.3g)%s", n, qwen_model_max_batch(inference_get_qwen_model(m)), d, cs,
                 ta, tb, tb, ta, b[tb] - b[ta], exact ? " (WMMA off: must be identical)" : "");
        auto same_choice = [&](int t) { return t == tb || b[tb] - b[t] < 0.5f; };
        check(rc1 == 0 && rc2 == 0 && (exact ? d == 0.0 : same_choice(ta) && cs >= 0.995), what);
        double d3 = 0.0, dot3 = 0.0, nc = 0.0;
        int tc = 0;
        for (int i = 0; i < V; ++i) {
            d3 = std::max(d3, (double)std::fabs(c[i] - b[i]));
            dot3 += (double)c[i] * b[i];
            nc += (double)c[i] * c[i];
            if (c[i] > c[tc]) tc = i;
        }
        const double cs3 = dot3 / std::sqrt(nc * nb);
        snprintf(what, sizeof(what), "prefill in calls of 200 tokens vs token by token: max |d| logits %.3g, cos %.7f, "
                 "top-1 %d / %d", d3, cs3, tc, tb);
        check(rc3 == 0 && (exact ? d3 == 0.0 : same_choice(tc) && cs3 >= 0.995), what);
    }

    // Opt-in int8 LM head: close to the float head's logits, and switching it off gives the float logits back
    // bit for bit. Heads that are already quantized are left alone (return 1).
    {
        const int32_t* pt = prompt.data();
        const int n = (int)prompt.size();
        const int V = inference_vocab_size(m);
        std::vector<float> f0(V), q(V), f1(V);
        std::vector<int> ids(V);
        auto logits_of_prompt = [&](std::vector<float>& out) {
            inference_reset(m);
            const int rc = inference_prefill(m, pt, n);
            inference_get_logits(m, ids.data(), out.data());
            return rc;
        };
        int rc = logits_of_prompt(f0);
        const int on = inference_set_lm_head_q8(m, 1);
        rc |= logits_of_prompt(q);
        const int off = inference_set_lm_head_q8(m, 0);
        rc |= logits_of_prompt(f1);
        double dot = 0.0, na = 0.0, nb = 0.0;
        int tf = 0, tq = 0;
        for (int i = 0; i < V; ++i) {
            dot += (double)f0[i] * q[i];
            na += (double)f0[i] * f0[i];
            nb += (double)q[i] * q[i];
            if (f0[i] > f0[tf]) tf = i;
            if (q[i] > q[tq]) tq = i;
        }
        const double cs = dot / std::sqrt(na * nb);
        const bool back = memcmp(f0.data(), f1.data(), (size_t)V * sizeof(float)) == 0;
        char what[200];
        if (on == 1) {
            snprintf(what, sizeof(what), "inference_set_lm_head_q8: head already quantized, logits unchanged");
            check(rc == 0 && off == 0 && back && memcmp(f0.data(), q.data(), (size_t)V * sizeof(float)) == 0, what);
        } else {
            snprintf(what, sizeof(what), "inference_set_lm_head_q8: int8 head logits cos %.7f, top-1 %d / %d; off "
                     "again: %s", cs, tq, tf, back ? "float logits restored exactly" : "logits DIFFER");
            check(rc == 0 && on == 0 && off == 0 && cs >= 0.9999 && tq == tf && back, what);
        }
    }

    // Streaming generation (inference_generate_stream)
    {
        const int SN = 48;
        const Stream a = stream(m, prompt, SN, greedy);
        const std::vector<int> ref = gen(m, prompt, SN, greedy);
        const bool eog = a.reason == INFERENCE_STOP_EOG;
        char what[256];
        snprintf(what, sizeof(what), "stream: greedy tokens equal inference_generate_ex's (%d tokens, stop reason %d)",
                 a.n, (int)a.reason);
        check(a.rc == 0 && std::vector<int>(a.ids.begin(), a.ids.end()) == ref &&
                  (a.reason == INFERENCE_STOP_MAX_TOKENS ? a.n == SN : eog),
              what);
        bool chunks_ok = true;
        for (const std::string& c : a.chunks) chunks_ok &= valid_utf8(c);
        check(a.text == detok(m, a.ids) && chunks_ok, "stream: callback text = detokenized tokens, every chunk valid UTF-8");

        // Multi-byte text: chunks still valid UTF-8, nothing lost
        int nz = 0;
        int32_t* zt = inference_tokenize(m, "\xe8\xaf\xb7\xe7\x94\xa8\xe4\xb8\xad\xe6\x96\x87\xe5\x86\x99\xe4\xb8\x80\xe9\xa6\x96\xe5\x85\xb3\xe4\xba\x8e\xe6\x9c\x88\xe4\xba\xae\xe7\x9a\x84\xe8\xaf\x97\xef\xbc\x9a", &nz);  // a Chinese prompt
        const std::vector<int32_t> zp(zt, zt + nz);
        inference_free_tokens(zt, nz);
        const Stream z = stream(m, zp, 32, greedy);
        bool z_ok = true;
        for (const std::string& c : z.chunks) z_ok &= valid_utf8(c);
        snprintf(what, sizeof(what), "stream: Chinese prompt, %d tokens, %zu bytes, every chunk valid UTF-8", z.n,
                 z.text.size());
        check(z.rc == 0 && z_ok && z.text == detok(m, z.ids), what);

        // A stop string from the middle of that greedy text: the output ends right before it
        if (a.text.size() > 40) {
            const std::string needle = a.text.substr(30, 6);
            const std::vector<const char*> stops = {"no such text \x01", needle.c_str()};
            const Stream b = stream(m, prompt, SN, greedy, stops);
            const std::string want = a.text.substr(0, a.text.find(needle));
            snprintf(what, sizeof(what), "stream: stop string \"%s\" ends the text before it (%zu of %zu bytes)",
                     needle.c_str(), b.text.size(), a.text.size());
            check(b.rc == 0 && b.reason == INFERENCE_STOP_STRING && b.text == want, what);
        }

        // Continuing: 10 tokens, then 10 more with no prompt, equal 20 in one call
        const Stream c1 = stream(m, prompt, 10, greedy);
        const Stream c2 = stream(m, {}, 10, greedy, {}, -1, false);
        const Stream c3 = stream(m, prompt, 20, greedy);
        std::vector<int32_t> joined = c1.ids;
        joined.insert(joined.end(), c2.ids.begin(), c2.ids.end());
        check(c1.rc == 0 && c2.rc == 0 && (c3.reason == INFERENCE_STOP_EOG || (joined == c3.ids && c1.text + c2.text == c3.text)),
              "stream: 10 tokens, then 10 more continuing the sequence, equal 20 in one call");

        // The callback stops it
        const Stream d = stream(m, prompt, SN, greedy, {}, 5);
        snprintf(what, sizeof(what), "stream: on_token returning 1 at token 5 stops it (%d tokens, reason %d)", d.n,
                 (int)d.reason);
        check(d.rc == 0 && d.n == 5 && d.reason == INFERENCE_STOP_CALLBACK, what);
        check(inference_token_is_eog(m, inference_vocab_size(m) + 5) == 0, "inference_token_is_eog: out of range is not EOG");

        // A chat turn ends at the end-of-turn token, which is neither emitted nor in the text
        int ne = 0;
        int32_t* et = inference_tokenize(m, "<|im_end|>", &ne);
        const bool im_end = ne == 1 && inference_token_is_eog(m, et[0]);
        inference_free_tokens(et, ne);
        int nc = 0;
        int32_t* ct = inference_tokenize(m, "<|im_start|>user\nReply with just the word OK.<|im_end|>\n"
                                            "<|im_start|>assistant\n<think>\n\n</think>\n\n", &nc);
        const std::vector<int32_t> cp(ct, ct + nc);
        inference_free_tokens(ct, nc);
        const Stream e = stream(m, cp, 64, greedy);
        snprintf(what, sizeof(what), "stream: chat turn ends at <|im_end|> (EOG token: %s; %d tokens, reason %d, \"%s\")",
                 im_end ? "yes" : "NO", e.n, (int)e.reason, e.text.c_str());
        check(im_end && e.rc == 0 && e.reason == INFERENCE_STOP_EOG && e.text.find("<|im_end|>") == std::string::npos,
              what);
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
