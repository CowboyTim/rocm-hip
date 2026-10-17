// sampler_verify -- unit checks for src/sampler.cpp on synthetic logits, plus its speed on a 248k vocab.
//
// Distribution checks draw 200k samples and compare frequencies with the exact expected distribution
// (tolerance 0.006, about 5 standard deviations at p = 0.5).

#include "sampler.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static std::vector<double> freq(sampler_params p, const std::vector<float>& logits, int n_draw = 200000) {
    sampler_t* s = sampler_create(&p);
    std::vector<double> f(logits.size(), 0.0);
    for (int i = 0; i < n_draw; ++i) f[sampler_sample(s, logits.data(), (int)logits.size())] += 1.0 / n_draw;
    sampler_free(s);
    return f;
}

static bool close(const std::vector<double>& f, const std::vector<double>& want, double tol = 0.006) {
    bool ok = true;
    for (size_t i = 0; i < f.size(); ++i) {
        const double w = i < want.size() ? want[i] : 0.0;
        if (std::fabs(f[i] - w) > tol) ok = false;
    }
    if (!ok) {
        printf("      got:");
        for (double v : f) printf(" %.4f", v);
        printf("\n      want:");
        for (double v : want) printf(" %.4f", v);
        printf("\n");
    }
    return ok;
}

static sampler_params plain() {  // pure temperature-1 sampling, every filter off
    sampler_params p = sampler_default_params();
    p.temperature = 1.0f;
    p.top_k = 0;
    p.top_p = 1.0f;
    p.min_p = 0.0f;
    p.repeat_penalty = 1.0f;
    p.seed = 42;
    return p;
}

int main() {
    printf("=== sampler_verify ===\n");
    // p = 0.5, 0.3, 0.2, ~0 (a fourth token far below the others)
    const std::vector<float> L = {logf(0.5f), logf(0.3f), logf(0.2f), -100.0f};

    {
        sampler_params p = plain();
        p.temperature = 0.0f;
        sampler_t* s = sampler_create(&p);
        const std::vector<float> ties = {1.0f, 3.0f, 3.0f, 2.0f};
        check(sampler_sample(s, L.data(), 4) == 0 && sampler_sample(s, ties.data(), 4) == 1,
              "temperature 0 is argmax (lowest id on ties)");
        sampler_free(s);
    }
    {
        sampler_params p = plain();
        p.top_k = 1;
        p.temperature = 5.0f;
        check(close(freq(p, L, 2000), {1.0}), "top_k 1 is argmax at any temperature");
    }
    check(close(freq(plain(), L), {0.5, 0.3, 0.2}), "temperature 1: draws follow softmax");
    {
        sampler_params p = plain();
        p.temperature = 2.0f;  // p ~ sqrt(p): 0.7071, 0.5477, 0.4472, normalized
        const double a = sqrt(0.5), b = sqrt(0.3), c = sqrt(0.2), z = a + b + c;
        check(close(freq(p, L), {a / z, b / z, c / z}), "temperature 2 flattens to p^(1/2)");
    }
    {
        sampler_params p = plain();
        p.top_k = 2;
        check(close(freq(p, L), {0.625, 0.375}), "top_k 2 keeps the two most likely, renormalized");
    }
    {
        sampler_params p = plain();
        p.top_p = 0.7f;  // 0.5 < 0.7 <= 0.8: keeps tokens 0 and 1
        check(close(freq(p, L), {0.625, 0.375}), "top_p 0.7 keeps the smallest set reaching 0.7");
        p.top_p = 0.4f;
        check(close(freq(p, L, 2000), {1.0}), "top_p below p_max keeps only the top token");
    }
    {
        sampler_params p = plain();
        p.min_p = 0.5f;  // keeps p >= 0.25: tokens 0 and 1
        check(close(freq(p, L), {0.625, 0.375}), "min_p 0.5 keeps p >= 0.5 * p_max");
    }
    {
        sampler_params p = plain();
        p.temperature = 0.0f;
        p.repeat_penalty = 2.0f;
        const std::vector<float> l2 = {3.0f, 2.0f, -1.0f};
        sampler_t* s = sampler_create(&p);
        const int t0 = sampler_sample(s, l2.data(), 3);
        sampler_accept(s, 0);  // 3 / 2 = 1.5 < 2: token 1 wins
        const int t1 = sampler_sample(s, l2.data(), 3);
        sampler_reset(s);
        const int t2 = sampler_sample(s, l2.data(), 3);
        sampler_free(s);
        check(t0 == 0 && t1 == 1 && t2 == 0, "repeat penalty demotes a recent token; reset forgets it");
    }
    {
        sampler_params p = plain();
        p.temperature = 0.0f;
        p.repeat_penalty = 2.0f;
        p.repeat_last_n = 1;
        const std::vector<float> l2 = {3.0f, 2.0f, -1.0f};
        sampler_t* s = sampler_create(&p);
        sampler_accept(s, 0);
        sampler_accept(s, 2);  // window of 1: token 0 is out of it again
        check(sampler_sample(s, l2.data(), 3) == 0, "repeat penalty only looks back repeat_last_n tokens");
        sampler_free(s);
    }
    {
        std::vector<float> big(248320);
        std::mt19937 g(1);
        std::normal_distribution<float> nd(0.0f, 3.0f);
        for (float& v : big) v = nd(g);
        sampler_params p = sampler_default_params();
        p.seed = 7;
        auto run = [&](sampler_params q) {
            sampler_t* s = sampler_create(&q);
            std::vector<int> out;
            for (int i = 0; i < 16; ++i) out.push_back(sampler_sample(s, big.data(), (int)big.size()));
            sampler_free(s);
            return out;
        };
        const std::vector<int> a = run(p), b = run(p);
        p.seed = 8;
        const std::vector<int> c = run(p);
        check(a == b && a != c, "same seed gives the same tokens, another seed differs");

        for (int mode = 0; mode < 2; ++mode) {
            sampler_params q = sampler_default_params();
            if (mode == 1) { q.top_k = 0; q.min_p = 0.0f; }  // top-p alone: the widest candidate set
            q.repeat_penalty = 1.1f;
            sampler_t* s = sampler_create(&q);
            for (int i = 0; i < 64; ++i) sampler_accept(s, i * 37);
            const int it = 50;
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < it; ++i) sampler_sample(s, big.data(), (int)big.size());
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / it;
            sampler_free(s);
            char what[128];
            snprintf(what, sizeof(what), "248k vocab, %s + penalty: %.2f ms/token (< %s)",
                     mode ? "flat logits, top_p 0.95 only (worst case)" : "default chain", ms, mode ? "10 ms" : "2 ms");
            check(ms < (mode ? 10.0 : 2.0), what);
        }
    }

    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
