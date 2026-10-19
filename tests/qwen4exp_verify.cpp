// qwen4exp_verify -- the hyper-connection architecture (qwen4exp, Qwen3.8-Flash-Next) vs the llama.cpp reference.
//
//   qwen4exp_verify <model.gguf> <ref_dir>
//
// ref_dir is a `ref_dump acts` directory of the same model and prompt. The release model (98.5 GiB) does not fit
// on this machine, so the model is normally tools/gen_qwen4exp_tiny.py's small random-weight GGUF: the release's
// tensor names, layer pattern, quant types and tokenizer at small widths.
//
// Checks against the reference:
//  - each layer alone, from the reference's input to it (teacher forcing, qwen_model_run_layers): the cosine of
//    the layer's update to the streams, per position. At least 95% of them >= 0.999 and none < 0.95. This is the
//    strict check: a wrong formula shows at every position of every layer of its kind.
//  - the whole model: the residual streams after every layer (l_last-N, all n_hc streams; worst cosine > 0.9) and
//    the last token's logits (cosine > 0.98, the same argmax or a near-tie). Errors compound here: the reference
//    quantizes activations to Q8 with other rounding, and with 512 experts (the release, top 10) many tokens
//    have a near-tie between the 10th and 11th expert, which that noise can swap. On the release model 559 of
//    564 single-layer updates match; the 5 others are at positions with the smallest router gaps, and the
//    streams carry such a swap on: worst residual cosine ~0.95, logits 0.99 (2026-10-18). (Matmuls on the
//    float path, which does not quantize its input like the reference, made 70 of them differ.)
// Then the engine against itself: token-by-token decoding and a prefill split in two (PLE n-gram history and
// conv state carried across calls) give the prompt-at-once logits: exactly with QMATVEC_NO_WMMA=1; else
// cosine > 0.98, the bar the logits meet against the reference (prefill attention and matmuls use WMMA, which
// sums in another order, and routing near-ties amplify that the same way: on the release model's 12-token
// prompt each path is at ~0.995 from the reference and 0.989 from the other, 2026-10-19).

#include "inference.h"
#include "qwen_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string g_dir;
static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

// The last tensor of index.tsv with this name (a name can repeat: the graph names each layer's mixer
// outputs the same), as floats
static bool ref(const std::string& name, std::vector<float>& out) {
    std::ifstream f(g_dir + "/index.tsv");
    std::string line, file;
    long long n = 0;
    std::getline(f, line);
    while (std::getline(f, line)) {
        std::vector<std::string> c;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) c.push_back(cell);
        if (c.size() < 8 || c[1] != name) continue;
        n = std::stoll(c[3]) * std::stoll(c[4]) * std::stoll(c[5]) * std::stoll(c[6]);
        file = c[7];
    }
    if (file.empty()) { fprintf(stderr, "reference tensor %s not found\n", name.c_str()); return false; }
    out.resize((size_t)n);
    FILE* fp = fopen((g_dir + "/" + file).c_str(), "rb");
    if (!fp) return false;
    const size_t got = fread(out.data(), sizeof(float), out.size(), fp);
    fclose(fp);
    return got == out.size();
}

static double cosine(const float* a, const float* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) ab += (double)a[i] * b[i], aa += (double)a[i] * a[i], bb += (double)b[i] * b[i];
    return ab / (sqrt(aa * bb) + 1e-30);
}

static double max_rel(const float* a, const float* b, size_t n) {
    double m = 0, s = 0;
    for (size_t i = 0; i < n; ++i) m = fmax(m, fabs((double)a[i] - b[i])), s = fmax(s, fabs((double)b[i]));
    return m / (s + 1e-30);
}

static int argmax(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

struct Capture {
    int hcd = 0;
    std::vector<std::vector<float>> layer;  // [layer][pos * hcd]
};

static void on_layer(int l, int pos, const float* x, int n, void* user) {
    Capture* c = (Capture*)user;
    if ((int)c->layer.size() <= l) c->layer.resize((size_t)l + 1);
    std::vector<float>& v = c->layer[l];
    if (v.size() < (size_t)(pos + 1) * n) v.resize((size_t)(pos + 1) * n);
    memcpy(v.data() + (size_t)pos * n, x, sizeof(float) * n);
    c->hcd = n;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <ref_dir>\n", argv[0]);
        return 2;
    }
    g_dir = argv[2];
    const bool exact = getenv("QMATVEC_NO_WMMA") != nullptr;
    std::vector<int> toks;
    {
        std::ifstream f(g_dir + "/tokens.txt");
        for (int t; f >> t;) toks.push_back(t);
    }
    int ref_next = -1;
    {
        std::ifstream f(g_dir + "/next_token.txt");
        f >> ref_next;
    }
    inference_model_t* m = inference_load_model(argv[1]);
    if (!m || inference_kv_cache_init(m, 512) != 0) {
        fprintf(stderr, "setup failed: %s\n", inference_get_last_error());
        return 1;
    }
    qwen_model_t* q = inference_get_qwen_model(m);
    const int V = inference_vocab_size(m), n = (int)toks.size();
    printf("qwen4exp_verify: %d prompt tokens, vocab %d%s\n", n, V, exact ? " (QMATVEC_NO_WMMA: exact checks)" : "");
    char what[256];

    // 1. Prompt at once vs the reference, layer by layer
    Capture cap;
    qwen_model_set_layer_callback(q, on_layer, &cap);
    std::vector<float> logits((size_t)V);
    if (qwen_model_prefill(q, toks.data(), n, 0, logits.data()) != 0) {
        fprintf(stderr, "prefill failed: %s\n", qwen_model_last_error());
        return 1;
    }
    qwen_model_set_layer_callback(q, nullptr, nullptr);
    double worst = 1.0;
    int worst_l = -1;
    for (size_t l = 0; l < cap.layer.size(); ++l) {
        std::vector<float> r;
        if (!ref("l_last-" + std::to_string(l), r) || r.size() != cap.layer[l].size()) {
            check(false, ("reference l_last-" + std::to_string(l) + " (size)").c_str());
            continue;
        }
        double lw = 1.0;
        for (int p = 0; p < n; ++p)
            lw = fmin(lw, cosine(cap.layer[l].data() + (size_t)p * cap.hcd, r.data() + (size_t)p * cap.hcd, cap.hcd));
        printf("    layer %2zu: worst cosine over %d positions %.6f\n", l, n, lw);
        if (getenv("QWEN4EXP_VERIFY_POS")) {  // per position, and the largest |value| (ours, reference)
            for (int p = 0; p < n; ++p) {
                const float* a = cap.layer[l].data() + (size_t)p * cap.hcd;
                const float* b = r.data() + (size_t)p * cap.hcd;
                float ma = 0, mb = 0;
                for (int i = 0; i < cap.hcd; ++i) ma = fmaxf(ma, fabsf(a[i])), mb = fmaxf(mb, fabsf(b[i]));
                printf("      pos %2d: cosine %.6f, max |x| %.4g / %.4g\n", p, cosine(a, b, cap.hcd), ma, mb);
            }
        }
        if (lw < worst) worst = lw, worst_l = (int)l;
    }
    snprintf(what, sizeof(what), "residual streams after every layer vs l_last-N: worst cosine %.6f (layer %d)", worst,
             worst_l);
    check(!cap.layer.empty() && worst > 0.9, what);
    // Each layer alone, from the reference's input to it (teacher forcing): errors do not compound, so a layer
    // that is wrong stands out. The cosine is over the layer's change to the streams (output - input), which
    // is what the layer computes.
    {
        double iso = 1.0;
        int iso_l = -1, n_upd = 0, n_close = 0;
        for (size_t l = 0; l < cap.layer.size(); ++l) {
            std::vector<float> in, out;
            if (l == 0 || !ref("l_last-" + std::to_string(l - 1), in) || !ref("l_last-" + std::to_string(l), out))
                continue;
            Capture one;
            qwen_model_set_layer_callback(q, on_layer, &one);
            const int rc = qwen_model_run_layers(q, toks.data(), n, in.data(), (int)l, (int)l + 1);
            qwen_model_set_layer_callback(q, nullptr, nullptr);
            if (rc != 0 || one.layer.size() <= l) { check(false, qwen_model_last_error()); break; }
            double lw = 1.0;
            for (int p = 0; p < n; ++p) {
                std::vector<float> da(cap.hcd), db(cap.hcd);
                for (int i = 0; i < cap.hcd; ++i) {
                    const size_t k = (size_t)p * cap.hcd + i;
                    da[i] = one.layer[l][k] - in[k];
                    db[i] = out[k] - in[k];
                }
                const double c = cosine(da.data(), db.data(), cap.hcd);
                ++n_upd;
                n_close += c >= 0.999;
                if (getenv("QWEN4EXP_VERIFY_POS") && c < 0.999) printf("      layer %2zu pos %2d: cosine %.6f\n", l, p, c);
                lw = fmin(lw, c);
            }
            if (getenv("QWEN4EXP_VERIFY_POS")) printf("    layer %2zu alone: worst cosine of its update %.6f\n", l, lw);
            if (lw < iso) iso = lw, iso_l = (int)l;
        }
        snprintf(what, sizeof(what), "each layer alone from the reference input: %d of %d updates (layer, position) at "
                 "cosine >= 0.999, worst %.6f (layer %d)", n_close, n_upd, iso, iso_l);
        check(n_upd > 0 && n_close >= 0.95 * n_upd && iso > 0.95, what);
    }
    std::vector<float> rl;
    if (ref("result_output", rl) && rl.size() >= (size_t)V) {
        const float* last = rl.data() + rl.size() - V;
        const double c = cosine(logits.data(), last, (size_t)V);
        const int a = argmax(logits.data(), V);
        snprintf(what, sizeof(what), "last-token logits vs result_output: cosine %.6f, argmax %d (reference %d, its "
                 "logit there %.4f vs max %.4f)", c, a, ref_next, last[a], last[argmax(last, V)]);
        check(c > 0.98 && (a == ref_next || last[argmax(last, V)] - last[a] < 0.5f), what);
    } else {
        check(false, "reference result_output");
    }

    // 2. Token by token and a split prefill give the same logits as the prompt at once
    auto same = [&](const std::vector<float>& a, const std::string how) {
        const double r = max_rel(a.data(), logits.data(), (size_t)V), c = cosine(a.data(), logits.data(), (size_t)V);
        snprintf(what, sizeof(what), "%s: max relative difference %.3g, cosine %.7f", how.c_str(), r, c);
        check(exact ? r == 0.0 : c > 0.98, what);
    };
    std::vector<float> l2((size_t)V);
    bool ok = true;
    for (int p = 0; p < n && ok; ++p) ok = qwen_model_forward(q, toks[p], p, l2.data()) == 0;
    if (ok) same(l2, "token by token");
    const int cut = n / 3;
    ok = qwen_model_prefill(q, toks.data(), cut, 0, nullptr) == 0 &&
         qwen_model_prefill(q, toks.data() + cut, n - cut, cut, l2.data()) == 0;
    snprintf(what, sizeof(what), "prefill of %d then %d tokens", cut, n - cut);
    if (ok) same(l2, what);
    // A checkpoint of the recurrent state (with the PLE history) restored mid-sequence
    ok = qwen_model_prefill(q, toks.data(), cut, 0, nullptr) == 0 && qwen_model_state_save(q, 0) == 0 &&
         qwen_model_prefill(q, toks.data() + cut, 3, cut, nullptr) == 0 && qwen_model_state_restore(q, 0) == 0 &&
         qwen_model_prefill(q, toks.data() + cut, n - cut, cut, l2.data()) == 0;
    if (ok) same(l2, "restored checkpoint, then the rest");
    if (!ok) check(false, qwen_model_last_error());

    inference_kv_cache_free(m);
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
