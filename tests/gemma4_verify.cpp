// gemma4_verify -- the Gemma-4 forward pass vs llama.cpp's activations (ref_dump acts), layer by layer.
//
//   gemma4_verify <model.gguf> <ref dir>        (run from build/co, or with INFERENCE_KERNEL_DIR)
//
// The reference dir comes from `ref_dump acts <model.gguf> <prompt> <dir>`: tokens.txt, index.tsv and one
// <seq>_<name>.f32 per graph tensor. The prompt's tokens go through the engine in one prefill; after every layer
// its residual stream (all tokens) is compared with the reference's l_out-<layer>, then the logits of the last
// token with result_output (after the soft cap). GEMMA4_VERIFY_LAYERS=1 prints every layer.

#include "inference.h"
#include "qwen_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static std::map<std::string, std::string> g_files;  // tensor name -> file in the reference dir
static std::string g_dir;

static bool load_ref(const std::string& name, std::vector<float>& out) {
    auto it = g_files.find(name);
    if (it == g_files.end()) return false;
    std::ifstream f(g_dir + "/" + it->second, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    out.resize((size_t)f.tellg() / sizeof(float));
    f.seekg(0);
    f.read((char*)out.data(), (std::streamsize)(out.size() * sizeof(float)));
    return (bool)f;
}

static double cosine(const float* a, const float* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) ab += (double)a[i] * b[i], aa += (double)a[i] * a[i], bb += (double)b[i] * b[i];
    return aa > 0 && bb > 0 ? ab / sqrt(aa * bb) : (aa == bb ? 1.0 : 0.0);
}

struct Capture {
    int n_embd = 0, n_tokens = 0;
    std::vector<std::vector<float>> layers;  // [layer][token * n_embd]
};

static void on_layer(int layer, int pos, const float* x, int n, void* user) {
    Capture* c = (Capture*)user;
    if ((int)c->layers.size() <= layer) c->layers.resize(layer + 1, std::vector<float>((size_t)c->n_tokens * n));
    if (pos < c->n_tokens) std::copy(x, x + n, c->layers[layer].begin() + (size_t)pos * n);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <ref dir>\n", argv[0]);
        return 2;
    }
    g_dir = argv[2];
    {
        std::ifstream idx(g_dir + "/index.tsv");
        std::string line;
        std::getline(idx, line);  // header
        while (std::getline(idx, line)) {
            std::istringstream ss(line);
            std::string seq, name, op, ne0, ne1, ne2, ne3, file;
            std::getline(ss, seq, '\t'), std::getline(ss, name, '\t'), std::getline(ss, op, '\t');
            std::getline(ss, ne0, '\t'), std::getline(ss, ne1, '\t'), std::getline(ss, ne2, '\t');
            std::getline(ss, ne3, '\t'), std::getline(ss, file, '\t');
            if (!name.empty() && !file.empty()) g_files[name] = file;  // the last one of a name wins
        }
    }
    std::vector<int> tokens;
    {
        std::ifstream tf(g_dir + "/tokens.txt");
        for (int t; tf >> t;) tokens.push_back(t);
    }
    if (tokens.empty() || g_files.empty()) {
        fprintf(stderr, "no reference in %s\n", argv[2]);
        return 2;
    }

    inference_model_t* m = inference_load_model(argv[1]);
    if (!m) { fprintf(stderr, "load failed: %s\n", inference_get_last_error()); return 1; }
    if (inference_kv_cache_init(m, 4096) != 0) { fprintf(stderr, "kv init: %s\n", inference_get_last_error()); return 1; }
    qwen_model_t* q = inference_get_qwen_model(m);
    const int V = inference_vocab_size(m);

    Capture cap;
    std::vector<float> ref0;
    if (!load_ref("l_out-0", ref0)) { fprintf(stderr, "no l_out-0 in the reference\n"); return 2; }
    cap.n_tokens = (int)tokens.size();
    cap.n_embd = (int)(ref0.size() / tokens.size());
    qwen_model_set_layer_callback(q, on_layer, &cap);
    std::vector<float> logits(V);
    if (qwen_model_prefill(q, tokens.data(), (int)tokens.size(), 0, logits.data()) != 0) {
        fprintf(stderr, "prefill failed: %s\n", qwen_model_last_error());
        return 1;
    }

    const bool verbose = getenv("GEMMA4_VERIFY_LAYERS") != nullptr;
    int fail = 0, worst_layer = -1;
    double worst = 2.0;
    const int E = cap.n_embd, T = cap.n_tokens;
    for (int l = 0; l < (int)cap.layers.size(); ++l) {
        // the last layer's l_out may hold only the output row (llama.cpp drops the others there): h_nextn has all
        std::vector<float> ref;
        if ((!load_ref("l_out-" + std::to_string(l), ref) || ref.size() != (size_t)E * T) &&
            (l + 1 != (int)cap.layers.size() || !load_ref("h_nextn", ref) || ref.size() != (size_t)E * T)) {
            fprintf(stderr, "l_out-%d: missing or %zu values (want %d x %d)\n", l, ref.size(), T, E);
            return 2;
        }
        double lw = 2.0, mean = 0.0;
        std::string per;
        for (int t = 0; t < T; ++t) {
            const double c = cosine(cap.layers[l].data() + (size_t)t * E, ref.data() + (size_t)t * E, E);
            lw = std::min(lw, c);
            mean += c / T;
            char b[16];
            snprintf(b, sizeof(b), " %.4f", c);
            per += b;
        }
        if (lw < worst) worst = lw, worst_layer = l;
        if (verbose) printf("  layer %2d: worst %.6f mean %.6f per token%s\n", l, lw, mean, per.c_str());
    }
    // Chained over 30 layers, rounding differences grow (MoE routing near-ties, many norms): 0.96 with WMMA, 0.94
    // without on the reference prompt, from summation order alone. Information only; the per-layer check below (each
    // layer on the reference's input) and the logits are the tests
    printf("  [info] l_out, all %zu layers x %d tokens chained: worst cos %.6f (layer %d)\n", cap.layers.size(), T, worst,
           worst_layer);

    // Teacher forcing: each layer alone on the reference's input (inp_scaled, or the previous l_out), so its error
    // does not carry into the next; a wrong layer stands out here even when the chained run drifts
    double tf_worst = 2.0;
    int tf_layer = -1;
    for (int l = 0; l + 1 < (int)cap.layers.size(); ++l) {
        std::vector<float> in, ref;
        if (!load_ref(l == 0 ? std::string("inp_scaled") : "l_out-" + std::to_string(l - 1), in) ||
            !load_ref("l_out-" + std::to_string(l), ref) || in.size() != (size_t)E * T || ref.size() != (size_t)E * T)
            continue;
        Capture one;
        one.n_tokens = T;
        one.n_embd = E;
        qwen_model_set_layer_callback(q, on_layer, &one);
        if (qwen_model_run_layers(q, tokens.data(), T, in.data(), l, l + 1) != 0) {
            fprintf(stderr, "run_layers %d: %s\n", l, qwen_model_last_error());
            return 1;
        }
        double lw = 2.0;
        for (int t = 0; t < T; ++t)
            lw = std::min(lw, cosine(one.layers[l].data() + (size_t)t * E, ref.data() + (size_t)t * E, E));
        if (verbose) printf("  layer %2d alone on the reference input: worst cos %.6f\n", l, lw);
        if (lw < tf_worst) tf_worst = lw, tf_layer = l;
    }
    const bool tf_ok = tf_worst >= 0.999;
    fail += !tf_ok;
    printf("  [%s] each layer on the reference's input: worst cos %.6f (layer %d)\n", tf_ok ? "PASS" : "FAIL", tf_worst,
           tf_layer);

    std::vector<float> ref_logits;
    if (!load_ref("result_output", ref_logits) || (int)ref_logits.size() < V) {
        fprintf(stderr, "no result_output in the reference\n");
        return 2;
    }
    const float* rl = ref_logits.data() + ref_logits.size() - V;  // the last token's row
    int a = 0, b = 0;
    for (int i = 1; i < V; ++i) {
        if (logits[i] > logits[a]) a = i;
        if (rl[i] > rl[b]) b = i;
    }
    const double lc = cosine(logits.data(), rl, V);
    const bool logits_ok = lc >= 0.99 && a == b;
    fail += !logits_ok;
    printf("  [%s] logits (soft-capped) cos %.6f, top token %d (reference %d)\n", logits_ok ? "PASS" : "FAIL", lc, a,
           b);
    // The same tokens one at a time: with QMATVEC_NO_WMMA=1 every kernel computes a token as a prefill does, so
    // the logits must be identical; with WMMA only the summation order differs
    qwen_model_set_layer_callback(q, nullptr, nullptr);
    std::vector<float> one(V);
    for (int t = 0; t < T; ++t)
        if (qwen_model_forward(q, tokens[t], t, one.data()) != 0) {
            fprintf(stderr, "forward %d: %s\n", t, qwen_model_last_error());
            return 1;
        }
    double md = 0.0;
    int a1 = 0;
    for (int i = 0; i < V; ++i) {
        md = std::max(md, (double)fabsf(one[i] - logits[i]));
        if (one[i] > one[a1]) a1 = i;
    }
    // (prefill batches of 8+ tokens take the WMMA expert kernels: only the summation order differs)
    const bool exact = getenv("QMATVEC_NO_WMMA") != nullptr;
    const double tc = cosine(one.data(), logits.data(), V);
    const bool tbt_ok = exact ? md == 0.0 : (tc >= 0.999 && a1 == a);
    fail += !tbt_ok;
    printf("  [%s] token by token vs prefill: cos %.6f, max |d| logits %.3g, top %d%s\n", tbt_ok ? "PASS" : "FAIL", tc,
           md, a1, exact ? " (QMATVEC_NO_WMMA: must be identical)" : "");
    inference_free_model(m);
    printf("%s (%d failed)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
