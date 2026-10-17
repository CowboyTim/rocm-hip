// matvec_bench -- cold-cache qmatvec bandwidth per tensor role (Phase 20 speed work). Run from build/co.
//
//   matvec_bench <model.gguf> [reps]
//
// A generated token reads every weight tensor exactly once, so nothing is warm in the 32 MB Infinity
// Cache. This runs one role (e.g. all 32 ffn_down) back to back across the layers, each tensor once per
// pass, and reports GB/s of weight bytes per role and per weight type (best of reps passes). Activations
// live in coarse-grained pinned memory, like the engine's scratch.

#include "inference.h"
#include "inference_internal.h"
#include "qmatvec.h"
#include "quant_types.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <sys/mman.h>
#include <vector>

static float* pinned(size_t n) {
    void* p = mmap(nullptr, n * 4, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED || hipHostRegister(p, n * 4, hipExtHostRegisterCoarseGrained) != hipSuccess) return nullptr;
    for (size_t i = 0; i < n; ++i) ((float*)p)[i] = 0.01f * (float)(i % 97) - 0.4f;
    return (float*)p;
}

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [reps]\n", argv[0]);
        return 2;
    }
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    inference_model* m = inference_load_model(argv[1]);
    if (!m || qmatvec_init("matvec_q.co") != 0) return 1;
    float* x = pinned(1 << 20);
    float* y = pinned(1 << 20);
    if (!x || !y) return 1;

    struct Role { std::string name; std::vector<const QuantWeight*> w; };
    std::vector<Role> roles;
    auto add = [&](const char* name, const QuantWeight& w) {
        if (!w.data) return;
        for (Role& r : roles)
            if (r.name == name) { r.w.push_back(&w); return; }
        roles.push_back({name, {&w}});
    };
    for (const LayerWeights& L : m->layers) {
        add("ffn_gate", L.W_gate); add("ffn_up", L.W_up); add("ffn_down", L.W_down);
        add("attn_qkv", L.W_qkv); add("attn_gate (z)", L.W_z); add("ssm_alpha", L.W_alpha);
        add("ssm_beta", L.W_beta); add("ssm_out", L.W_ssm_out);
        add("attn_q", L.Wq); add("attn_k", L.Wk); add("attn_v", L.Wv); add("attn_output", L.Wo);
    }
    add("lm_head", m->output);

    std::map<std::string, std::pair<double, double>> by_type;  // type name -> (bytes, seconds) per pass
    double all_bytes = 0, all_t = 0;
    printf("%-14s %-16s %5s %7s %6s %9s %9s %8s\n", "role", "type", "n", "N", "K", "MB/pass", "ms/pass", "GB/s");
    for (const Role& r : roles) {
        double bytes = 0;
        for (const QuantWeight* w : r.w) bytes += (double)w->row_bytes * w->N;
        double best = 1e9;
        for (int it = 0; it < reps + 1; ++it) {
            (void)hipDeviceSynchronize();
            const double t0 = now();
            for (const QuantWeight* w : r.w)
                if (qmatvec_launch(w->type, w->data, w->row_bytes, x, y, w->N, w->K, 0) != hipSuccess) return 1;
            (void)hipDeviceSynchronize();
            if (it > 0) best = std::min(best, now() - t0);  // pass 0 warms up code and TLB
        }
        const QuantWeight& w0 = *r.w[0];
        const char* tn = qt_get_traits(w0.type) ? qt_get_traits(w0.type)->name : "?";
        printf("%-14s %-16s %5zu %7d %6d %9.1f %9.3f %8.1f\n", r.name.c_str(), tn, r.w.size(), w0.N, w0.K, bytes / 1e6,
               best * 1e3, bytes / best / 1e9);
        by_type[tn].first += bytes;
        by_type[tn].second += best;
        all_bytes += bytes;
        all_t += best;
    }
    printf("\nby type:\n");
    for (auto& kv : by_type)
        printf("  %-16s %9.1f MB %9.3f ms %8.1f GB/s\n", kv.first.c_str(), kv.second.first / 1e6, kv.second.second * 1e3,
               kv.second.first / kv.second.second / 1e9);
    printf("all matvecs of a token: %.1f MB in %.3f ms = %.1f GB/s\n", all_bytes / 1e6, all_t * 1e3,
           all_bytes / all_t / 1e9);
    qmatvec_free();
    inference_free_model(m);
    return 0;
}
