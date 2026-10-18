// forward_verify — Phase 20 Step 3c: engine pieces vs the llama.cpp reference activations.
//
//   forward_verify <model.gguf> <ref_dir>
//
// ref_dir is a `ref_dump acts` directory (index.tsv + <seq>_<name>.f32 + tokens.txt + next_token.txt).
// Each check feeds a *reference* input through one engine piece (the loader's QuantWeight bindings,
// qmatvec, qwen_ops) and diffs the result with the reference output, for every prompt token, so errors
// do not compound across layers. Run from build/co (loads matvec_q.co).
//
// Projections: ggml's CPU matmul quantizes the activations to the weight type's vec_dot_type (Q8_0 for
// ROCMFP4, Q8_K for K-quants) before the dot product, while qmatvec keeps them in FP32. So each
// projection runs twice: on the FP32 input (reported, cosine ~0.999) and on the input round-tripped
// through that Q8 format, which must match the reference to 1e-4 relative. That second number checks
// the weight bindings and kernels; the first shows the reference's own activation-quantization noise.
// Element-wise ops (norms, RoPE, SwiGLU, gating, conv1d, delta rule) must agree to 1e-4 relative;
// attention to 2e-3, since the reference's flash attention rounds to FP16 internally.
//
// The last section runs the whole engine (qwen_model_forward, one token at a time) on the reference
// tokens and diffs the residual stream after every layer with l_out-N, then the next-token argmax.
// Errors compound there, and the reference quantizes activations to Q8 for every matmul, so that
// section passes on cosine similarity (> 0.99 per layer, the Phase 20 success criterion).

#include "inference.h"
#include "inference_internal.h"
#include "attention.h"
#include "deltanet.h"
#include "gpu_ops.h"
#include "qmatvec.h"
#include "quant_types.h"
#include "qwen_model.h"
#include "qwen_ops.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------- reference tensors

struct RefTensor {
    std::string name, op, file;
    int64_t ne[4] = {1, 1, 1, 1};
};

static std::string g_dir;
static std::vector<RefTensor> g_index;

static bool load_index() {
    std::ifstream f(g_dir + "/index.tsv");
    if (!f) return false;
    std::string line;
    std::getline(f, line);  // header
    while (std::getline(f, line)) {
        std::vector<std::string> c;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) c.push_back(cell);
        if (c.size() < 8) continue;
        RefTensor t;
        t.name = c[1];
        t.op = c[2];
        for (int i = 0; i < 4; ++i) t.ne[i] = std::stoll(c[3 + i]);
        t.file = c[7];
        g_index.push_back(t);
    }
    return !g_index.empty();
}

// First tensor with this name (and op and ne[0], if given), read as rows of ne[0] floats
static bool ref(const char* name, const char* op, std::vector<float>& out, int64_t* row_len = nullptr,
                int64_t ne0 = 0) {
    for (const RefTensor& t : g_index) {
        if (t.name != name || (op && t.op != op) || (ne0 && t.ne[0] != ne0)) continue;
        const int64_t n = t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
        out.resize((size_t)n);
        FILE* f = fopen((g_dir + "/" + t.file).c_str(), "rb");
        if (!f) return false;
        const size_t got = fread(out.data(), sizeof(float), (size_t)n, f);
        fclose(f);
        if (row_len) *row_len = t.ne[0];
        return got == (size_t)n;
    }
    fprintf(stderr, "reference tensor %s%s%s not found\n", name, op ? " op " : "", op ? op : "");
    return false;
}

// ---------------------------------------------------------------- pinned buffers + comparison

// /dev/shm-backed, hipHostRegister'ed buffer (project core pattern)
struct ShmBuf {
    float* p = nullptr;
    size_t n = 0;
    bool alloc(const char* name, size_t floats) {
        std::string path = std::string("/dev/shm/") + name;
        int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0 || ftruncate(fd, (off_t)(floats * sizeof(float))) != 0) return false;
        void* m = mmap(nullptr, floats * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        unlink(path.c_str());
        if (m == MAP_FAILED) return false;
        p = (float*)m;
        n = floats;
        return hipHostRegister(m, floats * sizeof(float), hipHostRegisterMapped) == hipSuccess;
    }
    ~ShmBuf() {
        if (p) {
            (void)hipHostUnregister(p);
            munmap(p, n * sizeof(float));
        }
    }
};

// Activation round trip through ggml's vec_dot_type for weight type wtype (ggml-cpu.c type_traits_cpu)
static void ref_act_roundtrip(int wtype, const float* x, float* out, int K) {
    const bool q8k = wtype >= QT_Q2_K && wtype <= QT_Q6_K;
    const bool q80 = wtype == QT_Q4_0 || wtype == QT_Q4_1 || wtype == QT_Q5_0 || wtype == QT_Q5_1 ||
                     wtype == QT_Q8_0 || wtype == QT_Q4_0_ROCMFP4 || wtype == QT_Q4_0_ROCMFP4_FAST;
    if (q8k) {  // quantize_row_q8_K_ref: d = max / -127 (signed max), q = nearest_int(x / d), clamped to 127
        for (int b = 0; b < K; b += 256) {
            float amax = 0.0f, mx = 0.0f;
            for (int j = 0; j < 256; ++j)
                if (fabsf(x[b + j]) > amax) { amax = fabsf(x[b + j]); mx = x[b + j]; }
            if (amax == 0.0f) { for (int j = 0; j < 256; ++j) out[b + j] = 0.0f; continue; }
            const float iscale = -127.0f / mx, d = 1.0f / iscale;
            for (int j = 0; j < 256; ++j) out[b + j] = d * (float)std::min(127, (int)nearbyintf(iscale * x[b + j]));
        }
    } else if (q80) {  // x86 SIMD quantize_row_q8_0 (the reference CPU build's path, not the _ref one):
        // d = amax / 127 stored as FP16, q = round-half-even(x * (127 / amax)). The _ref variant's
        // roundf(x * (1 / d)) differs by one step near ties, which blk.0's Q6_K-embedding inputs hit.
        for (int b = 0; b < K; b += 32) {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[b + j]));
            const float d = amax / 127.0f, id = amax != 0.0f ? 127.0f / amax : 0.0f;
            const float dh = __half2float(__float2half(d));
            for (int j = 0; j < 32; ++j) out[b + j] = dh * nearbyintf(x[b + j] * id);
        }
    } else if (wtype == QT_BF16) {  // ggml_fp32_to_bf16: round to nearest even
        for (int i = 0; i < K; ++i) {
            uint32_t u;
            memcpy(&u, &x[i], 4);
            u = (u + (0x7fff + ((u >> 16) & 1))) & 0xffff0000u;
            memcpy(&out[i], &u, 4);
        }
    } else {
        memcpy(out, x, (size_t)K * sizeof(float));
    }
}

static int g_fail = 0;

// FORWARD_VERIFY_LAYERS=1 prints the per-layer end-to-end diff
static bool verbose_layers() { return getenv("FORWARD_VERIFY_LAYERS") != nullptr; }

// Worst case over all tokens of one check
struct Stat {
    double min_cos = 1.0, max_rel = 0.0;
    void add(const float* a, const float* b, int n) {
        double dot = 0, na = 0, nb = 0, maxd = 0, maxb = 0;
        for (int i = 0; i < n; ++i) {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
            maxd = fmax(maxd, fabs((double)a[i] - b[i]));
            maxb = fmax(maxb, fabs((double)b[i]));
        }
        const double cos = (na > 0 && nb > 0) ? dot / sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
        min_cos = fmin(min_cos, cos);
        max_rel = fmax(max_rel, maxb > 0 ? maxd / maxb : maxd);
    }
};

// Element-wise op: pass on max_rel
static void report(const char* what, const Stat& s, double tol = 1e-4) {
    const bool ok = s.max_rel <= tol;
    if (!ok) ++g_fail;
    printf("  [%s] %-36s cos %.7f  max|d|/max|ref| %.2e\n", ok ? "PASS" : "FAIL", what, s.min_cos, s.max_rel);
}

// Projection: FP32 input (informational), reference-quantized input on the float kernel, and the engine's
// int8 kernel on the raw input (qmatvec_quantize + qmatvec_launch_q8, which quantizes like the reference).
// The last two must match the reference to 1e-4.
struct ProjStat {
    Stat f32, emu, q8;
    bool has_q8 = false;
};

static void report(const char* what, const ProjStat& s) {
    const bool ok = s.emu.max_rel <= 1e-4 && (!s.has_q8 || s.q8.max_rel <= 1e-4);
    if (!ok) ++g_fail;
    printf("  [%s] %-36s cos %.7f  max|d|/max|ref| %.2e", ok ? "PASS" : "FAIL", what, s.emu.min_cos, s.emu.max_rel);
    if (s.has_q8) printf("   int8: %.2e", s.q8.max_rel);
    printf("   (FP32 input: cos %.7f, %.2e)\n", s.f32.min_cos, s.f32.max_rel);
}

static float* g_x = nullptr;  // pinned input
static float* g_y = nullptr;  // pinned output
static qmatvec_q8 g_q8;       // quantized input for the int8 kernels

static bool launch(const QuantWeight& W) {
    hipError_t e = qmatvec_launch(W.type, W.data, W.row_bytes, g_x, g_y, W.N, W.K, 0);
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e != hipSuccess) fprintf(stderr, "qmatvec type %d: %s\n", W.type, hipGetErrorString(e));
    return e == hipSuccess;
}

// y = W x both ways, compared with the reference output ref (W.N values); leaves the FP32 result in g_y
static bool proj(const QuantWeight& W, const float* x, const float* ref_out, ProjStat& st) {
    ref_act_roundtrip(W.type, x, g_x, W.K);
    if (!launch(W)) return false;
    st.emu.add(g_y, ref_out, W.N);
    memcpy(g_x, x, (size_t)W.K * sizeof(float));
    if (!launch(W)) return false;
    st.f32.add(g_y, ref_out, W.N);
    if (qmatvec_q8_supported(W.type, W.K) && W.K <= g_q8.max_k) {
        hipError_t e = qmatvec_quantize(g_x, W.K, &g_q8, 0);
        if (e == hipSuccess) e = qmatvec_launch_q8(W.type, W.data, W.row_bytes, &g_q8, g_y, W.N, W.K, 0);
        if (e == hipSuccess) e = hipDeviceSynchronize();
        if (e != hipSuccess) { fprintf(stderr, "int8 qmatvec type %d: %s\n", W.type, hipGetErrorString(e)); return false; }
        st.q8.add(g_y, ref_out, W.N);
        st.has_q8 = true;
    }
    return true;
}

// ---------------------------------------------------------------- checks

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <ref_dir>   (run from build/co)\n", argv[0]);
        return 2;
    }
    g_dir = argv[2];
    if (!load_index()) { fprintf(stderr, "cannot read %s/index.tsv\n", argv[2]); return 2; }

    std::vector<int> tokens;
    {
        std::ifstream f(g_dir + "/tokens.txt");
        int t;
        while (f >> t) tokens.push_back(t);
    }
    int next_token = -1;
    {
        std::ifstream f(g_dir + "/next_token.txt");
        f >> next_token;
    }
    const int T = (int)tokens.size();

    inference_model* m = inference_load_model(argv[1]);
    if (!m) { fprintf(stderr, "load failed: %s\n", inference_get_last_error()); return 1; }
    if (qmatvec_init("matvec_q.co") != 0 || gpu_ops_init("gpu_ops.co") != 0) return 1;

    const int E = m->n_embd, HD = m->head_dim, HQ = m->n_head, HKV = m->n_head_kv, FF = m->n_ff;
    const int V = m->vocab_size;
    const float eps = (float)m->eps;
    printf("forward_verify: %s, %d tokens, E=%d HQ=%d HKV=%d HD=%d n_rot=%d FF=%d V=%d eps=%g\n", argv[1], T, E, HQ,
           HKV, HD, m->n_rot, FF, V, eps);

    ShmBuf xin, yout;
    const size_t big = (size_t)std::max(V, std::max(2 * HQ * HD, FF));
    if (!xin.alloc("fwdv_x.bin", big) || !yout.alloc("fwdv_y.bin", big)) { fprintf(stderr, "alloc failed\n"); return 1; }
    g_x = xin.p;
    g_y = yout.p;
    ShmBuf q8buf;
    const int q8_max = std::max((std::max(E, std::max(FF, HQ * HD)) + 255) / 256 * 256,
                                T * std::max(m->n_expert_used * m->n_ff_exp, E));
    if (!q8buf.alloc("fwdv_q8.bin", (qmatvec_q8_bytes(q8_max) + 3) / 4)) { fprintf(stderr, "alloc failed\n"); return 1; }
    qmatvec_q8_bind(&g_q8, q8buf.p, q8_max);
    float* y = g_y;
    std::vector<float> a, b, c, d;

    // ---- token embedding
    {
        Stat s;
        if (!ref("model.input_embed", nullptr, a)) return 1;
        for (int t = 0; t < T; ++t) {
            qt_to_float(m->token_embd.type, m->token_embd.row(tokens[t]), y, E);
            s.add(y, &a[(size_t)t * E], E);
        }
        report("token_embd row decode", s);
    }

    // ---- layer 0: attn_norm, post_attention_norm, FFN
    {
        const LayerWeights& L = m->layers[0];
        Stat sn, spn, ssw;
        ProjStat sg, su, sd;
        if (!ref("model.input_embed", nullptr, a) || !ref("attn_norm-0", nullptr, b)) return 1;
        for (int t = 0; t < T; ++t) {
            qwen_rmsnorm(&a[(size_t)t * E], L.attn_norm, y, E, eps);
            sn.add(y, &b[(size_t)t * E], E);
        }
        report("blk.0 attn_norm", sn);

        if (!ref("attn_residual-0", nullptr, a) || !ref("attn_post_norm-0", nullptr, b)) return 1;
        for (int t = 0; t < T; ++t) {
            qwen_rmsnorm(&a[(size_t)t * E], L.ffn_norm, y, E, eps);
            spn.add(y, &b[(size_t)t * E], E);
        }
        report("blk.0 post_attention_norm", spn);

        if (!L.moe) {
            std::vector<float> g, u, sw, out;
            if (!ref("ffn_gate-0", nullptr, g) || !ref("ffn_up-0", nullptr, u) || !ref("ffn_swiglu-0", nullptr, sw) ||
                !ref("ffn_out-0", nullptr, out))
                return 1;
            std::vector<float> act(FF);
            for (int t = 0; t < T; ++t) {
                if (!proj(L.W_gate, &b[(size_t)t * E], &g[(size_t)t * FF], sg)) return 1;
                if (!proj(L.W_up, &b[(size_t)t * E], &u[(size_t)t * FF], su)) return 1;
                for (int i = 0; i < FF; ++i) act[i] = qwen_silu(g[(size_t)t * FF + i]) * u[(size_t)t * FF + i];
                ssw.add(act.data(), &sw[(size_t)t * FF], FF);
                if (!proj(L.W_down, &sw[(size_t)t * FF], &out[(size_t)t * E], sd)) return 1;
            }
            printf("  (blk.0 types: ffn_gate %d, ffn_up %d, ffn_down %d)\n", L.W_gate.type, L.W_up.type, L.W_down.type);
            report("blk.0 ffn_gate (qmatvec)", sg);
            report("blk.0 ffn_up (qmatvec)", su);
            report("blk.0 SwiGLU", ssw);
            report("blk.0 ffn_down (qmatvec)", sd);
        } else {
            // Mixture of experts: router, expert selection and weights, the selected experts' projections (the
            // engine's expert-indexed int8 kernels, fed the reference's own expert choice so slots line up),
            // the shared expert, and the weighted combine. Every piece starts from reference inputs.
            const int NE = m->n_expert, NU = m->n_expert_used, FE = m->n_ff_exp, FS = m->n_ff_shexp;
            std::vector<float> lg, pr, wn, mg, mu, msw, md, shg, shu, shsw, shd, sgs, fo;
            if (!ref("ffn_moe_logits-0", "MUL_MAT", lg) || !ref("ffn_moe_probs-0", "SOFT_MAX", pr) ||
                !ref("ffn_moe_weights_norm-0", "DIV", wn) || !ref("ffn_moe_gate-0", nullptr, mg) ||
                !ref("ffn_moe_up-0", nullptr, mu) || !ref("ffn_moe_swiglu-0", nullptr, msw) ||
                !ref("ffn_moe_down-0", nullptr, md) || !ref("ffn_gate-0", "MUL_MAT", shg) ||
                !ref("ffn_up-0", "MUL_MAT", shu) || !ref("ffn_swiglu-0", nullptr, shsw) ||
                !ref("ffn_shexp-0", nullptr, shd) || !ref("shared_expert_gate_sigmoid-0", nullptr, sgs) ||
                !ref("ffn_out-0", nullptr, fo))
                return 1;

            ProjStat srl, ssg, ssu, ssd;
            for (int t = 0; t < T; ++t)
                if (!proj(L.W_router, &b[(size_t)t * E], &lg[(size_t)t * NE], srl)) return 1;

            // Reference expert choice: the NU largest probabilities, descending (ties: lowest index)
            std::vector<int> rids((size_t)T * NU);
            for (int t = 0; t < T; ++t) {
                std::vector<int> idx(NE);
                for (int e = 0; e < NE; ++e) idx[e] = e;
                const float* pt = &pr[(size_t)t * NE];
                std::stable_sort(idx.begin(), idx.end(), [&](int i, int j) { return pt[i] > pt[j]; });
                for (int j = 0; j < NU; ++j) rids[(size_t)t * NU + j] = idx[j];
            }

            ShmBuf b_lg, b_x, b_ids, b_w, b_sg, b_d, b_sh, b_out;
            if (!b_lg.alloc("fwdv_moe_lg.bin", (size_t)T * NE) || !b_x.alloc("fwdv_moe_x.bin", (size_t)T * E) ||
                !b_ids.alloc("fwdv_moe_ids.bin", (size_t)T * NU) || !b_w.alloc("fwdv_moe_w.bin", (size_t)T * NU) ||
                !b_sg.alloc("fwdv_moe_sg.bin", T) || !b_d.alloc("fwdv_moe_d.bin", (size_t)T * NU * E) ||
                !b_sh.alloc("fwdv_moe_sh.bin", (size_t)T * E) || !b_out.alloc("fwdv_moe_out.bin", (size_t)T * E)) {
                fprintf(stderr, "alloc failed\n");
                return 1;
            }
            int* gids = (int*)b_ids.p;
            memcpy(b_lg.p, lg.data(), (size_t)T * NE * sizeof(float));
            memcpy(b_x.p, b.data(), (size_t)T * E * sizeof(float));
            hipError_t e = gpu_moe_router(b_lg.p, b_x.p, L.shexp_gate, gids, b_w.p, b_sg.p, NE, NU, E, T, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { fprintf(stderr, "moe_router: %s\n", hipGetErrorString(e)); return 1; }
            int id_bad = 0;
            for (int i = 0; i < T * NU; ++i) id_bad += gids[i] != rids[i];
            Stat sw8, ssgs;
            sw8.add(b_w.p, wn.data(), T * NU);
            ssgs.add(b_sg.p, sgs.data(), T);

            // Expert projections with the reference's expert choice
            memcpy(gids, rids.data(), rids.size() * sizeof(int));
            Stat seg, seu, sesw, sed, scomb;
            const QuantWeight* W3[2] = {&L.We_gate, &L.We_up};
            const std::vector<float>* R3[2] = {&mg, &mu};
            Stat* S3[2] = {&seg, &seu};
            for (int k = 0; k < 2; ++k) {
                memcpy(g_x, b.data(), (size_t)T * E * sizeof(float));
                e = qmatvec_quantize(g_x, T * E, &g_q8, 0);
                if (e == hipSuccess)
                    e = qmatvec_launch_q8_id(W3[k]->type, W3[k]->data, W3[k]->row_bytes, gids, NU, 0, &g_q8, g_y, FE, E, T, 0);
                if (e == hipSuccess) e = hipDeviceSynchronize();
                if (e != hipSuccess) { fprintf(stderr, "expert matvec: %s\n", hipGetErrorString(e)); return 1; }
                for (int r = 0; r < T * NU; ++r) S3[k]->add(g_y + (size_t)r * FE, &(*R3[k])[(size_t)r * FE], FE);
            }
            std::vector<float> act((size_t)FE);
            for (int r = 0; r < T * NU; ++r) {
                for (int i = 0; i < FE; ++i) act[i] = qwen_silu(mg[(size_t)r * FE + i]) * mu[(size_t)r * FE + i];
                sesw.add(act.data(), &msw[(size_t)r * FE], FE);
            }
            memcpy(g_x, msw.data(), (size_t)T * NU * FE * sizeof(float));
            e = qmatvec_quantize(g_x, T * NU * FE, &g_q8, 0);
            if (e == hipSuccess)
                e = qmatvec_launch_q8_id(L.We_down.type, L.We_down.data, L.We_down.row_bytes, gids, NU, 1, &g_q8, g_y, E,
                                         FE, T, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { fprintf(stderr, "expert down: %s\n", hipGetErrorString(e)); return 1; }
            for (int r = 0; r < T * NU; ++r) sed.add(g_y + (size_t)r * E, &md[(size_t)r * E], E);

            // The same expert projections through the prefill path: entries grouped by expert on the GPU
            // (gpu_moe_group), then the expert-tiled batched kernels (WMMA where the type has one)
            Stat sbg, sbu, sbd;
            {
                const int tile = qmatvec_mm_tile(), mt = gpu_moe_max_tiles(T, NU, NE, tile);
                ShmBuf b_gt, b_gp, b_gn, b_ge, b_gs, b_gc;
                if (!b_gt.alloc("fwdv_moe_gt.bin", (size_t)T * NU) || !b_gp.alloc("fwdv_moe_gp.bin", (size_t)T * NU) ||
                    !b_gn.alloc("fwdv_moe_gn.bin", 1) || !b_ge.alloc("fwdv_moe_ge.bin", mt) ||
                    !b_gs.alloc("fwdv_moe_gs.bin", mt) || !b_gc.alloc("fwdv_moe_gc.bin", mt)) {
                    fprintf(stderr, "alloc failed\n");
                    return 1;
                }
                int *gt = (int*)b_gt.p, *gp = (int*)b_gp.p, *gn = (int*)b_gn.p, *ge = (int*)b_ge.p,
                    *gs = (int*)b_gs.p, *gc = (int*)b_gc.p;
                e = gpu_moe_group(gids, T, NU, NE, tile, gt, gp, gn, ge, gs, gc, 0);
                for (int k = 0; k < 2 && e == hipSuccess; ++k) {
                    memcpy(g_x, b.data(), (size_t)T * E * sizeof(float));
                    e = qmatvec_quantize(g_x, T * E, &g_q8, 0);
                    if (e == hipSuccess)
                        e = qmatvec_launch_q8_id_batch(W3[k]->type, W3[k]->data, W3[k]->row_bytes, &g_q8, g_y, FE, E, mt,
                                                       gn, ge, gs, gc, gt, gp, 0);
                    if (e == hipSuccess) e = hipDeviceSynchronize();
                    if (e == hipSuccess)
                        for (int r = 0; r < T * NU; ++r)
                            (k ? sbu : sbg).add(g_y + (size_t)r * FE, &(*R3[k])[(size_t)r * FE], FE);
                }
                if (e == hipSuccess) {
                    memcpy(g_x, msw.data(), (size_t)T * NU * FE * sizeof(float));
                    e = qmatvec_quantize(g_x, T * NU * FE, &g_q8, 0);
                }
                if (e == hipSuccess)
                    e = qmatvec_launch_q8_id_batch(L.We_down.type, L.We_down.data, L.We_down.row_bytes, &g_q8, g_y, E, FE,
                                                   mt, gn, ge, gs, gc, gp, gp, 0);
                if (e == hipSuccess) e = hipDeviceSynchronize();
                if (e != hipSuccess) { fprintf(stderr, "grouped expert matmul: %s\n", hipGetErrorString(e)); return 1; }
                for (int r = 0; r < T * NU; ++r) sbd.add(g_y + (size_t)r * E, &md[(size_t)r * E], E);
            }

            // Shared expert
            std::vector<float> sact((size_t)FS);
            Stat sssw;
            for (int t = 0; t < T; ++t) {
                if (!proj(L.W_gate, &b[(size_t)t * E], &shg[(size_t)t * FS], ssg) ||
                    !proj(L.W_up, &b[(size_t)t * E], &shu[(size_t)t * FS], ssu))
                    return 1;
                for (int i = 0; i < FS; ++i) sact[i] = qwen_silu(shg[(size_t)t * FS + i]) * shu[(size_t)t * FS + i];
                sssw.add(sact.data(), &shsw[(size_t)t * FS], FS);
                if (!proj(L.W_down, &shsw[(size_t)t * FS], &shd[(size_t)t * E], ssd)) return 1;
            }

            // Combine on the reference expert outputs, weights, shared output and gate
            memcpy(b_d.p, md.data(), (size_t)T * NU * E * sizeof(float));
            memcpy(b_w.p, wn.data(), (size_t)T * NU * sizeof(float));
            memcpy(b_sh.p, shd.data(), (size_t)T * E * sizeof(float));
            memcpy(b_sg.p, sgs.data(), (size_t)T * sizeof(float));
            e = gpu_moe_combine(b_d.p, b_w.p, b_sh.p, b_sg.p, b_out.p, NU, E, T, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { fprintf(stderr, "moe_combine: %s\n", hipGetErrorString(e)); return 1; }
            scomb.add(b_out.p, fo.data(), T * E);

            printf("  (blk.0 types: router %d, experts gate %d up %d down %d, shared %d/%d/%d)\n", L.W_router.type,
                   L.We_gate.type, L.We_up.type, L.We_down.type, L.W_gate.type, L.W_up.type, L.W_down.type);
            report("blk.0 router logits (F32)", srl);
            const bool ids_ok = id_bad == 0;
            if (!ids_ok) ++g_fail;
            printf("  [%s] blk.0 router: top-%d experts for %d tokens   %d of %d differ from the reference\n",
                   ids_ok ? "PASS" : "FAIL", NU, T, id_bad, T * NU);
            report("blk.0 router: renormalized weights", sw8);
            report("blk.0 router: shared-expert gate", ssgs);
            report("blk.0 experts gate (int8, by id)", seg);
            report("blk.0 experts up (int8, by id)", seu);
            report("blk.0 experts SwiGLU", sesw);
            report("blk.0 experts down (int8, by id)", sed);
            report("blk.0 experts gate, grouped prefill path", sbg);
            report("blk.0 experts up, grouped prefill path", sbu);
            report("blk.0 experts down, grouped prefill path", sbd);
            report("blk.0 shared expert gate (qmatvec)", ssg);
            report("blk.0 shared expert up (qmatvec)", ssu);
            report("blk.0 shared expert SwiGLU", sssw);
            report("blk.0 shared expert down (qmatvec)", ssd);
            report("blk.0 MoE combine (ffn_out)", scomb);
        }
    }

    // ---- first Gated DeltaNet layer: projections, conv1d with state, L2 norm, gates, delta rule, gated norm
    int ll = -1;
    for (int l = 0; l < m->n_layer && ll < 0; ++l)
        if (m->layers[l].kind == LAYER_LINEAR_ATTN) ll = l;
    if (ll >= 0) {
        const LayerWeights& L = m->layers[ll];
        auto nm = [&](const char* base) { return std::string(base) + "-" + std::to_string(ll); };
        const int DS = m->ssm_d_state, NG = m->ssm_n_group, NV = m->ssm_dt_rank, DI = m->ssm_d_inner;
        const int KD = NG * DS, CD = 2 * KD + DI, DC = m->ssm_d_conv;
        std::vector<float> xn, qkv, cst, csilu, qn, kn, alpha, gate, beta, bsig, st0, st1, ao, nrm, z, fo, lo;
        if (!ref(nm("attn_norm").c_str(), nullptr, xn) || !ref(nm("linear_attn_qkv_mixed").c_str(), nullptr, qkv) ||
            !ref(nm("conv_states").c_str(), nullptr, cst) || !ref(nm("conv_output_silu").c_str(), nullptr, csilu) ||
            !ref(nm("q_conv_predelta").c_str(), nullptr, qn) || !ref(nm("k_conv_predelta").c_str(), nullptr, kn) ||
            !ref(nm("alpha").c_str(), nullptr, alpha) || !ref(nm("gate").c_str(), "MUL", gate) ||
            !ref(nm("beta").c_str(), nullptr, beta) || !ref(nm("beta_sigmoid").c_str(), nullptr, bsig) ||
            !ref(nm("state_predelta").c_str(), nullptr, st0) || !ref(nm("new_state").c_str(), nullptr, st1) ||
            !ref(nm("attn_output").c_str(), nullptr, ao) || !ref(nm("norm").c_str(), "RMS_NORM", nrm, nullptr, DS) ||
            !ref(nm("z").c_str(), "MUL_MAT", z) || !ref(nm("final_output").c_str(), nullptr, fo) ||
            !ref(nm("linear_attn_out").c_str(), "MUL_MAT", lo))
            return 1;
        if ((int)cst.size() != CD * (DC - 1) || (int)st0.size() != NV * DS * DS) {
            fprintf(stderr, "blk.%d: unexpected recurrent state sizes %zu %zu\n", ll, cst.size(), st0.size());
            return 1;
        }

        Stat sconv, sqn, skn, sgate, sbeta, sdelta, sstate, snorm, sfo;
        ProjStat sqkv, sz, sal, sbe, sout;
        std::vector<float> conv(CD), h(DS), gt(NV), bt(NV), o(DI), fin(DI);
        std::vector<float> cs = cst, S = st0;  // carried across tokens, starting from the reference state
        for (int t = 0; t < T; ++t) {
            const float* xt = &xn[(size_t)t * E];
            if (!proj(L.W_qkv, xt, &qkv[(size_t)t * CD], sqkv) || !proj(L.W_z, xt, &z[(size_t)t * DI], sz) ||
                !proj(L.W_alpha, xt, &alpha[(size_t)t * NV], sal) || !proj(L.W_beta, xt, &beta[(size_t)t * NV], sbe))
                return 1;

            // conv1d + SiLU on the reference qkv, our own conv state
            qwen_ssm_conv_step(cs.data(), &qkv[(size_t)t * CD], L.ssm_conv1d, conv.data(), CD, DC);
            for (int i = 0; i < CD; ++i) conv[i] = qwen_silu(conv[i]);
            sconv.add(conv.data(), &csilu[(size_t)t * CD], CD);

            // L2 norm of the reference conv output's q and k heads
            for (int hh = 0; hh < NG; ++hh) {
                qwen_l2norm(&csilu[(size_t)t * CD + (size_t)hh * DS], h.data(), DS, eps);
                sqn.add(h.data(), &qn[((size_t)t * NG + hh) * DS], DS);
                qwen_l2norm(&csilu[(size_t)t * CD + KD + (size_t)hh * DS], h.data(), DS, eps);
                skn.add(h.data(), &kn[((size_t)t * NG + hh) * DS], DS);
            }

            for (int hh = 0; hh < NV; ++hh) {
                gt[hh] = qwen_softplus(alpha[(size_t)t * NV + hh] + L.ssm_dt[hh]) * L.ssm_a[hh];
                bt[hh] = qwen_sigmoid(beta[(size_t)t * NV + hh]);
            }
            sgate.add(gt.data(), &gate[(size_t)t * NV], NV);
            sbeta.add(bt.data(), &bsig[(size_t)t * NV], NV);

            // Delta rule on the reference q/k/v/gate/beta, our own state
            qwen_gated_delta_step(S.data(), &qn[(size_t)t * KD], &kn[(size_t)t * KD], &csilu[(size_t)t * CD + 2 * KD],
                                  &gate[(size_t)t * NV], &bsig[(size_t)t * NV], NV, NG, DS, o.data());
            sdelta.add(o.data(), &ao[(size_t)t * DI], DI);

            // Gated RMSNorm on the reference delta output and z
            for (int hh = 0; hh < NV; ++hh) {
                const float* src = &ao[(size_t)t * DI + (size_t)hh * DS];
                qwen_rmsnorm(src, nullptr, h.data(), DS, eps);
                snorm.add(h.data(), &nrm[(size_t)t * DI + (size_t)hh * DS], DS);
                qwen_rmsnorm(src, L.ssm_norm, &fin[(size_t)hh * DS], DS, eps);
                for (int i = 0; i < DS; ++i)
                    fin[(size_t)hh * DS + i] *= qwen_silu(z[(size_t)t * DI + (size_t)hh * DS + i]);
            }
            sfo.add(fin.data(), &fo[(size_t)t * DI], DI);

            if (!proj(L.W_ssm_out, &fo[(size_t)t * DI], &lo[(size_t)t * E], sout)) return 1;
        }
        sstate.add(S.data(), st1.data(), NV * DS * DS);

        printf("  (blk.%d types: attn_qkv %d, attn_gate %d, ssm_alpha %d, ssm_beta %d, ssm_out %d)\n", ll,
               L.W_qkv.type, L.W_z.type, L.W_alpha.type, L.W_beta.type, L.W_ssm_out.type);
        char what[64];
        snprintf(what, sizeof(what), "blk.%d attn_qkv (qmatvec)", ll); report(what, sqkv);
        snprintf(what, sizeof(what), "blk.%d attn_gate z (qmatvec)", ll); report(what, sz);
        snprintf(what, sizeof(what), "blk.%d ssm_alpha (qmatvec)", ll); report(what, sal);
        snprintf(what, sizeof(what), "blk.%d ssm_beta (qmatvec)", ll); report(what, sbe);
        snprintf(what, sizeof(what), "blk.%d conv1d + SiLU (with state)", ll); report(what, sconv);
        snprintf(what, sizeof(what), "blk.%d q L2 norm", ll); report(what, sqn);
        snprintf(what, sizeof(what), "blk.%d k L2 norm", ll); report(what, skn);
        snprintf(what, sizeof(what), "blk.%d decay gate (softplus * a)", ll); report(what, sgate);
        snprintf(what, sizeof(what), "blk.%d beta sigmoid", ll); report(what, sbeta);
        snprintf(what, sizeof(what), "blk.%d delta rule output", ll); report(what, sdelta);
        snprintf(what, sizeof(what), "blk.%d delta rule final state", ll); report(what, sstate);
        snprintf(what, sizeof(what), "blk.%d ssm_norm (no gamma)", ll); report(what, snorm);
        snprintf(what, sizeof(what), "blk.%d gated norm (final_output)", ll); report(what, sfo);
        snprintf(what, sizeof(what), "blk.%d ssm_out (qmatvec)", ll); report(what, sout);

        // The engine's GPU kernels (deltanet.hip) on the same reference inputs, state carried across
        // tokens from the reference's initial state, mapped to the kernels' layout (deltanet_state_index).
        if (DS != DELTANET_HEAD_SIZE || deltanet_init("deltanet.co") != 0) {
            fprintf(stderr, "deltanet kernels unavailable\n");
            return 1;
        }
        ShmBuf b_qkv, b_cs, b_out, b_al, b_be, b_z, b_st, b_lin;
        if (!b_qkv.alloc("fwdv_dn_qkv.bin", CD) || !b_cs.alloc("fwdv_dn_cs.bin", (size_t)CD * (DC - 1)) ||
            !b_out.alloc("fwdv_dn_out.bin", CD) || !b_al.alloc("fwdv_dn_al.bin", NV) || !b_be.alloc("fwdv_dn_be.bin", NV) ||
            !b_z.alloc("fwdv_dn_z.bin", DI) || !b_st.alloc("fwdv_dn_st.bin", (size_t)NV * DS * DS) ||
            !b_lin.alloc("fwdv_dn_lin.bin", DI)) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        memcpy(b_cs.p, cst.data(), cst.size() * sizeof(float));
        for (int hh = 0; hh < NV; ++hh)
            for (int i = 0; i < DS; ++i)
                for (int j = 0; j < DS; ++j)
                    b_st.p[deltanet_state_index(hh, i, j)] = st0[((size_t)hh * DS + j) * DS + i];
        Stat gq, gk, gv, gg, gb, gfo, gst;
        for (int t = 0; t < T; ++t) {
            memcpy(b_qkv.p, &qkv[(size_t)t * CD], CD * sizeof(float));
            memcpy(b_al.p, &alpha[(size_t)t * NV], NV * sizeof(float));
            memcpy(b_be.p, &beta[(size_t)t * NV], NV * sizeof(float));
            memcpy(b_z.p, &z[(size_t)t * DI], DI * sizeof(float));
            hipError_t e = deltanet_pre(b_qkv.p, b_cs.p, L.ssm_conv1d, DC, b_out.p, NG, NV, b_al.p, b_be.p, L.ssm_dt,
                                        L.ssm_a, b_al.p, b_be.p, eps, 1, 0);
            if (e == hipSuccess)
                e = deltanet_rule(b_st.p, b_out.p, b_al.p, b_be.p, b_z.p, L.ssm_norm, b_lin.p, NG, NV, eps, 1, nullptr, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { fprintf(stderr, "deltanet kernels: %s\n", hipGetErrorString(e)); return 1; }
            gq.add(b_out.p, &qn[(size_t)t * KD], KD);
            gk.add(b_out.p + KD, &kn[(size_t)t * KD], KD);
            gv.add(b_out.p + 2 * KD, &csilu[(size_t)t * CD + 2 * KD], DI);
            gg.add(b_al.p, &gate[(size_t)t * NV], NV);
            gb.add(b_be.p, &bsig[(size_t)t * NV], NV);
            gfo.add(b_lin.p, &fo[(size_t)t * DI], DI);
        }
        std::vector<float> st_t((size_t)NV * DS * DS);
        for (int hh = 0; hh < NV; ++hh)
            for (int i = 0; i < DS; ++i)
                for (int j = 0; j < DS; ++j)
                    st_t[((size_t)hh * DS + j) * DS + i] = b_st.p[deltanet_state_index(hh, i, j)];
        gst.add(st_t.data(), st1.data(), NV * DS * DS);
        snprintf(what, sizeof(what), "blk.%d GPU conv + SiLU + L2: q", ll); report(what, gq);
        snprintf(what, sizeof(what), "blk.%d GPU conv + SiLU + L2: k", ll); report(what, gk);
        snprintf(what, sizeof(what), "blk.%d GPU conv + SiLU: v", ll); report(what, gv);
        snprintf(what, sizeof(what), "blk.%d GPU decay gate", ll); report(what, gg);
        snprintf(what, sizeof(what), "blk.%d GPU beta sigmoid", ll); report(what, gb);
        snprintf(what, sizeof(what), "blk.%d GPU delta rule + gated norm", ll); report(what, gfo);
        snprintf(what, sizeof(what), "blk.%d GPU delta rule final state", ll); report(what, gst);

        // The same kernels on all T tokens in one call (prefill), from the reference's initial state
        {
            ShmBuf c_qkv, c_out, c_al, c_be, c_z, c_lin;
            if (!c_qkv.alloc("fwdv_dnb_qkv.bin", (size_t)T * CD) || !c_out.alloc("fwdv_dnb_out.bin", (size_t)T * CD) ||
                !c_al.alloc("fwdv_dnb_al.bin", (size_t)T * NV) || !c_be.alloc("fwdv_dnb_be.bin", (size_t)T * NV) ||
                !c_z.alloc("fwdv_dnb_z.bin", (size_t)T * DI) || !c_lin.alloc("fwdv_dnb_lin.bin", (size_t)T * DI)) {
                fprintf(stderr, "alloc failed\n");
                return 1;
            }
            memcpy(b_cs.p, cst.data(), cst.size() * sizeof(float));
            for (int hh = 0; hh < NV; ++hh)
                for (int i = 0; i < DS; ++i)
                    for (int j = 0; j < DS; ++j)
                        b_st.p[deltanet_state_index(hh, i, j)] = st0[((size_t)hh * DS + j) * DS + i];
            memcpy(c_qkv.p, qkv.data(), (size_t)T * CD * sizeof(float));
            memcpy(c_al.p, alpha.data(), (size_t)T * NV * sizeof(float));
            memcpy(c_be.p, beta.data(), (size_t)T * NV * sizeof(float));
            memcpy(c_z.p, z.data(), (size_t)T * DI * sizeof(float));
            hipError_t e = deltanet_pre(c_qkv.p, b_cs.p, L.ssm_conv1d, DC, c_out.p, NG, NV, c_al.p, c_be.p, L.ssm_dt,
                                        L.ssm_a, c_al.p, c_be.p, eps, T, 0);
            if (e == hipSuccess)
                e = deltanet_rule(b_st.p, c_out.p, c_al.p, c_be.p, c_z.p, L.ssm_norm, c_lin.p, NG, NV, eps, T, nullptr, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { fprintf(stderr, "batched deltanet kernels: %s\n", hipGetErrorString(e)); return 1; }
            Stat bq, bfo, bst;
            for (int t = 0; t < T; ++t) {
                bq.add(c_out.p + (size_t)t * CD, &qn[(size_t)t * KD], KD);
                bfo.add(c_lin.p + (size_t)t * DI, &fo[(size_t)t * DI], DI);
            }
            for (int hh = 0; hh < NV; ++hh)
                for (int i = 0; i < DS; ++i)
                    for (int j = 0; j < DS; ++j)
                        st_t[((size_t)hh * DS + j) * DS + i] = b_st.p[deltanet_state_index(hh, i, j)];
            bst.add(st_t.data(), st1.data(), NV * DS * DS);
            snprintf(what, sizeof(what), "blk.%d GPU batched (%d tok) conv+L2: q", ll, T); report(what, bq);
            snprintf(what, sizeof(what), "blk.%d GPU batched (%d tok) output", ll, T); report(what, bfo);
            snprintf(what, sizeof(what), "blk.%d GPU batched (%d tok) state", ll, T); report(what, bst);
        }
    }

    // ---- first full-attention layer: projections, q/k norm, RoPE, attention, gate, output
    int la = -1;
    for (int l = 0; l < m->n_layer && la < 0; ++l)
        if (m->layers[l].kind == LAYER_FULL_ATTN) la = l;
    if (la >= 0) {
        const LayerWeights& L = m->layers[la];
        auto nm = [&](const char* base) {
            static char buf[64];
            snprintf(buf, sizeof(buf), "%s-%d", base, la);
            return std::string(buf);
        };
        const int DQ = HQ * HD, DKV = HKV * HD;
        std::vector<float> xn, qf, kc, vc, qn, qr, kn, kr, pre, gs, gated, ao;
        if (!ref(nm("attn_norm").c_str(), nullptr, xn) || !ref(nm("Qcur_full").c_str(), nullptr, qf) ||
            !ref(nm("Kcur").c_str(), "MUL_MAT", kc) || !ref(nm("Vcur").c_str(), "MUL_MAT", vc) ||
            !ref(nm("Qcur_normed").c_str(), nullptr, qn) || !ref(nm("Qcur").c_str(), "ROPE", qr) ||
            !ref(nm("Kcur_normed").c_str(), nullptr, kn) || !ref(nm("Kcur").c_str(), "ROPE", kr) ||
            !ref(nm("attn_pregate").c_str(), nullptr, pre) || !ref(nm("gate_sigmoid").c_str(), nullptr, gs) ||
            !ref(nm("attn_gated").c_str(), nullptr, gated) || !ref(nm("attn_output").c_str(), nullptr, ao))
            return 1;

        Stat sqn, sqr, skn, skr, sat, sgs, sg;
        ProjStat sq, sk, sv, so;
        std::vector<float> h(HD), att(DQ), gate(DQ);
        std::vector<__half> Kh((size_t)T * DKV), Vh((size_t)T * DKV);
        for (int t = 0; t < T; ++t) {
            const float* xt = &xn[(size_t)t * E];
            if (!proj(L.Wq, xt, &qf[(size_t)t * 2 * DQ], sq) || !proj(L.Wk, xt, &kc[(size_t)t * DKV], sk) ||
                !proj(L.Wv, xt, &vc[(size_t)t * DKV], sv))
                return 1;

            // Q: split [Q | gate] per head, RMSNorm with q_norm; RoPE on the reference normed Q
            for (int hh = 0; hh < HQ; ++hh) {
                const float* src = &qf[(size_t)t * 2 * DQ + (size_t)hh * 2 * HD];
                qwen_rmsnorm(src, L.q_norm, h.data(), HD, eps);
                sqn.add(h.data(), &qn[(size_t)t * DQ + (size_t)hh * HD], HD);
                memcpy(h.data(), &qn[(size_t)t * DQ + (size_t)hh * HD], HD * sizeof(float));
                qwen_rope_neox(h.data(), m->n_rot, t, m->rope_freq_base);
                sqr.add(h.data(), &qr[(size_t)t * DQ + (size_t)hh * HD], HD);
                for (int i = 0; i < HD; ++i) gate[(size_t)hh * HD + i] = qwen_sigmoid(src[HD + i]);
            }
            sgs.add(gate.data(), &gs[(size_t)t * DQ], DQ);
            for (int g = 0; g < HKV; ++g) {
                qwen_rmsnorm(&kc[(size_t)t * DKV + (size_t)g * HD], L.k_norm, h.data(), HD, eps);
                skn.add(h.data(), &kn[(size_t)t * DKV + (size_t)g * HD], HD);
                memcpy(h.data(), &kn[(size_t)t * DKV + (size_t)g * HD], HD * sizeof(float));
                qwen_rope_neox(h.data(), m->n_rot, t, m->rope_freq_base);
                skr.add(h.data(), &kr[(size_t)t * DKV + (size_t)g * HD], HD);
            }

            // Attention over reference K/V (stored as FP16, like both KV caches), causal up to t
            for (int i = 0; i < DKV; ++i) {
                Kh[(size_t)t * DKV + i] = __float2half(kr[(size_t)t * DKV + i]);
                Vh[(size_t)t * DKV + i] = __float2half(vc[(size_t)t * DKV + i]);
            }
            qwen_attention(&qr[(size_t)t * DQ], Kh.data(), Vh.data(), DKV, t + 1, HQ, HKV, HD,
                           1.0f / sqrtf((float)HD), att.data());
            sat.add(att.data(), &pre[(size_t)t * DQ], DQ);
            for (int i = 0; i < DQ; ++i) att[i] = pre[(size_t)t * DQ + i] * gs[(size_t)t * DQ + i];
            sg.add(att.data(), &gated[(size_t)t * DQ], DQ);

            if (!proj(L.Wo, &gated[(size_t)t * DQ], &ao[(size_t)t * E], so)) return 1;
        }
        printf("  (blk.%d types: attn_q %d, attn_k %d, attn_v %d, attn_output %d)\n", la, L.Wq.type, L.Wk.type,
               L.Wv.type, L.Wo.type);
        char what[64];
        snprintf(what, sizeof(what), "blk.%d attn_q [Q|gate] (qmatvec)", la); report(what, sq);
        snprintf(what, sizeof(what), "blk.%d attn_k (qmatvec)", la); report(what, sk);
        snprintf(what, sizeof(what), "blk.%d attn_v (qmatvec)", la); report(what, sv);
        snprintf(what, sizeof(what), "blk.%d Q split + q_norm", la); report(what, sqn);
        snprintf(what, sizeof(what), "blk.%d Q RoPE", la); report(what, sqr);
        snprintf(what, sizeof(what), "blk.%d K k_norm", la); report(what, skn);
        snprintf(what, sizeof(what), "blk.%d K RoPE", la); report(what, skr);
        // ggml's CPU FLASH_ATTN_EXT casts Q to FP16 for the K dot and accumulates softmax*V in an FP16
        // buffer (VKQ16); qwen_attention keeps both in FP32, hence the looser bound here.
        snprintf(what, sizeof(what), "blk.%d attention (FP16 KV)", la); report(what, sat, 2e-3);
        snprintf(what, sizeof(what), "blk.%d gate split + sigmoid", la); report(what, sgs);
        snprintf(what, sizeof(what), "blk.%d gated output", la); report(what, sg);
        snprintf(what, sizeof(what), "blk.%d attn_output (qmatvec)", la); report(what, so);
    }

    // ---- GPU attention kernels vs the CPU reference ops at a long context (10 256-position chunks: more than
    // ATTN_MAX_SPLITS, so attn_flash splits the positions and attn_partial runs in token groups).
    // Random K/V history for positions 0..P-2; position P-1 goes through attn_prep (norm, RoPE, KV write)
    // from random projections, with the first attention layer's q/k norm weights.
    if (la >= 0) {
        const LayerWeights& L = m->layers[la];
        const int P = 2500, DQ = HQ * HD, DKV = HKV * HD, pos = P - 1;
        if (HD != ATTN_HEAD_DIM || attention_init("attention.co") != 0) { fprintf(stderr, "attention kernels unavailable\n"); return 1; }
        ShmBuf b_qf, b_k, b_v, b_q, b_ml, b_acc, b_out, b_K, b_V;
        if (!b_qf.alloc("fwdv_at_qf.bin", 2 * DQ) || !b_k.alloc("fwdv_at_k.bin", DKV) || !b_v.alloc("fwdv_at_v.bin", DKV) ||
            !b_q.alloc("fwdv_at_q.bin", DQ) || !b_ml.alloc("fwdv_at_ml.bin", attention_part_ml_floats(HQ, P, 1)) ||
            !b_acc.alloc("fwdv_at_acc.bin", attention_part_acc_floats(HQ, P, 1)) || !b_out.alloc("fwdv_at_out.bin", DQ) ||
            !b_K.alloc("fwdv_at_K.bin", (size_t)P * DKV / 2 + 1) || !b_V.alloc("fwdv_at_V.bin", (size_t)P * DKV / 2 + 1)) {
            fprintf(stderr, "alloc failed\n");
            return 1;
        }
        __half* K = (__half*)b_K.p;
        __half* Vc = (__half*)b_V.p;
        uint32_t rng = 12345;
        auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f; };
        for (size_t i = 0; i < (size_t)(P - 1) * DKV; ++i) { K[i] = __float2half(rnd()); Vc[i] = __float2half(rnd()); }
        for (int i = 0; i < 2 * DQ; ++i) b_qf.p[i] = 3.0f * rnd();
        for (int i = 0; i < DKV; ++i) { b_k.p[i] = 3.0f * rnd(); b_v.p[i] = rnd(); }

        // CPU reference first (the kernel writes row pos of K/V itself)
        std::vector<float> q(DQ), kk(DKV), ref_out(DQ);
        std::vector<__half> Kr(K, K + (size_t)P * DKV), Vr(Vc, Vc + (size_t)P * DKV);
        for (int h = 0; h < HQ; ++h) {
            qwen_rmsnorm(&b_qf.p[(size_t)h * 2 * HD], L.q_norm, &q[(size_t)h * HD], HD, eps);
            qwen_rope_neox(&q[(size_t)h * HD], m->n_rot, pos, m->rope_freq_base);
        }
        for (int g = 0; g < HKV; ++g) {
            qwen_rmsnorm(&b_k.p[(size_t)g * HD], L.k_norm, &kk[(size_t)g * HD], HD, eps);
            qwen_rope_neox(&kk[(size_t)g * HD], m->n_rot, pos, m->rope_freq_base);
        }
        for (int i = 0; i < DKV; ++i) {
            Kr[(size_t)pos * DKV + i] = __float2half(kk[i]);
            Vr[(size_t)pos * DKV + i] = __float2half(b_v.p[i]);
        }
        qwen_attention(q.data(), Kr.data(), Vr.data(), DKV, P, HQ, HKV, HD, 1.0f / sqrtf((float)HD), ref_out.data());
        for (int h = 0; h < HQ; ++h)
            for (int d = 0; d < HD; ++d) ref_out[(size_t)h * HD + d] *= qwen_sigmoid(b_qf.p[(size_t)h * 2 * HD + HD + d]);

        hipError_t e = attention_forward(b_qf.p, b_k.p, b_v.p, L.q_norm, L.k_norm, K, Vc, HQ, HKV, m->n_rot, pos, 1,
                                         m->rope_freq_base, eps, b_q.p, b_ml.p, b_acc.p, b_out.p, nullptr, 0);
        if (e == hipSuccess) e = hipDeviceSynchronize();
        if (e != hipSuccess) { fprintf(stderr, "attention kernels: %s\n", hipGetErrorString(e)); return 1; }
        Stat sq, skv, so;
        sq.add(b_q.p, q.data(), DQ);
        std::vector<float> kg(DKV), kr(DKV);
        for (int i = 0; i < DKV; ++i) {
            kg[i] = __half2float(K[(size_t)pos * DKV + i]);
            kr[i] = __half2float(Kr[(size_t)pos * DKV + i]);
        }
        skv.add(kg.data(), kr.data(), DKV);
        so.add(b_out.p, ref_out.data(), DQ);
        char what[80];
        // RoPE angles: GPU float pos * base^(-2i/n_rot) (as ggml), CPU reference double; at pos 2499 that
        // differs by ~1e-4 rad, hence 1e-3 on the rotated values
        snprintf(what, sizeof(what), "GPU attn_prep Q norm + RoPE (pos %d)", pos); report(what, sq, 1e-3);
        snprintf(what, sizeof(what), "GPU attn_prep K row written to cache"); report(what, skv, 1e-3);
        snprintf(what, sizeof(what), "GPU attention, %d positions (%d chunks), gated", P, (P + ATTN_CHUNK - 1) / ATTN_CHUNK);
        report(what, so, 2e-3);

        // Batched: TB tokens at positions pos0 .. P - 1 in one call (prefill). Each token's projections are
        // random; the reference runs the CPU ops token by token, causally (token t sees positions <= pos0 + t).
        // TB = 8 runs attn_partial; TB = 40 runs attn_flash (FP16 WMMA), with a partial last token tile.
        for (const int TB : {8, 40}) {
            const int pos0 = P - TB;
            ShmBuf c_qf, c_k, c_v, c_q, c_ml, c_acc, c_out;
            if (!c_qf.alloc("fwdv_atb_qf.bin", (size_t)TB * 2 * DQ) || !c_k.alloc("fwdv_atb_k.bin", (size_t)TB * DKV) ||
                !c_v.alloc("fwdv_atb_v.bin", (size_t)TB * DKV) || !c_q.alloc("fwdv_atb_q.bin", (size_t)TB * DQ) ||
                !c_ml.alloc("fwdv_atb_ml.bin", attention_part_ml_floats(HQ, P, TB)) ||
                !c_acc.alloc("fwdv_atb_acc.bin", attention_part_acc_floats(HQ, P, TB)) ||
                !c_out.alloc("fwdv_atb_out.bin", (size_t)TB * DQ)) {
                fprintf(stderr, "alloc failed\n");
                return 1;
            }
            for (size_t i = 0; i < (size_t)TB * 2 * DQ; ++i) c_qf.p[i] = 3.0f * rnd();
            for (size_t i = 0; i < (size_t)TB * DKV; ++i) { c_k.p[i] = 3.0f * rnd(); c_v.p[i] = rnd(); }
            // reference K/V rows for the batch positions, then attention per token
            for (int t = 0; t < TB; ++t) {
                std::vector<float> kt(DKV);
                for (int g = 0; g < HKV; ++g) {
                    qwen_rmsnorm(&c_k.p[(size_t)t * DKV + (size_t)g * HD], L.k_norm, &kt[(size_t)g * HD], HD, eps);
                    qwen_rope_neox(&kt[(size_t)g * HD], m->n_rot, pos0 + t, m->rope_freq_base);
                }
                for (int i = 0; i < DKV; ++i) {
                    Kr[(size_t)(pos0 + t) * DKV + i] = __float2half(kt[i]);
                    Vr[(size_t)(pos0 + t) * DKV + i] = __float2half(c_v.p[(size_t)t * DKV + i]);
                }
            }
            std::vector<float> refb((size_t)TB * DQ), qt(DQ);
            for (int t = 0; t < TB; ++t) {
                for (int h = 0; h < HQ; ++h) {
                    qwen_rmsnorm(&c_qf.p[(size_t)t * 2 * DQ + (size_t)h * 2 * HD], L.q_norm, &qt[(size_t)h * HD], HD, eps);
                    qwen_rope_neox(&qt[(size_t)h * HD], m->n_rot, pos0 + t, m->rope_freq_base);
                }
                qwen_attention(qt.data(), Kr.data(), Vr.data(), DKV, pos0 + t + 1, HQ, HKV, HD, 1.0f / sqrtf((float)HD),
                               &refb[(size_t)t * DQ]);
                for (int h = 0; h < HQ; ++h)
                    for (int d = 0; d < HD; ++d)
                        refb[(size_t)t * DQ + (size_t)h * HD + d] *=
                            qwen_sigmoid(c_qf.p[(size_t)t * 2 * DQ + (size_t)h * 2 * HD + HD + d]);
            }
            // The output is also quantized in the same launches (attn_combine, per token group); compare that
            // with quantizing it separately
            ShmBuf qa, qb;
            const int nq8 = TB * DQ;
            qmatvec_q8 q8a, q8b;
            if (!qa.alloc("fwdv_atb_q8a.bin", (qmatvec_q8_bytes(nq8) + 3) / 4) ||
                !qb.alloc("fwdv_atb_q8b.bin", (qmatvec_q8_bytes(nq8) + 3) / 4)) {
                fprintf(stderr, "alloc failed\n");
                return 1;
            }
            qmatvec_q8_bind(&q8a, qa.p, nq8);
            qmatvec_q8_bind(&q8b, qb.p, nq8);
            e = attention_forward(c_qf.p, c_k.p, c_v.p, L.q_norm, L.k_norm, K, Vc, HQ, HKV, m->n_rot, pos0, TB,
                                  m->rope_freq_base, eps, c_q.p, c_ml.p, c_acc.p, c_out.p, &q8a, 0);
            if (e == hipSuccess) e = qmatvec_quantize(c_out.p, nq8, &q8b, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { fprintf(stderr, "batched attention kernels: %s\n", hipGetErrorString(e)); return 1; }
            const bool q8_same = memcmp(qa.p, qb.p, qmatvec_q8_bytes(nq8)) == 0;
            if (!q8_same) ++g_fail;
            printf("  [%s] GPU attention batched, %d tok: output quantized in the same launches %s\n",
                   q8_same ? "PASS" : "FAIL", TB, q8_same ? "as qmatvec_quantize" : "DIFFERS from qmatvec_quantize");
            Stat sb;
            for (int t = 0; t < TB; ++t) sb.add(c_out.p + (size_t)t * DQ, &refb[(size_t)t * DQ], DQ);
            snprintf(what, sizeof(what), "GPU attention batched, %d tok at pos %d..%d", TB, pos0, P - 1);
            report(what, sb, 2e-3);
        }
        attention_free();
    }

    // ---- output norm + LM head on the last token
    {
        int64_t rl = 0;
        char lname[32];
        snprintf(lname, sizeof(lname), "l_out-%d", m->n_layer - 1);
        if (!ref(lname, nullptr, a, &rl) || !ref("result_norm", nullptr, b) || !ref("result_output", nullptr, c)) return 1;
        const size_t last = a.size() / (size_t)rl - 1;
        Stat sn;
        ProjStat sl;
        qwen_rmsnorm(&a[last * E], m->output_norm, y, E, eps);
        sn.add(y, b.data(), E);
        report("output_norm (last token)", sn);
        if (!proj(m->output, b.data(), c.data(), sl)) return 1;  // leaves the FP32-input logits in y
        printf("  (LM head type %d%s)\n", m->output.type, m->output.data == m->token_embd.data ? ", tied" : "");
        report("LM head (qmatvec)", sl);
        int best = 0;
        for (int i = 1; i < V; ++i)
            if (y[i] > y[best]) best = i;
        const bool ok = best == next_token;
        if (!ok) ++g_fail;
        printf("  [%s] LM head argmax on reference hidden state: %d (reference %d)\n", ok ? "PASS" : "FAIL", best,
               next_token);

        // Opt-in Q8_0 copy of a float head (inference_set_lm_head_q8): the GPU-built values and scales must be
        // ggml's quantize_row_q8_0_ref exactly (checked on a few rows), and the logits close to the reference
        const QuantWeight& H = m->output;
        if (qmatvec_q8_0s_source_supported(H.type, H.K)) {
            ShmBuf hq, hd;
            if (!hq.alloc("fwdv_hq.bin", (size_t)H.N * H.K / 4) || !hd.alloc("fwdv_hd.bin", (size_t)H.N * (H.K / 32) / 2 + 1)) {
                fprintf(stderr, "alloc failed\n");
                return 1;
            }
            int8_t* qs = (int8_t*)hq.p;
            uint16_t* qd = (uint16_t*)hd.p;
            memcpy(g_x, b.data(), (size_t)E * sizeof(float));
            hipError_t e = qmatvec_quantize_rows_q8_0(H.type, H.data, H.row_bytes, H.N, H.K, qs, qd, 0);
            if (e == hipSuccess) e = qmatvec_quantize(g_x, E, &g_q8, 0);
            if (e == hipSuccess) e = qmatvec_launch_q8_0s(qs, qd, &g_q8, g_y, H.N, H.K, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { fprintf(stderr, "LM head Q8_0 copy: %s\n", hipGetErrorString(e)); return 1; }
            int bad = 0;
            for (const int r : {0, H.N / 2, H.N - 1}) {
                const uint8_t* wr = (const uint8_t*)H.data + (size_t)r * H.row_bytes;
                for (int bl = 0; bl < H.K / 32; ++bl) {
                    float xs[32], amax = 0.0f;
                    for (int j = 0; j < 32; ++j) {
                        const int k = bl * 32 + j;
                        uint16_t h16;
                        memcpy(&h16, wr + 2 * (size_t)k, 2);
                        uint32_t u = (uint32_t)h16 << 16;
                        float bfv;
                        memcpy(&bfv, &u, 4);
                        xs[j] = H.type == QT_F32 ? ((const float*)wr)[k] : H.type == QT_F16 ? __half2float(*(const __half*)(wr + 2 * (size_t)k)) : bfv;
                        amax = std::max(amax, std::fabs(xs[j]));
                    }
                    const float dd = amax / 127.0f, id = dd != 0.0f ? 1.0f / dd : 0.0f;
                    const __half hdd = __float2half(dd);
                    if (memcmp(&qd[(size_t)r * (H.K / 32) + bl], &hdd, 2) != 0) ++bad;
                    for (int j = 0; j < 32; ++j)
                        if (qs[(size_t)r * H.K + bl * 32 + j] != (int8_t)(int)std::round(xs[j] * id)) ++bad;
                }
            }
            if (bad) ++g_fail;
            printf("  [%s] LM head Q8_0 copy: rows 0, N/2, N-1 as ggml's quantize_row_q8_0_ref (%d mismatches)\n",
                   bad ? "FAIL" : "PASS", bad);
            Stat sq;
            sq.add(g_y, c.data(), H.N);
            report("LM head Q8_0 copy (opt-in) vs reference logits", sq, 3e-2);
            int bq = 0;
            for (int i = 1; i < V; ++i)
                if (g_y[i] > g_y[bq]) bq = i;
            if (bq != next_token) ++g_fail;
            printf("  [%s] LM head Q8_0 copy argmax on reference hidden state: %d (reference %d)\n",
                   bq == next_token ? "PASS" : "FAIL", bq, next_token);
        }
    }

    // ---- end to end: the engine on the reference tokens, residual stream vs l_out-N after every layer
    {
        if (inference_kv_cache_init(m, 64) != 0) { fprintf(stderr, "kv init: %s\n", inference_get_last_error()); return 1; }
        qwen_model_t* qm = inference_get_qwen_model(m);
        struct Ctx { std::vector<float> l_out; int T; };
        Ctx ctx;
        ctx.T = T;
        ctx.l_out.resize((size_t)m->n_layer * T * E);
        qwen_model_set_layer_callback(qm, [](int l, int pos, const float* x, int n, void* u) {
            Ctx* c = (Ctx*)u;
            memcpy(&c->l_out[((size_t)l * c->T + pos) * n], x, (size_t)n * sizeof(float));
        }, &ctx);
        std::vector<float> logits(V);
        for (int t = 0; t < T; ++t)
            if (qwen_model_forward(qm, tokens[t], t, logits.data()) != 0) { fprintf(stderr, "forward failed\n"); return 1; }
        qwen_model_set_layer_callback(qm, nullptr, nullptr);

        double worst = 1.0;
        int worst_l = -1, n_bad = 0;
        for (int l = 0; l < m->n_layer; ++l) {
            if (!ref(("l_out-" + std::to_string(l)).c_str(), nullptr, a)) return 1;
            Stat s;
            for (int t = 0; t < T; ++t) s.add(&ctx.l_out[((size_t)l * T + t) * E], &a[(size_t)t * E], E);
            if (s.min_cos < worst) { worst = s.min_cos; worst_l = l; }
            if (s.min_cos <= 0.99) ++n_bad;
            if (verbose_layers())
                printf("    l_out-%-2d %-6s cos %.6f  max|d|/max|ref| %.2e\n", l,
                       m->layers[l].kind == LAYER_FULL_ATTN ? "attn" : "linear", s.min_cos, s.max_rel);
        }
        const bool ok = n_bad == 0;
        if (!ok) ++g_fail;
        printf("  [%s] engine l_out, all %d layers x %d tokens   worst cos %.6f (layer %d), %d below 0.99\n",
               ok ? "PASS" : "FAIL", m->n_layer, T, worst, worst_l, n_bad);

        int best = 0;
        for (int i = 1; i < V; ++i)
            if (logits[i] > logits[best]) best = i;
        const bool tok_ok = best == next_token;
        if (!tok_ok) ++g_fail;
        printf("  [%s] engine next token after the prompt: %d (reference %d)\n", tok_ok ? "PASS" : "FAIL", best,
               next_token);

        // Prefill: the same tokens in one qwen_model_prefill call. Every kernel computes each token exactly as
        // the one-at-a-time path does, so the residuals and logits must be identical, not just close.
        Ctx pctx;
        pctx.T = T;
        pctx.l_out.resize((size_t)m->n_layer * T * E);
        qwen_model_set_layer_callback(qm, [](int l, int pos, const float* x, int n, void* u) {
            Ctx* c = (Ctx*)u;
            memcpy(&c->l_out[((size_t)l * c->T + pos) * n], x, (size_t)n * sizeof(float));
        }, &pctx);
        std::vector<float> plogits(V);
        if (qwen_model_prefill(qm, tokens.data(), T, 0, plogits.data()) != 0) { fprintf(stderr, "prefill failed\n"); return 1; }
        qwen_model_set_layer_callback(qm, nullptr, nullptr);
        // With the WMMA prefill kernels (batches of 32+ tokens, or QMATVEC_WMMA_MIN_T for tests) the float
        // accumulation order differs. A ~1e-9 difference can flip an int8 activation rounding (one Q8 step) and,
        // in the MoE, a near-tie expert choice; both propagate (measured: 4B layer 0 3.7e-9, layer 1 8e-6; 35B
        // smooth to 1e-5, then a jump at a routing flip). ggml behaves the same under any reordering. So the check
        // becomes: every layer within cos 0.99 of token by token and of the reference, and the reference's top-1.
        double dmax = 0.0;
        for (size_t i = 0; i < ctx.l_out.size(); ++i) dmax = std::max(dmax, (double)std::fabs(pctx.l_out[i] - ctx.l_out[i]));
        double lmax = 0.0;
        for (int i = 0; i < V; ++i) lmax = std::max(lmax, (double)std::fabs(plogits[i] - logits[i]));
        double pcos = 1.0, rcos = 1.0;
        for (int l = 0; l < m->n_layer; ++l) {
            if (!ref(("l_out-" + std::to_string(l)).c_str(), nullptr, a)) return 1;
            Stat sp, sr;
            for (int t = 0; t < T; ++t) {
                sp.add(&pctx.l_out[((size_t)l * T + t) * E], &ctx.l_out[((size_t)l * T + t) * E], E);
                sr.add(&pctx.l_out[((size_t)l * T + t) * E], &a[(size_t)t * E], E);
            }
            pcos = std::min(pcos, sp.min_cos);
            rcos = std::min(rcos, sr.min_cos);
            if (verbose_layers())
                printf("    prefill l_out-%-2d %-6s 1 - cos vs token by token %.2e, max rel %.2e\n", l,
                       m->layers[l].kind == LAYER_FULL_ATTN ? "attn" : "linear", 1.0 - sp.min_cos, sp.max_rel);
        }
        int pbest = 0;
        for (int i = 1; i < V; ++i)
            if (plogits[i] > plogits[pbest]) pbest = i;
        const bool exact = dmax == 0.0 && lmax == 0.0;
        const bool pok = exact || (pcos >= 0.99 && rcos > 0.99 && pbest == next_token);
        if (!pok) ++g_fail;
        printf("  [%s] prefill (%d tokens, one call, max batch %d) vs token by token: %s max |d| l_out %.3g, logits %.3g;"
               " worst layer cos vs token by token %.6f, vs reference %.6f\n",
               pok ? "PASS" : "FAIL", T, qwen_model_max_batch(qm), exact ? "identical," : "", dmax, lmax, pcos, rcos);
    }

    deltanet_free();
    qmatvec_free();
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
