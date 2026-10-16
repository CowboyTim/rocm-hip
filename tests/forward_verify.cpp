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
// Element-wise ops (norms, RoPE, SwiGLU, gating) must agree to 1e-4 relative; attention to 2e-3, since
// the reference's flash attention rounds to FP16 internally.

#include "inference.h"
#include "inference_internal.h"
#include "qmatvec.h"
#include "quant_types.h"
#include "qwen_ops.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// First tensor with this name (and op, if given), read as rows of ne[0] floats
static bool ref(const char* name, const char* op, std::vector<float>& out, int64_t* row_len = nullptr) {
    for (const RefTensor& t : g_index) {
        if (t.name != name || (op && t.op != op)) continue;
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
    } else if (q80) {  // quantize_row_q8_0_ref: d = amax / 127 stored as FP16, q = roundf(x / d)
        for (int b = 0; b < K; b += 32) {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[b + j]));
            const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
            const float dh = __half2float(__float2half(d));
            for (int j = 0; j < 32; ++j) out[b + j] = dh * roundf(x[b + j] * id);
        }
    } else {
        memcpy(out, x, (size_t)K * sizeof(float));
    }
}

static int g_fail = 0;

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

// Projection: FP32 input (informational) and reference-quantized input (pass on max_rel)
struct ProjStat {
    Stat f32, emu;
};

static void report(const char* what, const ProjStat& s) {
    const bool ok = s.emu.max_rel <= 1e-4;
    if (!ok) ++g_fail;
    printf("  [%s] %-36s cos %.7f  max|d|/max|ref| %.2e   (FP32 input: cos %.7f, %.2e)\n", ok ? "PASS" : "FAIL",
           what, s.emu.min_cos, s.emu.max_rel, s.f32.min_cos, s.f32.max_rel);
}

static float* g_x = nullptr;  // pinned input
static float* g_y = nullptr;  // pinned output

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
    if (qmatvec_init("matvec_q.co") != 0) return 1;

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
    }

    qmatvec_free();
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
