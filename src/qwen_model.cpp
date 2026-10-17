// qwen_model.cpp -- Qwen3.5 single-token forward pass (Phase 20 Step 3c).
//
// Every projection (attention Q/K/V/O, FFN gate/up/down, LM head) is one qmatvec_launch on the raw GGUF
// blocks, read in place from the pinned file mapping. Activations live in one /dev/shm scratch buffer,
// pinned once, so the GPU reads and writes them without copies. The small per-token vector ops (norms,
// RoPE, single-query attention, SwiGLU) run on the CPU via qwen_ops for now.
//
// Both token mixers follow the llama.cpp qwen35 graph: gated full attention (gated Q, q/k RMSNorm, partial
// NeoX RoPE, GQA, sigmoid output gate) and Gated DeltaNet (conv1d with state, L2-normed q/k, gated delta
// rule over a d_state x d_state state per V head, gated RMSNorm). Their state persists across tokens: the
// KV cache for attention, RecurrentState (conv + delta-rule state) for DeltaNet. Phase 20 Steps 4-5.

#include "qwen_model.h"
#include "inference.h"
#include "inference_internal.h"
#include "kv_cache.h"
#include "attention.h"
#include "deltanet.h"
#include "qmatvec.h"
#include "quant_types.h"
#include "qwen_ops.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// Thread-local error buffer for qwen_model errors
static thread_local char g_qwen_error[1024] = {0};

static void qwen_set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_qwen_error, sizeof(g_qwen_error), fmt, ap);
    va_end(ap);
    fprintf(stderr, "[qwen] %s\n", g_qwen_error);
}

static bool qwen_debug() {
    static const int on = getenv("QWEN_DEBUG") ? 1 : 0;
    return on;
}

// One /dev/shm file, mmap(MAP_SHARED), hipHostRegister'ed once (project core pattern), carved into
// 256-byte-aligned float arrays.
//
// Registration mode matters on the UMA APU, where both modes are zero-copy views of the same DRAM:
// hipHostRegisterMapped is fine-grained (coherent at every access, so the GPU does not cache it), while
// hipExtHostRegisterCoarseGrained lets the GPU cache it and makes it coherent at kernel boundaries and
// syncs. Activations take coarse-grained: the GPU re-reads x once per output row, and every CPU/GPU
// handoff already goes through a launch or hipDeviceSynchronize. That made the K = 9216 ffn_down matvec
// 5x faster on gfx1151.
struct ShmRegion {
    void* base = nullptr;
    size_t total = 0;

    bool alloc(const char* path, const size_t* sizes, float** const* ptrs, size_t n,
               unsigned flags = hipHostRegisterMapped) {
        std::vector<size_t> off(n);
        total = 0;
        for (size_t i = 0; i < n; ++i) {
            off[i] = total;
            total += (sizes[i] * sizeof(float) + 255) & ~(size_t)255;
        }
        int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) return false;
        const bool sized = ftruncate(fd, (off_t)total) == 0;
        void* p = sized ? mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
        close(fd);
        unlink(path);
        if (p == MAP_FAILED) return false;
        if (hipHostRegister(p, total, flags) != hipSuccess) {
            munmap(p, total);
            return false;
        }
        base = p;
        for (size_t i = 0; i < n; ++i) *ptrs[i] = (float*)((char*)p + off[i]);
        return true;
    }

    void free() {
        if (!base) return;
        (void)hipHostUnregister(base);
        munmap(base, total);
        base = nullptr;
    }
};

// Per-token activations
struct Scratch {
    float* x = nullptr;       // residual stream [E]
    float* xn = nullptr;      // normed input to the projections [E]
    float* qfull = nullptr;   // attn_q output, [Q | gate] per head [2 * HQ * HD]
    float* q = nullptr;       // Q after norm + RoPE [HQ * HD]
    float* k = nullptr;       // [HKV * HD]
    float* v = nullptr;       // [HKV * HD]
    float* attn = nullptr;    // gated attention output, input to attn_output [HQ * HD]
    float* proj = nullptr;    // attn_output / ssm_out / ffn_down output [E]
    float* ffg = nullptr;     // ffn_gate output [FF]
    float* ffu = nullptr;     // ffn_up output [FF]
    float* act = nullptr;     // SiLU(gate) * up, input to ffn_down [FF]
    float* logits = nullptr;  // [vocab]
    // Gated DeltaNet
    float* qkv = nullptr;     // attn_qkv output [conv_dim]
    float* conv = nullptr;    // SiLU(conv1d(qkv)) = [q | k | v] [conv_dim]
    float* z = nullptr;       // attn_gate output [d_inner]
    float* alpha = nullptr;   // ssm_alpha output, then the decay gate [dt_rank]
    float* beta = nullptr;    // ssm_beta output, then sigmoid [dt_rank]
    float* lin = nullptr;     // delta-rule output, then gated-normed: input to ssm_out [d_inner]
    // Attention partial results, per (head, 256-position chunk) up to n_ctx (attention.h)
    float* part_ml = nullptr;
    float* part_acc = nullptr;

    ShmRegion mem;

    bool alloc(int E, int HQ, int HKV, int HD, int FF, int V, int conv_dim, int DI, int NV, int n_ctx) {
        const size_t sizes[] = {(size_t)E, (size_t)E, (size_t)2 * HQ * HD, (size_t)HQ * HD,
                                (size_t)HKV * HD, (size_t)HKV * HD, (size_t)HQ * HD, (size_t)E,
                                (size_t)FF, (size_t)FF, (size_t)FF, (size_t)V,
                                (size_t)conv_dim, (size_t)conv_dim, (size_t)DI, (size_t)NV, (size_t)NV, (size_t)DI,
                                attention_part_ml_floats(HQ, n_ctx), attention_part_acc_floats(HQ, n_ctx)};
        float** const ptrs[] = {&x, &xn, &qfull, &q, &k, &v, &attn, &proj, &ffg, &ffu, &act, &logits,
                                &qkv, &conv, &z, &alpha, &beta, &lin, &part_ml, &part_acc};
        static_assert(sizeof(sizes) / sizeof(sizes[0]) == sizeof(ptrs) / sizeof(ptrs[0]), "scratch table");
        return mem.alloc("/dev/shm/qwen_scratch.bin", sizes, ptrs, sizeof(sizes) / sizeof(sizes[0]),
                         hipExtHostRegisterCoarseGrained);
    }
};

// Recurrent state of the Gated DeltaNet layers, persisted across tokens like the KV cache: per linear
// layer a conv state [conv_dim][d_conv - 1] and a delta-rule state [dt_rank][d_state][d_state] (row-major
// S per head, the layout of deltanet.hip). One sequence; zeroed when a sequence starts at pos 0. Only the
// GPU touches it (the kernels, and hipMemsetAsync for the reset), so it is registered coarse-grained.
struct RecurrentState {
    std::vector<float*> conv, ssm;   // indexed by layer; nullptr for full-attention layers
    ShmRegion mem;

    bool alloc(const inference_model* eng) {
        const int conv_dim = 2 * eng->ssm_n_group * eng->ssm_d_state + eng->ssm_d_inner;
        conv.assign(eng->n_layer, nullptr);
        ssm.assign(eng->n_layer, nullptr);
        std::vector<size_t> sizes;
        std::vector<float**> ptrs;
        for (int l = 0; l < eng->n_layer; ++l) {
            if (eng->layers[l].kind != LAYER_LINEAR_ATTN) continue;
            sizes.push_back((size_t)conv_dim * (eng->ssm_d_conv - 1));
            ptrs.push_back(&conv[l]);
            sizes.push_back((size_t)eng->ssm_dt_rank * eng->ssm_d_state * eng->ssm_d_state);
            ptrs.push_back(&ssm[l]);
        }
        if (sizes.empty()) return true;
        return mem.alloc("/dev/shm/qwen_recurrent_state.bin", sizes.data(), ptrs.data(), sizes.size(),
                         hipExtHostRegisterCoarseGrained);
    }

    hipError_t reset() {
        return mem.base ? hipMemsetAsync(mem.base, 0, mem.total, 0) : hipSuccess;
    }
};

// Wall-time breakdown of the forward pass (QWEN_PROFILE=1 prints it; see qwen_model_profile_print). Sections
// are timed by laps: lap(c) charges the time since the previous lap to c. A GPU section runs from its first
// qmatvec launch to the hipDeviceSynchronize after it, so it includes launch overhead and the sync wait.
enum ProfCat {
    P_EMBED, P_ATTN_CPU, P_ATTN_GPU, P_LIN_CPU, P_LIN_GPU, P_FFN_CPU, P_FFN_GPU, P_HEAD_CPU,
    P_HEAD_GPU, P_COUNT
};
static const char* const PROF_NAME[P_COUNT] = {
    "embedding (CPU)", "attention: norm, residual (CPU)",
    "attention: matvecs + norm/RoPE/attention kernels (GPU)", "DeltaNet: norm, residual (CPU)",
    "DeltaNet: matvecs + conv/delta-rule kernels (GPU)", "FFN: norm, SwiGLU (CPU)",
    "FFN: gate/up/down matvecs (GPU)", "output norm + logits out (CPU)", "LM head matvec (GPU)",
};
static const bool PROF_GPU[P_COUNT] = {false, false, true, false, true, false, true, false, true};

struct Profile {
    double t[P_COUNT] = {};
    double bytes[P_COUNT] = {};   // weight bytes streamed by the GPU sections
    double last = 0.0;
    int tokens = 0;

    static double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void start() { last = now(); }
    void lap(ProfCat c) {
        const double t1 = now();
        t[c] += t1 - last;
        last = t1;
    }
};

struct QwenModel {
    inference_model* eng = nullptr;
    Profile prof;
    Scratch s;
    RecurrentState rs;
    qwen_layer_cb layer_cb = nullptr;
    void* layer_cb_user = nullptr;

    ~QwenModel() {
        rs.mem.free();
        s.mem.free();
        attention_free();
        deltanet_free();
        qmatvec_free();
    }

    bool init(inference_model* model) {
        eng = model;
        if (qmatvec_init("matvec_q.co") != 0) {
            qwen_set_error("cannot load matvec_q.co (run from build/co)");
            return false;
        }
        bool has_linear = false;
        for (const LayerWeights& L : eng->layers) has_linear |= L.kind == LAYER_LINEAR_ATTN;
        if (has_linear) {
            if (eng->ssm_d_state != DELTANET_HEAD_SIZE) {
                qwen_set_error("ssm.state_size %d: the DeltaNet kernels are built for %d", eng->ssm_d_state,
                               DELTANET_HEAD_SIZE);
                return false;
            }
            if (deltanet_init("deltanet.co") != 0) {
                qwen_set_error("cannot load deltanet.co (run from build/co)");
                return false;
            }
        }
        const int conv_dim = 2 * eng->ssm_n_group * eng->ssm_d_state + eng->ssm_d_inner;
        if (eng->n_kv_layers > 0) {
            if (eng->head_dim != ATTN_HEAD_DIM) {
                qwen_set_error("head_dim %d: the attention kernels are built for %d", eng->head_dim, ATTN_HEAD_DIM);
                return false;
            }
            if (attention_init("attention.co") != 0) {
                qwen_set_error("cannot load attention.co (run from build/co)");
                return false;
            }
        }
        if (!s.alloc(eng->n_embd, eng->n_head, eng->n_head_kv, eng->head_dim, eng->n_ff, eng->vocab_size, conv_dim,
                     eng->ssm_d_inner, eng->ssm_dt_rank, kv_cache_n_ctx(eng->kvc))) {
            qwen_set_error("scratch allocation failed");
            return false;
        }
        if (!rs.alloc(eng)) {
            qwen_set_error("recurrent state allocation failed");
            return false;
        }
        return true;
    }

    // y = W x on the GPU. x and y must be in pinned memory (the scratch buffer). c: profile section
    bool matvec(const QuantWeight& W, const float* x, float* y, ProfCat c) {
        prof.bytes[c] += (double)W.row_bytes * W.N;
        hipError_t e = qmatvec_launch(W.type, W.data, W.row_bytes, x, y, W.N, W.K, 0);
        if (e != hipSuccess) {
            qwen_set_error("qmatvec type %d %dx%d: %s", W.type, W.N, W.K, hipGetErrorString(e));
            return false;
        }
        return true;
    }

    bool sync() {
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) { qwen_set_error("hipDeviceSynchronize: %s", hipGetErrorString(e)); return false; }
        return true;
    }

    // Gated full attention for one token at position pos; adds its output to the residual. q/k norm, RoPE,
    // the KV cache write, attention over positions 0..pos and the output gate run as GPU kernels
    // (attention.hip), queued between the q/k/v projections and attn_output: one sync.
    bool full_attention(const LayerWeights& L, int pos) {
        const int E = eng->n_embd, HQ = eng->n_head, HKV = eng->n_head_kv;
        const float eps = (float)eng->eps;

        qwen_rmsnorm(s.x, L.attn_norm, s.xn, E, eps);
        prof.lap(P_ATTN_CPU);
        if (!matvec(L.Wq, s.xn, s.qfull, P_ATTN_GPU) || !matvec(L.Wk, s.xn, s.k, P_ATTN_GPU) ||
            !matvec(L.Wv, s.xn, s.v, P_ATTN_GPU))
            return false;
        __half* K = kv_cache_k_row(eng->kvc, L.kv_slot, 0);
        __half* V = kv_cache_v_row(eng->kvc, L.kv_slot, 0);
        if (!K || !V) { qwen_set_error("no KV cache for KV layer %d", L.kv_slot); return false; }
        const hipError_t e = attention_forward(s.qfull, s.k, s.v, L.q_norm, L.k_norm, K, V, HQ, HKV, eng->n_rot, pos,
                                               eng->rope_freq_base, eps, s.q, s.part_ml, s.part_acc, s.attn, 0);
        if (e != hipSuccess) { qwen_set_error("attention kernels, KV layer %d: %s", L.kv_slot, hipGetErrorString(e)); return false; }
        if (!matvec(L.Wo, s.attn, s.proj, P_ATTN_GPU) || !sync()) return false;
        prof.lap(P_ATTN_GPU);
        for (int i = 0; i < E; ++i) s.x[i] += s.proj[i];
        prof.lap(P_ATTN_CPU);
        return true;
    }

    // Gated DeltaNet token mixer for one token; updates the layer's conv and delta-rule state and adds its
    // output to the residual. Follows llama.cpp's qwen35 build_layer_attn_linear.
    bool linear_attention(int l, const LayerWeights& L) {
        const int E = eng->n_embd, NG = eng->ssm_n_group, NV = eng->ssm_dt_rank;
        const float eps = (float)eng->eps;

        qwen_rmsnorm(s.x, L.attn_norm, s.xn, E, eps);
        prof.lap(P_LIN_CPU);
        if (!matvec(L.W_qkv, s.xn, s.qkv, P_LIN_GPU) || !matvec(L.W_z, s.xn, s.z, P_LIN_GPU) ||
            !matvec(L.W_alpha, s.xn, s.alpha, P_LIN_GPU) || !matvec(L.W_beta, s.xn, s.beta, P_LIN_GPU))
            return false;

        // conv1d + SiLU + q/k L2 norm + gates, then the delta rule + gated norm, then ssm_out: all queued on
        // the GPU behind the projections, one sync (deltanet.hip). alpha/beta become g/sigmoid(beta) in place.
        hipError_t e = deltanet_pre(s.qkv, rs.conv[l], L.ssm_conv1d, eng->ssm_d_conv, s.conv, NG, NV, s.alpha, s.beta,
                                    L.ssm_dt, L.ssm_a, s.alpha, s.beta, eps, 0);
        if (e == hipSuccess) e = deltanet_rule(rs.ssm[l], s.conv, s.alpha, s.beta, s.z, L.ssm_norm, s.lin, NG, NV, eps, 0);
        if (e != hipSuccess) { qwen_set_error("deltanet kernels, layer %d: %s", l, hipGetErrorString(e)); return false; }
        if (!matvec(L.W_ssm_out, s.lin, s.proj, P_LIN_GPU) || !sync()) return false;
        prof.lap(P_LIN_GPU);
        for (int i = 0; i < E; ++i) s.x[i] += s.proj[i];
        prof.lap(P_LIN_CPU);
        return true;
    }

    // SwiGLU FFN; adds its output to the residual
    bool ffn(const LayerWeights& L) {
        const int E = eng->n_embd, FF = eng->n_ff;
        qwen_rmsnorm(s.x, L.ffn_norm, s.xn, E, (float)eng->eps);
        prof.lap(P_FFN_CPU);
        if (!matvec(L.W_gate, s.xn, s.ffg, P_FFN_GPU) || !matvec(L.W_up, s.xn, s.ffu, P_FFN_GPU) || !sync())
            return false;
        prof.lap(P_FFN_GPU);
        for (int i = 0; i < FF; ++i) s.act[i] = qwen_silu(s.ffg[i]) * s.ffu[i];
        prof.lap(P_FFN_CPU);
        if (!matvec(L.W_down, s.act, s.proj, P_FFN_GPU) || !sync()) return false;
        prof.lap(P_FFN_GPU);
        for (int i = 0; i < E; ++i) s.x[i] += s.proj[i];
        prof.lap(P_FFN_CPU);
        return true;
    }

    int forward(int token_id, int pos, float* logits_out) {
        const int E = eng->n_embd;
        if (token_id < 0 || token_id >= eng->vocab_size) { qwen_set_error("token %d out of range", token_id); return -1; }
        if (!eng->kvc || pos < 0 || pos >= kv_cache_n_ctx(eng->kvc)) { qwen_set_error("position %d out of range", pos); return -1; }

        prof.start();
        // Embedding: decode one row of token_embd on the CPU
        if (qt_to_float(eng->token_embd.type, eng->token_embd.row(token_id), s.x, E) != 0) {
            qwen_set_error("cannot decode token_embd type %d", eng->token_embd.type);
            return -1;
        }

        // A sequence starting at position 0 starts from zero recurrent state (the KV cache needs no reset:
        // attention only reads positions 0..pos)
        if (pos == 0 && rs.reset() != hipSuccess) { qwen_set_error("recurrent state reset failed"); return -1; }
        prof.lap(P_EMBED);

        for (int l = 0; l < eng->n_layer; ++l) {
            prof.start();  // the debug trace and layer callback of the previous layer are not charged
            const LayerWeights& L = eng->layers[l];
            const bool ok = L.kind == LAYER_FULL_ATTN ? full_attention(L, pos) : linear_attention(l, L);
            if (!ok || !ffn(L)) return -1;
            if (layer_cb) layer_cb(l, pos, s.x, E, layer_cb_user);
            if (qwen_debug()) {
                double ss = 0.0;
                for (int i = 0; i < E; ++i) ss += (double)s.x[i] * s.x[i];
                fprintf(stderr, "[qwen] pos %d blk %2d %-6s rms(x)=%.5g x[0..3]=%.5g %.5g %.5g %.5g\n", pos, l,
                        L.kind == LAYER_FULL_ATTN ? "attn" : "linear", sqrt(ss / E), s.x[0], s.x[1], s.x[2], s.x[3]);
            }
        }

        prof.start();  // the debug trace and layer callback above are not charged
        qwen_rmsnorm(s.x, eng->output_norm, s.xn, E, (float)eng->eps);
        prof.lap(P_HEAD_CPU);
        if (!matvec(eng->output, s.xn, s.logits, P_HEAD_GPU) || !sync()) return -1;
        prof.lap(P_HEAD_GPU);
        if (logits_out) memcpy(logits_out, s.logits, (size_t)eng->vocab_size * sizeof(float));
        prof.lap(P_HEAD_CPU);
        prof.tokens++;
        return 0;
    }
};

extern "C" {

qwen_model_t* qwen_model_create(inference_model_t* engine_model) {
    QwenModel* m = new QwenModel();
    if (!m->init(engine_model)) { delete m; return nullptr; }
    return m;
}

void qwen_model_free(qwen_model_t* model) {
    delete model;
}

const float* qwen_model_logits(qwen_model_t* model) {
    return model ? model->s.logits : nullptr;
}

void qwen_model_profile_reset(qwen_model_t* model) {
    if (model) model->prof = Profile();
}

void qwen_model_profile_print(qwen_model_t* model, const char* title) {
    if (!model) return;
    const Profile& p = model->prof;
    const int n = p.tokens > 0 ? p.tokens : 1;
    double total = 0.0, gpu = 0.0, bytes = 0.0;
    for (int c = 0; c < P_COUNT; ++c) {
        total += p.t[c];
        if (PROF_GPU[c]) { gpu += p.t[c]; bytes += p.bytes[c]; }
    }
    fprintf(stderr, "[profile] %s: %d tokens, %.1f ms/token (%.2f tok/s); GPU sections %.0f%%, CPU %.0f%%\n",
            title ? title : "forward", p.tokens, 1e3 * total / n, total > 0 ? n / total : 0.0,
            total > 0 ? 100 * gpu / total : 0.0, total > 0 ? 100 * (total - gpu) / total : 0.0);
    for (int c = 0; c < P_COUNT; ++c) {
        if (p.t[c] == 0.0) continue;
        fprintf(stderr, "[profile]   %-50s %8.2f ms/token %5.1f%%", PROF_NAME[c], 1e3 * p.t[c] / n,
                total > 0 ? 100 * p.t[c] / total : 0.0);
        if (PROF_GPU[c] && p.t[c] > 0)
            fprintf(stderr, "  %7.1f MB/token %6.1f GB/s", p.bytes[c] / n / 1e6, p.bytes[c] / p.t[c] / 1e9);
        fprintf(stderr, "\n");
    }
    if (gpu > 0)
        fprintf(stderr, "[profile]   weights streamed: %.0f MB/token, %.1f GB/s averaged over the GPU sections\n",
                bytes / n / 1e6, bytes / gpu / 1e9);
}

void qwen_model_set_layer_callback(qwen_model_t* model, qwen_layer_cb cb, void* user) {
    if (!model) return;
    model->layer_cb = cb;
    model->layer_cb_user = user;
}

int qwen_model_forward(qwen_model_t* model, int token_id, int pos, float* logits) {
    if (!model) return -1;
    return model->forward(token_id, pos, logits);
}

// The old batched path assumed dense full-attention layers and the repacked FP4 weights. It is removed
// until the single-token path matches the reference (Phase 20); batching comes back on top of qmatvec.
int qwen_model_forward_batch(qwen_model_t* model, const int32_t* tokens, const int* positions,
                             const int* seq_lens, int batch_size, int seq_len, float* logits_out) {
    (void)model; (void)tokens; (void)positions; (void)seq_lens; (void)batch_size; (void)seq_len; (void)logits_out;
    qwen_set_error("batched forward is not implemented for the qmatvec path yet");
    return INFERENCE_ERR_NOT_IMPLEMENTED;
}

}
