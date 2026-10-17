// qwen_model.cpp -- Qwen3.5 single-token forward pass (Phase 20).
//
// Every projection (attention Q/K/V/O, DeltaNet, FFN gate/up/down, LM head) is one qmatvec_launch on the raw
// GGUF blocks, read in place from the pinned file mapping. Activations live in one /dev/shm scratch buffer,
// pinned once, so the GPU reads and writes them without copies. Everything after the embedding row decode
// runs on the GPU: projections, the token mixers (attention.hip, deltanet.hip), norms, residual adds and
// SwiGLU (gpu_ops.hip), queued back to back with one sync per token. qwen_ops holds the CPU reference
// versions of these ops, which tests/forward_verify checks the kernels against.
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
#include "gpu_ops.h"
#include "qmatvec.h"
#include "quant_types.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <algorithm>
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
    float* x_alt = nullptr;   // the other residual buffer: add_rmsnorm writes x + add here, then they swap
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
    // Quantized copy of the current matvec input for the int8 kernels (qmatvec.h), up to q8_max_k values
    float* q8mem = nullptr;
    qmatvec_q8 q8;
    int q8_max_k = 0;

    ShmRegion mem;

    bool alloc(int E, int HQ, int HKV, int HD, int FF, int V, int conv_dim, int DI, int NV, int n_ctx) {
        q8_max_k = (std::max({E, HQ * HD, DI, FF}) + 255) / 256 * 256;
        const size_t sizes[] = {(size_t)E, (size_t)E, (size_t)E, (size_t)2 * HQ * HD, (size_t)HQ * HD,
                                (size_t)HKV * HD, (size_t)HKV * HD, (size_t)HQ * HD, (size_t)E,
                                (size_t)FF, (size_t)FF, (size_t)FF, (size_t)V,
                                (size_t)conv_dim, (size_t)conv_dim, (size_t)DI, (size_t)NV, (size_t)NV, (size_t)DI,
                                attention_part_ml_floats(HQ, n_ctx), attention_part_acc_floats(HQ, n_ctx),
                                (qmatvec_q8_bytes(q8_max_k) + 3) / 4};
        float** const ptrs[] = {&x, &x_alt, &xn, &qfull, &q, &k, &v, &attn, &proj, &ffg, &ffu, &act, &logits,
                                &qkv, &conv, &z, &alpha, &beta, &lin, &part_ml, &part_acc, &q8mem};
        static_assert(sizeof(sizes) / sizeof(sizes[0]) == sizeof(ptrs) / sizeof(ptrs[0]), "scratch table");
        if (!mem.alloc("/dev/shm/qwen_scratch.bin", sizes, ptrs, sizeof(sizes) / sizeof(sizes[0]),
                       hipExtHostRegisterCoarseGrained))
            return false;
        qmatvec_q8_bind(&q8, q8mem, q8_max_k);
        return true;
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

// Wall-time breakdown of the forward pass (QWEN_PROFILE=1; see qwen_model_profile_print). A token is one GPU
// queue with a single sync, so section times only exist when profiling: then each section ends with a
// hipDeviceSynchronize and lap(c) charges the time since the previous lap to c. Those extra syncs make a
// profiled token a little slower than an unprofiled one.
enum ProfCat { P_EMBED, P_ATTN, P_LIN, P_FFN, P_HEAD, P_COUNT };
static const char* const PROF_NAME[P_COUNT] = {
    "embedding row decode (CPU)",
    "attention: matvecs + norm/RoPE/attention kernels",
    "DeltaNet: matvecs + conv/delta-rule kernels",
    "FFN: residual + norms, gate/up/down matvecs, SwiGLU",
    "LM head matvec + logits out",
};
static const bool PROF_GPU[P_COUNT] = {false, true, true, true, true};

struct Profile {
    bool on = getenv("QWEN_PROFILE") != nullptr;
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
    // int8 matvec path: s.q8 holds the quantized form of q8_src[q8_k]; matvec() uses it when its input matches.
    // QWEN_MATVEC_FP32=1 keeps every matvec on the float kernels (A/B and debugging).
    bool use_q8 = getenv("QWEN_MATVEC_FP32") == nullptr;
    const float* q8_src = nullptr;
    int q8_k = 0;

    ~QwenModel() {
        rs.mem.free();
        s.mem.free();
        gpu_ops_free();
        attention_free();
        deltanet_free();
        qmatvec_free();
    }

    bool init(inference_model* model) {
        eng = model;
        if (qmatvec_init("matvec_q.co") != 0 || gpu_ops_init("gpu_ops.co") != 0) {
            qwen_set_error("cannot load matvec_q.co / gpu_ops.co (run from build/co)");
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

    // The int8 buffer for a producer kernel to quantize its K-value output into, or NULL (float path)
    const qmatvec_q8* q8_for(int K) const { return use_q8 && K % 256 == 0 && K <= s.q8_max_k ? &s.q8 : nullptr; }

    // A producer kernel just wrote x[K], quantized into q (NULL: not quantized); the next matvecs read it
    void mark(const float* x, int K, const qmatvec_q8* q) {
        q8_src = q ? x : nullptr;
        q8_k = K;
    }

    // x[K] was written by a kernel that cannot quantize its output: quantize it in a separate launch
    bool produced(const float* x, int K) {
        const qmatvec_q8* q = q8_for(K);
        if (q && !check(qmatvec_quantize(x, K, q, 0), "quantize_q8")) return false;
        mark(x, K, q);
        return true;
    }

    // y = W x on the GPU. x and y must be in pinned memory (the scratch buffer). c: profile section. Uses the
    // int8 kernel when there is one and s.q8 holds this x; otherwise the float kernel.
    bool matvec(const QuantWeight& W, const float* x, float* y, ProfCat c) {
        prof.bytes[c] += (double)W.row_bytes * W.N;
        const bool q8 = q8_src == x && q8_k == W.K && qmatvec_q8_supported(W.type, W.K);
        hipError_t e = q8 ? qmatvec_launch_q8(W.type, W.data, W.row_bytes, &s.q8, y, W.N, W.K, 0)
                          : qmatvec_launch(W.type, W.data, W.row_bytes, x, y, W.N, W.K, 0);
        if (e != hipSuccess) {
            qwen_set_error("qmatvec type %d %dx%d: %s", W.type, W.N, W.K, hipGetErrorString(e));
            return false;
        }
        return true;
    }

    bool check(hipError_t e, const char* what) {
        if (e != hipSuccess) qwen_set_error("%s: %s", what, hipGetErrorString(e));
        return e == hipSuccess;
    }

    bool sync() { return check(hipDeviceSynchronize(), "hipDeviceSynchronize"); }

    // End of a profiled section: only when profiling, sync so the GPU time lands in this section
    bool section(ProfCat c) {
        if (prof.on && !sync()) return false;
        prof.lap(c);
        return true;
    }

    // Gated full attention for one token at position pos, from the normed input s.xn; output in s.proj.
    // q/k norm, RoPE, the KV cache write, attention over positions 0..pos and the output gate are GPU
    // kernels (attention.hip).
    bool full_attention(const LayerWeights& L, int pos) {
        const int HQ = eng->n_head, HKV = eng->n_head_kv;
        if (!matvec(L.Wq, s.xn, s.qfull, P_ATTN) || !matvec(L.Wk, s.xn, s.k, P_ATTN) || !matvec(L.Wv, s.xn, s.v, P_ATTN))
            return false;
        __half* K = kv_cache_k_row(eng->kvc, L.kv_slot, 0);
        __half* V = kv_cache_v_row(eng->kvc, L.kv_slot, 0);
        if (!K || !V) { qwen_set_error("no KV cache for KV layer %d", L.kv_slot); return false; }
        const qmatvec_q8* q8 = q8_for(HQ * eng->head_dim);
        if (!check(attention_forward(s.qfull, s.k, s.v, L.q_norm, L.k_norm, K, V, HQ, HKV, eng->n_rot, pos,
                                     eng->rope_freq_base, (float)eng->eps, s.q, s.part_ml, s.part_acc, s.attn, q8, 0),
                   "attention kernels"))
            return false;
        mark(s.attn, HQ * eng->head_dim, q8);
        return matvec(L.Wo, s.attn, s.proj, P_ATTN);
    }

    // Gated DeltaNet token mixer for one token, from s.xn; output in s.proj. Updates the layer's conv and
    // delta-rule state. Follows llama.cpp's qwen35 build_layer_attn_linear (deltanet.hip).
    bool linear_attention(int l, const LayerWeights& L) {
        const int NG = eng->ssm_n_group, NV = eng->ssm_dt_rank;
        const float eps = (float)eng->eps;
        if (!matvec(L.W_qkv, s.xn, s.qkv, P_LIN) || !matvec(L.W_z, s.xn, s.z, P_LIN)) return false;
        // alpha and beta: two N = 32 matvecs on the same input, one launch on the int8 path
        const QuantWeight &A = L.W_alpha, &B = L.W_beta;
        if (q8_src == s.xn && q8_k == A.K && A.type == B.type && A.K == B.K && A.row_bytes == B.row_bytes &&
            qmatvec_q8_supported(A.type, A.K)) {
            prof.bytes[P_LIN] += (double)A.row_bytes * (A.N + B.N);
            if (!check(qmatvec_launch_q8_pair(A.type, A.data, A.N, s.alpha, B.data, B.N, s.beta, A.row_bytes, &s.q8,
                                              A.K, 0), "alpha/beta matvec"))
                return false;
        } else if (!matvec(A, s.xn, s.alpha, P_LIN) || !matvec(B, s.xn, s.beta, P_LIN)) {
            return false;
        }
        // conv1d + SiLU + q/k L2 norm + gates, then the delta rule + gated norm; alpha/beta become
        // g / sigmoid(beta) in place
        if (!check(deltanet_pre(s.qkv, rs.conv[l], L.ssm_conv1d, eng->ssm_d_conv, s.conv, NG, NV, s.alpha, s.beta,
                                L.ssm_dt, L.ssm_a, s.alpha, s.beta, eps, 0), "deltanet_pre") ||
            !check(deltanet_rule(rs.ssm[l], s.conv, s.alpha, s.beta, s.z, L.ssm_norm, s.lin, NG, NV, eps, 0),
                   "deltanet_rule") ||
            !produced(s.lin, eng->ssm_d_inner))
            return false;
        return matvec(L.W_ssm_out, s.lin, s.proj, P_LIN);
    }

    // Residual add of the mixer output, the FFN's RMSNorm, SwiGLU FFN, then the residual add of its output
    // fused with the next RMSNorm (next layer's attn_norm, or output_norm after the last layer)
    bool ffn(const LayerWeights& L, const float* next_norm) {
        const int E = eng->n_embd, FF = eng->n_ff;
        const float eps = (float)eng->eps;
        const qmatvec_q8 *qe = q8_for(E), *qf = q8_for(FF);
        if (!check(gpu_add_rmsnorm(s.x, s.proj, s.x_alt, L.ffn_norm, s.xn, E, eps, qe, 0), "add_rmsnorm")) return false;
        std::swap(s.x, s.x_alt);
        mark(s.xn, E, qe);
        if (!matvec(L.W_gate, s.xn, s.ffg, P_FFN) || !matvec(L.W_up, s.xn, s.ffu, P_FFN) ||
            !check(gpu_swiglu(s.ffg, s.ffu, s.act, FF, qf, 0), "swiglu"))
            return false;
        mark(s.act, FF, qf);
        if (!matvec(L.W_down, s.act, s.proj, P_FFN) ||
            !check(gpu_add_rmsnorm(s.x, s.proj, s.x_alt, next_norm, s.xn, E, eps, qe, 0), "add_rmsnorm"))
            return false;
        std::swap(s.x, s.x_alt);
        mark(s.xn, E, qe);
        return true;
    }

    // One token: the CPU decodes the embedding row into the residual stream, then the whole forward pass is
    // one GPU queue with a single sync before the logits are read (more syncs when profiling or when a layer
    // callback / QWEN_DEBUG wants the residual after each layer).
    int forward(int token_id, int pos, float* logits_out) {
        const int E = eng->n_embd;
        const float eps = (float)eng->eps;
        if (token_id < 0 || token_id >= eng->vocab_size) { qwen_set_error("token %d out of range", token_id); return -1; }
        if (!eng->kvc || pos < 0 || pos >= kv_cache_n_ctx(eng->kvc)) { qwen_set_error("position %d out of range", pos); return -1; }

        prof.start();
        if (qt_to_float(eng->token_embd.type, eng->token_embd.row(token_id), s.x, E) != 0) {
            qwen_set_error("cannot decode token_embd type %d", eng->token_embd.type);
            return -1;
        }
        // A sequence starting at position 0 starts from zero recurrent state (the KV cache needs no reset:
        // attention only reads positions 0..pos)
        if (pos == 0 && !check(rs.reset(), "recurrent state reset")) return -1;
        if (!check(gpu_add_rmsnorm(s.x, nullptr, s.x, eng->layers[0].attn_norm, s.xn, E, eps, q8_for(E), 0),
                   "add_rmsnorm"))
            return -1;
        mark(s.xn, E, q8_for(E));
        prof.lap(P_EMBED);

        const bool per_layer = layer_cb || qwen_debug();
        for (int l = 0; l < eng->n_layer; ++l) {
            const LayerWeights& L = eng->layers[l];
            const bool attn = L.kind == LAYER_FULL_ATTN;
            if (!(attn ? full_attention(L, pos) : linear_attention(l, L)) || !section(attn ? P_ATTN : P_LIN)) return -1;
            const float* next_norm = l + 1 < eng->n_layer ? eng->layers[l + 1].attn_norm : eng->output_norm;
            if (!ffn(L, next_norm) || !section(P_FFN)) return -1;
            if (per_layer) {
                if (!sync()) return -1;
                if (layer_cb) layer_cb(l, pos, s.x, E, layer_cb_user);
                if (qwen_debug()) {
                    double ss = 0.0;
                    for (int i = 0; i < E; ++i) ss += (double)s.x[i] * s.x[i];
                    fprintf(stderr, "[qwen] pos %d blk %2d %-6s rms(x)=%.5g x[0..3]=%.5g %.5g %.5g %.5g\n", pos, l,
                            attn ? "attn" : "linear", sqrt(ss / E), s.x[0], s.x[1], s.x[2], s.x[3]);
                }
                prof.start();  // the callback and debug trace are not charged
            }
        }

        if (!matvec(eng->output, s.xn, s.logits, P_HEAD) || !sync()) return -1;
        if (logits_out) memcpy(logits_out, s.logits, (size_t)eng->vocab_size * sizeof(float));
        prof.lap(P_HEAD);
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
