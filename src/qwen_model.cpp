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

#include "inference_log.h"
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
#include <initializer_list>
#include <unordered_map>
#include <vector>

// Thread-local error buffer for qwen_model errors
static thread_local char g_qwen_error[1024] = {0};

static void qwen_set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_qwen_error, sizeof(g_qwen_error), fmt, ap);
    va_end(ap);
    INFER_LOG("[qwen] %s\n", g_qwen_error);
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
// Per-forward activations for up to B tokens (one micro-batch of a prefill; B = 1 row is used when
// generating). Each array holds B rows of the per-token size given in brackets.
struct Scratch {
    int B = 0;                // tokens per forward (rows)
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
    float* logits = nullptr;  // [vocab], one row: only the last token's logits are computed
    // Gated DeltaNet
    float* qkv = nullptr;     // attn_qkv output [conv_dim]
    float* conv = nullptr;    // SiLU(conv1d(qkv)) = [q | k | v] [conv_dim]
    float* z = nullptr;       // attn_gate output [d_inner]
    float* alpha = nullptr;   // ssm_alpha output, then the decay gate [dt_rank]
    float* beta = nullptr;    // ssm_beta output, then sigmoid [dt_rank]
    float* lin = nullptr;     // delta-rule output, then gated-normed: input to ssm_out [d_inner]
    // Mixture of experts (the shared expert uses ffg / ffu / act / proj like a dense FFN)
    float* r_logits = nullptr;  // router logits [n_expert]
    int* ids = nullptr;         // selected experts [n_used]
    float* wts = nullptr;       // their renormalized weights [n_used]
    float* sg = nullptr;        // shared-expert gate [1]
    float* moe_g = nullptr;     // expert gate projections [n_used * n_ff_exp]
    float* moe_u = nullptr;     // expert up projections [n_used * n_ff_exp]
    float* moe_act = nullptr;   // SiLU(gate) * up per expert [n_used * n_ff_exp]
    float* moe_down = nullptr;  // expert down projections [n_used * n_embd]
    float* moe_gu = nullptr;    // Gemma-4: fused expert gate|up projections [n_used * 2 n_ff_exp]
    float* sh_out = nullptr;    // shared expert output [n_embd]
    // MoE prefill grouping (gpu_moe_group): entries sorted by expert, and tiles of up to g_tile of them
    int g_tile = 0;
    int *g_tok = nullptr, *g_pair = nullptr, *g_ntiles = nullptr, *g_tile_e = nullptr, *g_tile_start = nullptr,
        *g_tile_n = nullptr;
    // Attention partial results, per (token, head, 256-position chunk) up to n_ctx (attention.h)
    float* part_ml = nullptr;
    float* part_acc = nullptr;
    // Hyper-connections (qwen4exp; n_hc streams of n_embd, HCD = n_hc * n_embd per token; empty otherwise)
    float* res = nullptr;     // the residual streams [HCD]
    float* res_alt = nullptr; // the other residual buffer: hc_norm's combine writes res + update here, then they swap
    float* hxn = nullptr;     // a mixer's normed streams [HCD], input to its down / inject projections
    float* hlo = nullptr;     // down projection [hc_rank]
    float* hlo_pad = nullptr; // SiLU(hlo / n_hc), zero-padded to hc_rank_pad: the up projection's input
    float* hgate = nullptr;   // up projection: the streams' gate logits [HCD]
    float* inj = nullptr;     // inject projection: the stream weights' logits [n_hc]
    // PLE (qwen4exp)
    float* ple_emb = nullptr;     // the gathered n-gram hash rows [n_embd]
    float* ple_key = nullptr;     // ple_key projection [HCD]
    float* ple_val = nullptr;     // ple_value projection [n_embd]
    float* ple_gated = nullptr;   // value * per-stream gate [HCD]
    float* ple_normed = nullptr;  // its grouped RMSNorm: the conv input [HCD]
    // Quantized copy of the current matmul input for the int8 kernels (qmatvec.h): B rows of up to the
    // largest per-token K, as one flattened vector of q8_max_k values
    float* q8mem = nullptr;
    qmatvec_q8 q8;
    int q8_max_k = 0;

    ShmRegion mem;

    bool alloc(const inference_model* m, int n_ctx, int rows) {
        const int E = m->n_embd, HQ = m->n_head, HKV = m->n_head_kv, HD = m->head_dim, V = m->vocab_size;
        const int FF = std::max(m->n_ff, m->n_ff_shexp);  // dense FFN or the shared expert
        const int conv_dim = 2 * m->ssm_n_group * m->ssm_d_state + m->ssm_d_inner, DI = m->ssm_d_inner;
        const int NV = m->ssm_dt_rank, NE = m->n_expert, NU = m->n_expert_used, FE = m->n_ff_exp;
        const int HCD = m->n_hc * E, NHC = m->n_hc, R = m->hc_rank, PE = m->ple_table.data ? E : 0;
        const int RP = (R + 127) / 128 * 128;  // QwenModel::hc_up_pad
        const int PHCD = m->ple_table.data ? HCD : 0;
        const int GU = m->gemma4 ? 2 * FE : 0;  // Gemma-4's fused gate|up expert rows
        B = rows;
        const size_t b = (size_t)rows;
        q8_max_k = (std::max({E, HQ * HD, DI, FF, NU * FE, HCD}) + 255) / 256 * 256 * rows;
        const size_t sizes[] = {b * E, b * E, b * E, b * 2 * HQ * HD, b * HQ * HD,
                                b * HKV * HD, b * HKV * HD, b * HQ * HD, b * E,
                                b * FF, b * FF, b * FF, (size_t)V,
                                b * conv_dim, b * conv_dim, b * DI, b * NV, b * NV, b * DI,
                                b * NE, b * NU, b * NU, b, b * NU * FE, b * NU * FE, b * NU * FE, b * NU * E, b * E,
                                b * NU, b * NU, 1, b * NU, b * NU, b * NU,
                                attention_part_ml_floats(HQ, n_ctx, rows), attention_part_acc_floats_d(HQ, n_ctx, rows, HD),
                                b * HCD, b * HCD, b * HCD, b * R, b * RP, b * HCD, b * NHC,
                                b * PE, b * PHCD, b * PE, b * PHCD, b * PHCD, b * NU * GU,
                                (qmatvec_q8_bytes(q8_max_k) + 3) / 4};
        float *ids_f = nullptr, *gt = nullptr, *gp = nullptr, *gn = nullptr, *ge = nullptr, *gs = nullptr, *gc = nullptr;
        float** const ptrs[] = {&x, &x_alt, &xn, &qfull, &q, &k, &v, &attn, &proj, &ffg, &ffu, &act, &logits,
                                &qkv, &conv, &z, &alpha, &beta, &lin,
                                &r_logits, &ids_f, &wts, &sg, &moe_g, &moe_u, &moe_act, &moe_down, &sh_out,
                                &gt, &gp, &gn, &ge, &gs, &gc,
                                &part_ml, &part_acc,
                                &res, &res_alt, &hxn, &hlo, &hlo_pad, &hgate, &inj,
                                &ple_emb, &ple_key, &ple_val, &ple_gated, &ple_normed, &moe_gu,
                                &q8mem};
        static_assert(sizeof(sizes) / sizeof(sizes[0]) == sizeof(ptrs) / sizeof(ptrs[0]), "scratch table");
        if (!mem.alloc("/dev/shm/qwen_scratch.bin", sizes, ptrs, sizeof(sizes) / sizeof(sizes[0]),
                       hipExtHostRegisterCoarseGrained))
            return false;
        ids = (int*)ids_f;
        g_tok = (int*)gt;
        g_pair = (int*)gp;
        g_ntiles = (int*)gn;
        g_tile_e = (int*)ge;
        g_tile_start = (int*)gs;
        g_tile_n = (int*)gc;
        qmatvec_q8_bind(&q8, q8mem, q8_max_k);
        return true;
    }
};

// Recurrent state of the Gated DeltaNet layers, persisted across tokens like the KV cache: per linear
// layer a conv state [conv_dim][d_conv - 1] and a delta-rule state of dt_rank heads x d_state^2 floats (laid out
// by deltanet_state_index, deltanet.h). One sequence; zeroed when a sequence starts at pos 0. Only the
// GPU touches it (the kernels, and hipMemsetAsync for the reset), so it is registered coarse-grained.
//
// qwen4exp adds, per PLE layer, the PLE conv's input history [(ple_conv - 1) * ple_ngram][n_hc * n_embd], and once
// the last ple_ngram - 1 tokens of the sequence (ple_hist, int32, most recent last), which the CPU hashes the next
// tokens with: kept here so that checkpoints and store snapshots carry them.
struct RecurrentState {
    std::vector<float*> conv, ssm, ple;   // indexed by layer; nullptr where the layer has none
    int32_t* ple_hist = nullptr;
    ShmRegion mem;

    bool alloc(const inference_model* eng) {
        const int conv_dim = 2 * eng->ssm_n_group * eng->ssm_d_state + eng->ssm_d_inner;
        conv.assign(eng->n_layer, nullptr);
        ssm.assign(eng->n_layer, nullptr);
        ple.assign(eng->n_layer, nullptr);
        std::vector<size_t> sizes;
        std::vector<float**> ptrs;
        float* hist = nullptr;
        for (int l = 0; l < eng->n_layer; ++l) {
            if (eng->layers[l].ple) {
                sizes.push_back((size_t)(eng->ple_conv - 1) * eng->ple_ngram * eng->n_hc * eng->n_embd);
                ptrs.push_back(&ple[l]);
            }
            if (eng->layers[l].kind != LAYER_LINEAR_ATTN) continue;
            sizes.push_back((size_t)conv_dim * (eng->ssm_d_conv - 1));
            ptrs.push_back(&conv[l]);
            sizes.push_back((size_t)eng->ssm_dt_rank * eng->ssm_d_state * eng->ssm_d_state);
            ptrs.push_back(&ssm[l]);
        }
        if (eng->ple_table.data) {
            sizes.push_back((size_t)eng->ple_ngram - 1);
            ptrs.push_back(&hist);
        }
        if (sizes.empty()) return true;
        if (!mem.alloc("/dev/shm/qwen_recurrent_state.bin", sizes.data(), ptrs.data(), sizes.size(),
                       hipExtHostRegisterCoarseGrained))
            return false;
        ple_hist = (int32_t*)hist;
        return true;
    }

    hipError_t reset() {
        return mem.base ? hipMemsetAsync(mem.base, 0, mem.total, 0) : hipSuccess;
    }

    // Snapshots of the whole state (prefix reuse: the recurrent state cannot be rewound, so a sequence can only
    // restart from a position where it was saved). GPU-side copies, queued in order with the forward passes.
    std::vector<ShmRegion> ckpt;

    hipError_t save(int slot) {
        if (!mem.base) return hipSuccess;
        if (slot < 0) return hipErrorInvalidValue;
        if ((int)ckpt.size() <= slot) ckpt.resize((size_t)slot + 1);
        if (!ckpt[slot].base) {
            const size_t floats = mem.total / sizeof(float);
            float* p = nullptr;
            float** const pp[] = {&p};
            if (!ckpt[slot].alloc("/dev/shm/qwen_recurrent_ckpt.bin", &floats, pp, 1, hipExtHostRegisterCoarseGrained))
                return hipErrorOutOfMemory;
        }
        return hipMemcpyAsync(ckpt[slot].base, mem.base, mem.total, hipMemcpyDeviceToDevice, 0);
    }

    hipError_t restore(int slot) {
        if (!mem.base) return hipSuccess;
        if (slot < 0 || slot >= (int)ckpt.size() || !ckpt[slot].base) return hipErrorInvalidValue;
        return hipMemcpyAsync(mem.base, ckpt[slot].base, mem.total, hipMemcpyDeviceToDevice, 0);
    }

    void free() {
        for (ShmRegion& c : ckpt) c.free();
        ckpt.clear();
        mem.free();
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
    int q8_k = 0, q8_t = 0;
    // Opt-in Q8_0 copy of a float LM head (qwen_model_set_lm_head_q8), built on the GPU when first enabled
    ShmRegion head_mem;
    int8_t* head_q = nullptr;
    uint16_t* head_d = nullptr;
    bool head_q8 = false;

    // The mixers' up projections (K = hc_rank) zero-padded to a multiple of 128 values, so they run on the int8 /
    // WMMA kernels (ROCmFP4: K % 64 / % 128; the release's rank is 320): one copy of all of them, built at init.
    // Padding blocks are all zero bytes, which decode to 0. Empty when the rank needs no padding or the type
    // has no int8 kernel (then the original weights run on the float path).
    ShmRegion hc_pad_mem;
    std::unordered_map<const HcMix*, QuantWeight> hc_up_pad;

    bool pad_hc_up() {
        std::vector<const HcMix*> mixes;
        for (const LayerWeights& L : eng->layers) mixes.push_back(&L.hc_attn), mixes.push_back(&L.hc_ffn);
        mixes.push_back(&eng->hc_head);
        return pad_hc_up(mixes, hc_pad_mem, "/dev/shm/qwen_hc_up_pad.bin");
    }

    bool pad_hc_up(const std::vector<const HcMix*>& mixes, ShmRegion& mem, const char* path) {
        const int R = eng->hc_rank, RP = (R + 127) / 128 * 128;
        const QuantWeight& U = mixes.back()->up;
        if (RP == R || (U.type != QT_Q4_0_ROCMFP4 && U.type != QT_Q4_0_ROCMFP4_FAST)) return true;
        const size_t prow = qt_row_size(U.type, RP), bytes = prow * U.N;
        size_t floats = (bytes * mixes.size() + 3) / 4;
        float* base = nullptr;
        float** const pp[] = {&base};
        if (!mem.alloc(path, &floats, pp, 1, hipHostRegisterReadOnly)) return false;
        char* p = (char*)base;
        for (const HcMix* M : mixes) {
            const QuantWeight& W = M->up;
            if (W.type != U.type || W.K != R || W.N != U.N) return false;
            for (int n = 0; n < W.N; ++n) {
                memcpy(p + (size_t)n * prow, W.row(n), W.row_bytes);
                memset(p + (size_t)n * prow + W.row_bytes, 0, prow - W.row_bytes);
            }
            QuantWeight P = W;
            P.K = RP;
            P.row_bytes = prow;
            P.data = p;
            hc_up_pad[M] = P;
            p += bytes;
        }
        return true;
    }

    ~QwenModel() {
        gemma_mem.free();
        mtp_free();
        hc_pad_mem.free();
        head_mem.free();
        rs.free();
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
        if (eng->n_kv_layers > 0) {
            const int hd = eng->head_dim;
            if ((hd != 128 && hd != 256 && !(hd == 512 && eng->gemma4)) || (long long)eng->n_head * hd % 256 != 0) {
                qwen_set_error("head_dim %d / %d heads: the attention kernels are built for head_dim 128 or 256 "
                               "(and 512 for Gemma-4)", hd, eng->n_head);
                return false;
            }
            if (attention_init("attention.co") != 0) {
                qwen_set_error("cannot load attention.co (run from build/co)");
                return false;
            }
        }
        // Prefill micro-batch: up to eng->prefill_batch tokens (1024 by default; inference_set_prefill_batch).
        // Bigger batches give each MoE expert more tokens per tile (35B prefill 256 -> 1024 tokens was 845 ->
        // 1,132 tokens/s). Fewer if attention's partial buffers (at most ATTN_MAX_SPLITS slots per token,
        // whatever n_ctx) would exceed ~256 MB per 1024 tokens
        const int n_ctx = kv_cache_n_ctx(eng->kvc);
        const size_t part_per_token =  // ATTN_MAX_SPLITS slots
            attention_part_acc_floats_d(eng->n_head, ATTN_CHUNK * ATTN_MAX_SPLITS, 1, eng->head_dim) * sizeof(float);
        const int cap = eng->prefill_batch > 0 ? eng->prefill_batch : 1024;
        const size_t fit = ((size_t)256 << 20) / 1024 * (size_t)cap / std::max<size_t>(1, part_per_token);
        const int rows = (int)std::max<size_t>(1, std::min<size_t>((size_t)cap, fit));
        if (!s.alloc(eng, n_ctx, rows)) {
            qwen_set_error("scratch allocation failed");
            return false;
        }
        if (!rs.alloc(eng)) {
            qwen_set_error("recurrent state allocation failed");
            return false;
        }
        if (eng->n_hc > 0 && !pad_hc_up()) {
            qwen_set_error("hyper-connection up projections: padded copy failed");
            return false;
        }
        if (eng->gemma4 && !gemma_setup()) return false;
        return true;
    }

    // The int8 buffer for a producer kernel to quantize its T rows of K values into, or NULL (float path).
    // K % 256 == 0: both formats. q8_0_rows: also rows of K % 32 == 0 (only Q8_0 is complete per row, which
    // is all the ROCmFP4 kernels read; qmatvec_q8_supported keeps K-quants to K % 256), for the producers that
    // handle them (swiglu, qmatvec_quantize).
    const qmatvec_q8* q8_for(int K, int T, bool q8_0_rows = false) const {
        return use_q8 && K % (q8_0_rows ? 32 : 256) == 0 && (long long)K * T <= s.q8_max_k ? &s.q8 : nullptr;
    }

    // A producer kernel just wrote T rows of x[K], quantized into q (NULL: not quantized); the next matmuls
    // read it
    void mark(const float* x, int K, int T, const qmatvec_q8* q) {
        q8_src = q ? x : nullptr;
        q8_k = K;
        q8_t = T;
    }

    // T rows of x[K] were written by a kernel that cannot quantize its output: quantize them in a separate launch
    bool produced(const float* x, int K, int T) {
        const qmatvec_q8* q = q8_for(K, T, true);
        if (q && !check(qmatvec_quantize(x, K * T, q, 0), "quantize_q8")) return false;
        mark(x, K, T, q);
        return true;
    }

    // Y[t] = W X[t] for T rows on the GPU (X, Y in pinned memory, rows of W.K / W.N values). c: profile
    // section. Uses the int8 kernels when there is one for the type and s.q8 holds these rows; otherwise the
    // float kernels (one batched launch for F32, else one launch per row).
    bool matvec(const QuantWeight& W, const float* x, float* y, int T, ProfCat c) {
        prof.bytes[c] += (double)W.row_bytes * W.N;
        const bool q8 = q8_src == x && q8_k == W.K && q8_t == T && qmatvec_q8_supported(W.type, W.K);
        hipError_t e = hipSuccess;
        if (q8 && T == 1) e = qmatvec_launch_q8(W.type, W.data, W.row_bytes, &s.q8, y, W.N, W.K, 0);
        else if (q8) e = qmatvec_launch_q8_batch(W.type, W.data, W.row_bytes, &s.q8, y, W.N, W.N, W.K, T, 0);
        else e = qmatvec_launch_batch(W.type, W.data, W.row_bytes, x, y, W.N, W.K, T, 0);
        if (e != hipSuccess) {
            qwen_set_error("qmatvec type %d %dx%d (T %d): %s", W.type, W.N, W.K, T, hipGetErrorString(e));
            return false;
        }
        return true;
    }

    // Two matrices on the same input x: one launch when generating (T = 1) on the int8 path and A, B share
    // the type and K; otherwise two matvecs. Each output row has the same arithmetic either way.
    bool matvec_pair(const QuantWeight& A, const QuantWeight& B, const float* x, float* ya, float* yb, int T,
                     ProfCat c) {
        if (T == 1 && q8_src == x && q8_k == A.K && q8_t == 1 && A.type == B.type && A.K == B.K &&
            A.row_bytes == B.row_bytes && qmatvec_q8_supported(A.type, A.K)) {
            prof.bytes[c] += (double)A.row_bytes * (A.N + B.N);
            return check(qmatvec_launch_q8_pair(A.type, A.data, A.N, ya, B.data, B.N, yb, A.row_bytes, &s.q8, A.K, 0),
                         "matvec pair");
        }
        if (T >= 2 && T <= 4) return matvec_multi({{&A, ya}, {&B, yb}}, x, T, c);  // one launch where they match
        return matvec(A, x, ya, T, c) && matvec(B, x, yb, T, c);
    }

    // Several matrices on the same input x (DeltaNet's qkv / z / alpha / beta, attention's q / k / v): generating
    // (T = 1) on the int8 path, one launch per group of up to 4 with the same type and row size; otherwise one
    // matvec each. Every output row has the same arithmetic either way.
    struct MvOut { const QuantWeight* w; float* y; };
    bool matvec_multi(std::initializer_list<MvOut> list, const float* x, int T, ProfCat c) {
        std::vector<MvOut> todo(list);
        while (!todo.empty()) {
            const QuantWeight& A = *todo[0].w;
            if (q8_src != x || q8_k != A.K || q8_t != T || !qmatvec_q8_few_supported(A.type, A.K, T)) {
                if (!matvec(A, x, todo[0].y, T, c)) return false;
                todo.erase(todo.begin());
                continue;
            }
            const void* Ws[4];
            int Ns[4];
            float* ys[4];
            int n = 0;
            for (size_t i = 0; i < todo.size() && n < 4;) {
                const QuantWeight& B = *todo[i].w;
                if (B.type == A.type && B.K == A.K && B.row_bytes == A.row_bytes) {
                    Ws[n] = B.data, Ns[n] = B.N, ys[n] = todo[i].y, ++n;
                    prof.bytes[c] += (double)B.row_bytes * B.N;
                    todo.erase(todo.begin() + (long)i);
                } else {
                    ++i;
                }
            }
            if (!check(qmatvec_launch_q8_multi_t(A.type, n, Ws, Ns, ys, A.row_bytes, &s.q8, A.K, T, 0), "matvec group"))
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

    // Gated full attention for T tokens at positions pos0 .., from the normed input s.xn; output in s.proj.
    // q/k norm, RoPE, the KV cache write, causal attention and the output gate are GPU kernels (attention.hip).
    bool full_attention(const LayerWeights& L, int pos0, int T) {
        const int HQ = eng->n_head, HKV = eng->n_head_kv, DQ = HQ * eng->head_dim;
        if (!matvec_multi({{&L.Wq, s.qfull}, {&L.Wk, s.k}, {&L.Wv, s.v}}, s.xn, T, P_ATTN)) return false;
        inference_kv_cache_t* kvc = kvc_cur ? kvc_cur : eng->kvc;
        __half* K = kv_cache_k_row(kvc, L.kv_slot, 0);
        __half* V = kv_cache_v_row(kvc, L.kv_slot, 0);
        if (!K || !V) { qwen_set_error("no KV cache for KV layer %d", L.kv_slot); return false; }
        const qmatvec_q8* q8 = q8_for(DQ, T);
        if (!check(attention_forward(s.qfull, s.k, s.v, L.q_norm, L.k_norm, K, V, kv_cache_block_shift(kvc),
                                     kv_cache_block_stride(kvc), HQ, HKV, eng->n_rot, pos0, T,
                                     eng->rope_freq_base, (float)eng->eps, s.q, s.part_ml, s.part_acc, s.attn,
                                     eng->head_dim, eng->attn_gated ? 1 : 0, q8, 0),
                   "attention kernels"))
            return false;
        mark(s.attn, DQ, T, q8);
        return matvec(L.Wo, s.attn, s.proj, T, P_ATTN);
    }

    // Gated DeltaNet token mixer for T tokens, from s.xn; output in s.proj. Updates the layer's conv and
    // delta-rule state. Follows llama.cpp's qwen35 build_layer_attn_linear (deltanet.hip).
    bool linear_attention(int l, const LayerWeights& L, int T) {
        const int NG = eng->ssm_n_group, NV = eng->ssm_dt_rank;
        const float eps = (float)eng->eps;
        if (!matvec_multi({{&L.W_qkv, s.qkv}, {&L.W_z, s.z}, {&L.W_alpha, s.alpha}, {&L.W_beta, s.beta}}, s.xn, T,
                          P_LIN))
            return false;
        // conv1d + SiLU + q/k L2 norm + gates, then the delta rule + gated norm; alpha/beta become
        // g / sigmoid(beta) in place
        // a verify pass (spec_mid) also keeps the states after its first T - 1 tokens in rs_mid
        const deltanet_mid_states mc = mids(&RecurrentState::conv, l, T), ms = mids(&RecurrentState::ssm, l, T);
        if (!check(deltanet_pre(s.qkv, rs.conv[l], L.ssm_conv1d, eng->ssm_d_conv, s.conv, NG, NV, s.alpha, s.beta,
                                L.ssm_dt, L.ssm_a, s.alpha, s.beta, eps, T, spec_mid ? &mc : nullptr, 0),
                   "deltanet_pre") ||
            !check(deltanet_rule(rs.ssm[l], s.conv, s.alpha, s.beta, s.z, L.ssm_norm, s.lin, NG, NV, eps, T,
                                 eng->gdn_sigmoid_gate ? 1 : 0, q8_for(eng->ssm_d_inner, T), spec_mid ? &ms : nullptr,
                                 0),
                   "deltanet_rule"))
            return false;
        mark(s.lin, eng->ssm_d_inner, T, q8_for(eng->ssm_d_inner, T));
        return matvec(L.W_ssm_out, s.lin, s.proj, T, P_LIN);
    }

    // Expert matvecs for T tokens with the experts in s.ids (n_used per token): int8 when s.q8 holds the input
    // rows (T of them, or T * n_used with x_per_slot), else float. W stacks all experts' N rows. With `grouped`
    // (a prefill batch after gpu_moe_group), the expert-tiled matmul: each expert's weights are decoded once
    // per tile of up to s.g_tile of its (token, slot) entries instead of once per entry.
    bool matvec_id(const QuantWeight& W, const float* x, int x_per_slot, float* y, int N, int T, bool grouped,
                   ProfCat c) {
        const int NU = eng->n_expert_used, K = W.K;
        prof.bytes[c] += (double)W.row_bytes * N * NU;
        const int rows_in = x_per_slot ? T * NU : T;
        const bool q8 = q8_src == x && q8_k == K && q8_t == rows_in && qmatvec_q8_supported(W.type, K);
        hipError_t e;
        if (q8 && grouped && qmatvec_q8_id_batch_supported(W.type, K))
            e = qmatvec_launch_q8_id_batch(W.type, W.data, W.row_bytes, &s.q8, y, N, K,
                                           gpu_moe_max_tiles(T, NU, eng->n_expert, s.g_tile), s.g_tile, s.g_ntiles,
                                           s.g_tile_e, s.g_tile_start, s.g_tile_n, x_per_slot ? s.g_pair : s.g_tok,
                                           s.g_pair, 0);
        else if (q8)
            e = qmatvec_launch_q8_id(W.type, W.data, W.row_bytes, s.ids, NU, x_per_slot, &s.q8, y, N, K, T, 0);
        else if (grouped && qmatvec_id_batch_f_supported(W.type))
            e = qmatvec_launch_id_batch_f(W.type, W.data, W.row_bytes, x, y, N, K,
                                          gpu_moe_max_tiles(T, NU, eng->n_expert, s.g_tile), s.g_tile, s.g_ntiles,
                                          s.g_tile_e, s.g_tile_start, s.g_tile_n, x_per_slot ? s.g_pair : s.g_tok,
                                          s.g_pair, 0);
        else
            e = qmatvec_launch_id(W.type, W.data, W.row_bytes, s.ids, NU, x_per_slot, x, y, N, K, T, 0);
        if (e != hipSuccess) {
            qwen_set_error("expert matvec type %d %dx%d (T %d): %s", W.type, N, K, T, hipGetErrorString(e));
            return false;
        }
        return true;
    }

    // The experts' gate and up projections (same input): one launch when generating on the int8 path
    bool matvec_id_gate_up(const QuantWeight& G, const QuantWeight& U, const float* x, float* yg, float* yu, int N,
                           int T, bool grouped, ProfCat c) {
        if (!grouped && q8_src == x && q8_k == G.K && q8_t == T && G.type == U.type && G.K == U.K &&
            G.row_bytes == U.row_bytes && qmatvec_q8_supported(G.type, G.K)) {
            const int NU = eng->n_expert_used;
            prof.bytes[c] += 2.0 * G.row_bytes * N * NU;
            return check(qmatvec_launch_q8_id_pair(G.type, G.data, U.data, G.row_bytes, s.ids, NU, 0, &s.q8, yg, yu, N,
                                                   G.K, T, 0),
                         "expert gate/up matvec");
        }
        return matvec_id(G, x, 0, yg, N, T, grouped, c) && matvec_id(U, x, 0, yu, N, T, grouped, c);
    }

    // Mixture-of-experts FFN (qwen35moe; llama.cpp build_moe_ffn + the gated shared expert), all on the GPU:
    // router logits -> softmax / top-k / renormalized weights + shared gate (moe_router) -> the selected experts'
    // gate/up, SwiGLU, down (expert-indexed matvecs reading s.ids) -> weighted sum + gated shared expert
    // (moe_combine). The residual adds and norms around it are the same as the dense FFN's.
    bool moe_ffn(const LayerWeights& L, const float* next_norm, int T) {
        const int E = eng->n_embd;
        const float eps = (float)eng->eps;
        const qmatvec_q8* qe = q8_for(E, T);
        if (!check(gpu_add_rmsnorm(s.x, s.proj, s.x_alt, L.ffn_norm, s.xn, E, T, eps, qe, 0), "add_rmsnorm"))
            return false;
        std::swap(s.x, s.x_alt);
        mark(s.xn, E, T, qe);
        if (!moe_core(L, T) || !check(gpu_add_rmsnorm(s.x, s.proj, s.x_alt, next_norm, s.xn, E, T, eps, qe, 0),
                                      "add_rmsnorm"))
            return false;
        std::swap(s.x, s.x_alt);
        mark(s.xn, E, T, qe);
        return true;
    }

    // Whether the shared expert can ride along as one more slot of the experts' int8 launches (one token): the
    // same width, type and row size as the experts, and s.xn quantized
    bool shared_as_slot(const LayerWeights& L) const {
        const int E = eng->n_embd, FE = eng->n_ff_exp;
        return s.B >= 2 && eng->n_ff_shexp == FE && q8_src == s.xn && q8_k == E && q8_t == 1 &&
               L.W_gate.type == L.We_gate.type && L.W_up.type == L.We_up.type && L.We_gate.type == L.We_up.type &&
               L.W_down.type == L.We_down.type && L.W_gate.row_bytes == L.We_gate.row_bytes &&
               L.W_down.row_bytes == L.We_down.row_bytes && qmatvec_q8_supported(L.We_gate.type, E) &&
               qmatvec_q8_supported(L.We_down.type, FE);
    }

    // moe_core for one token with the shared expert as slot NU of the experts' launches (ids[NU] = -1 from the
    // router): gate/up, SwiGLU and down of the experts and the shared expert in 3 launches instead of 6. The
    // arithmetic per row is the same, and moe_combine adds the shared output last as before, so the results are
    // the same bit for bit.
    bool moe_core_one(const LayerWeights& L) {
        const int E = eng->n_embd, NE = eng->n_expert, NU = eng->n_expert_used, FE = eng->n_ff_exp, NS = NU + 1;
        prof.bytes[P_FFN] += 2.0 * L.We_gate.row_bytes * FE * NS + (double)L.We_down.row_bytes * E * NS;
        if (!matvec(L.W_router, s.xn, s.r_logits, 1, P_FFN) ||
            !check(gpu_moe_router(s.r_logits, s.xn, L.shexp_gate, s.ids, s.wts, s.sg, NE, NU, E, 1, 0, NS),
                   "moe_router") ||
            !check(qmatvec_launch_q8_id_x(L.We_gate.type, L.We_gate.data, L.We_up.data, L.W_gate.data, L.W_up.data,
                                          L.We_gate.row_bytes, s.ids, NS, 0, &s.q8, s.moe_g, s.moe_u, FE, E, 1, 0),
                   "expert gate/up matvec"))
            return false;
        const qmatvec_q8* qx = q8_for(FE, NS, true);
        if (!check(gpu_swiglu(s.moe_g, s.moe_u, s.moe_act, NS * FE, qx, 0), "swiglu (experts)")) return false;
        mark(s.moe_act, FE, NS, qx);
        if (!qx) {
            qwen_set_error("moe_core_one: no int8 buffer for the expert inputs");
            return false;
        }
        return check(qmatvec_launch_q8_id_x(L.We_down.type, L.We_down.data, nullptr, L.W_down.data, nullptr,
                                            L.We_down.row_bytes, s.ids, NS, 1, &s.q8, s.moe_down, nullptr, E, FE, 1, 0),
                     "expert down matvec") &&
               check(gpu_moe_combine(s.moe_down, s.wts, s.moe_down + (size_t)NU * E, s.sg, s.proj, NU, E, 1, 0),
                     "moe_combine");
    }

    // The MoE FFN itself, from s.xn (marked) to s.proj
    bool moe_core(const LayerWeights& L, int T) {
        const int E = eng->n_embd, NE = eng->n_expert, NU = eng->n_expert_used, FE = eng->n_ff_exp;
        const int FS = eng->n_ff_shexp;
        if (T == 1 && shared_as_slot(L)) return moe_core_one(L);
        // A prefill batch groups its (token, slot) entries by expert for the expert-tiled matmuls. A few tokens
        // (a speculative verify pass) take the per-token kernels, which give exactly what decoding them one at a
        // time gives
        const bool grouped = T >= 8;
        if (grouped)  // the largest tile all three expert matrices' kernels take
            s.g_tile = std::min({qmatvec_id_batch_tile(L.We_gate.type, FE, E),
                                 qmatvec_id_batch_tile(L.We_up.type, FE, E),
                                 qmatvec_id_batch_tile(L.We_down.type, E, FE)});
        if (!matvec(L.W_router, s.xn, s.r_logits, T, P_FFN) ||
            !check(gpu_moe_router(s.r_logits, s.xn, L.shexp_gate, s.ids, s.wts, s.sg, NE, NU, E, T, 0, 0),
                   "moe_router") ||
            (grouped && !check(gpu_moe_group(s.ids, T, NU, NE, s.g_tile, s.g_tok, s.g_pair, s.g_ntiles,
                                             s.g_tile_e, s.g_tile_start, s.g_tile_n, 0), "moe_group")) ||
            !matvec_id_gate_up(L.We_gate, L.We_up, s.xn, s.moe_g, s.moe_u, FE, T, grouped, P_FFN) ||
            !matvec_pair(L.W_gate, L.W_up, s.xn, s.ffg, s.ffu, T, P_FFN))
            return false;

        const qmatvec_q8* qx = q8_for(FE, T * NU, true);
        if (!check(gpu_swiglu(s.moe_g, s.moe_u, s.moe_act, T * NU * FE, qx, 0), "swiglu (experts)")) return false;
        mark(s.moe_act, FE, T * NU, qx);
        if (!matvec_id(L.We_down, s.moe_act, 1, s.moe_down, E, T, grouped, P_FFN)) return false;

        const qmatvec_q8* qs = q8_for(FS, T, true);
        if (!check(gpu_swiglu(s.ffg, s.ffu, s.act, T * FS, qs, 0), "swiglu (shared expert)")) return false;
        mark(s.act, FS, T, qs);
        return matvec(L.W_down, s.act, s.sh_out, T, P_FFN) &&
               check(gpu_moe_combine(s.moe_down, s.wts, s.sh_out, s.sg, s.proj, NU, E, T, 0), "moe_combine");
    }

    // Residual add of the mixer output, the FFN's RMSNorm, SwiGLU FFN, then the residual add of its output
    // fused with the next RMSNorm (next layer's attn_norm, or output_norm after the last layer)
    bool ffn(const LayerWeights& L, const float* next_norm, int T) {
        if (L.moe) return moe_ffn(L, next_norm, T);
        const int E = eng->n_embd, FF = eng->n_ff;
        const float eps = (float)eng->eps;
        const qmatvec_q8 *qe = q8_for(E, T), *qf = q8_for(FF, T, true);
        if (!check(gpu_add_rmsnorm(s.x, s.proj, s.x_alt, L.ffn_norm, s.xn, E, T, eps, qe, 0), "add_rmsnorm"))
            return false;
        std::swap(s.x, s.x_alt);
        mark(s.xn, E, T, qe);
        if (!matvec_pair(L.W_gate, L.W_up, s.xn, s.ffg, s.ffu, T, P_FFN) ||
            !check(gpu_swiglu(s.ffg, s.ffu, s.act, FF * T, qf, 0), "swiglu"))
            return false;
        mark(s.act, FF, T, qf);
        if (!matvec(L.W_down, s.act, s.proj, T, P_FFN) ||
            !check(gpu_add_rmsnorm(s.x, s.proj, s.x_alt, next_norm, s.xn, E, T, eps, qe, 0), "add_rmsnorm"))
            return false;
        std::swap(s.x, s.x_alt);
        mark(s.xn, E, T, qe);
        return true;
    }

    // T consecutive tokens at positions pos0 .. pos0 + T - 1 (T <= s.B): the CPU decodes their embedding rows into
    // the residual stream, then the whole forward pass is one GPU queue with a single sync before the last
    // token's logits are read (more syncs when profiling or when a layer callback / QWEN_DEBUG wants the residual
    // after each layer). Every kernel computes each token exactly as it would one at a time, so a prefill gives
    // the same results as feeding the tokens one by one.
    int forward(const int* tokens, int T, int pos0, float* logits_out) {
        const int E = eng->n_embd;
        const float eps = (float)eng->eps;
        if (T <= 0 || T > s.B) { qwen_set_error("batch of %d tokens (max %d)", T, s.B); return -1; }
        if (!eng->kvc || pos0 < 0 || pos0 + T > kv_cache_n_ctx(eng->kvc)) {
            qwen_set_error("positions %d..%d out of range", pos0, pos0 + T - 1);
            return -1;
        }

        if (eng->n_hc > 0) return forward_hc(tokens, T, pos0, logits_out);
        if (eng->gemma4) return forward_gemma(tokens, T, pos0, logits_out);
        prof.start();
        for (int t = 0; t < T; ++t) {
            if (tokens[t] < 0 || tokens[t] >= eng->vocab_size) { qwen_set_error("token %d out of range", tokens[t]); return -1; }
            if (qt_to_float(eng->token_embd.type, eng->token_embd.row(tokens[t]), s.x + (size_t)t * E, E) != 0) {
                qwen_set_error("cannot decode token_embd type %d", eng->token_embd.type);
                return -1;
            }
        }
        // A sequence starting at position 0 starts from zero recurrent state (the KV cache needs no reset:
        // attention only reads positions up to its own)
        if (pos0 == 0 && !check(rs.reset(), "recurrent state reset")) return -1;
        if (!check(gpu_add_rmsnorm(s.x, nullptr, s.x, eng->layers[0].attn_norm, s.xn, E, T, eps, q8_for(E, T), 0),
                   "add_rmsnorm"))
            return -1;
        mark(s.xn, E, T, q8_for(E, T));
        prof.lap(P_EMBED);

        const bool per_layer = layer_cb || qwen_debug();
        for (int l = 0; l < eng->n_layer; ++l) {
            const LayerWeights& L = eng->layers[l];
            const bool attn = L.kind == LAYER_FULL_ATTN;
            if (!(attn ? full_attention(L, pos0, T) : linear_attention(l, L, T)) || !section(attn ? P_ATTN : P_LIN))
                return -1;
            const float* next_norm = l + 1 < eng->n_layer ? eng->layers[l + 1].attn_norm : eng->output_norm;
            if (!ffn(L, next_norm, T) || !section(P_FFN)) return -1;
            if (per_layer) {
                if (!sync()) return -1;
                for (int t = 0; t < T; ++t) {
                    const float* xt = s.x + (size_t)t * E;
                    if (layer_cb) layer_cb(l, pos0 + t, xt, E, layer_cb_user);
                    if (qwen_debug()) {
                        double ss = 0.0;
                        for (int i = 0; i < E; ++i) ss += (double)xt[i] * xt[i];
                        fprintf(stderr, "[qwen] pos %d blk %2d %-6s rms(x)=%.5g x[0..3]=%.5g %.5g %.5g %.5g\n", pos0 + t,
                                l, attn ? "attn" : "linear", sqrt(ss / E), xt[0], xt[1], xt[2], xt[3]);
                    }
                }
                prof.start();  // the callback and debug trace are not charged
            }
        }

        return lm_head(T, logits_out);
    }

    // The LM head on the last of T tokens of s.xn (its row of the quantized buffer when s.q8 holds them), then
    // the sync that ends the forward pass; the logits are copied to logits_out if given
    int lm_head(int T, float* logits_out) {
        // a verify pass needs the logits of all its tokens (rows of spec_logits; the last one also in s.logits):
        // one launch on all rows where there is a few-token kernel for the head's type (each row as the
        // single-row launch gives it), else one per row
        const QuantWeight& H = eng->output;
        const int E = eng->n_embd, V = eng->vocab_size;
        if (spec_mid && T >= 2 && !head_q8 && q8_src == s.xn && qmatvec_q8_few_supported(H.type, E, T)) {
            qmatvec_q8 rows = s.q8;
            rows.max_k = T * E;
            const void* Ws[] = {H.data};
            const int Ns[] = {V};
            float* ys[] = {spec_logits};
            prof.bytes[P_HEAD] += (double)H.row_bytes * H.N;
            if (!check(qmatvec_launch_q8_multi_t(H.type, 1, Ws, Ns, ys, H.row_bytes, &rows, E, T, 0),
                       "LM head (rows)") ||
                !check(hipMemcpyAsync(s.logits, spec_logits + (size_t)(T - 1) * V, sizeof(float) * V,
                                      hipMemcpyDeviceToDevice, 0),
                       "LM head (rows)") ||
                !sync())
                return -1;
            if (logits_out) memcpy(logits_out, s.logits, sizeof(float) * V);
            prof.lap(P_HEAD);
            prof.tokens += T;
            return 0;
        }
        if (spec_mid)
            for (int r = 0; r + 1 < T; ++r)
                if (lm_head_with(H, spec_logits + (size_t)r * V, T, nullptr, true, r) != 0) return -1;
        return lm_head_with(H, s.logits, T, logits_out, true);
    }

    // The LM head H (the model's, or the MTP head's) on row `row` (default the last) of T rows of s.xn into
    // logits, then the sync
    int lm_head_with(const QuantWeight& H, float* logits, int T, float* logits_out, bool may_q8_copy, int row = -1) {
        const int E = eng->n_embd;
        if (row < 0) row = T - 1;
        // LM head on one token only: its row of s.xn, and its row of the quantized buffer
        const float* xl = s.xn + (size_t)row * E;
        qmatvec_q8 last = s.q8;
        if (q8_src == s.xn) {
            const size_t r = (size_t)row * E;
            last.q0 += r;
            last.qK += r;
            last.d0 += r / 32;
            last.dK += r / 256;
            last.bsK += r / 16;
            last.s0 += r / 32;
            last.max_k = E;
        }
        hipError_t e;
        if (may_q8_copy && head_q8 && q8_src == s.xn) {  // opt-in Q8_0 copy
            prof.bytes[P_HEAD] += (double)H.N * H.K * (1.0 + 2.0 / 32);
            e = qmatvec_launch_q8_0s(head_q, head_d, &last, logits, H.N, E, 0);
        } else {
            prof.bytes[P_HEAD] += (double)H.row_bytes * H.N;
            e = q8_src == s.xn && qmatvec_q8_supported(H.type, E)
                    ? qmatvec_launch_q8(H.type, H.data, H.row_bytes, &last, logits, H.N, E, 0)
                    : qmatvec_launch(H.type, H.data, H.row_bytes, xl, logits, H.N, E, 0);
        }
        if (!check(e, "LM head") || !sync()) return -1;
        if (logits_out) memcpy(logits_out, logits, (size_t)eng->vocab_size * sizeof(float));
        prof.lap(P_HEAD);
        prof.tokens += T;
        return 0;
    }

    // ------------------------------------------------------------ Gemma-4 (gemma4; llama.cpp gemma4.cpp)
    //
    // Per layer, from the residual x: attention on RMSNorm(x) * attn_norm (q / k norms, V normed, RoPE and sliding
    // window per layer, scale 1), x' = x + RMSNorm(attn) * post_attention_norm; the dense GeGLU MLP on
    // RMSNorm(x') * ffn_norm and the experts on RMSNorm(x') * pre_ffw_norm_2 (router input RMSNorm(x') *
    // ffn_gate_inp.scale / sqrt(E); softmax top-k renormalized; fused gate|up; per-expert output scale), each
    // normed, summed and normed again: x = (x' + RMSNorm(mlp_n + moe_n) * post_ffw_norm) * layer_output_scale.
    // The embeddings are scaled by sqrt(E), the logits soft-capped.

    ShmRegion gemma_mem;                 // the router inputs' gammas (ffn_gate_inp.scale / sqrt(E)), per layer
    std::vector<const float*> router_w;  // [n_layer]: into gemma_mem, NULL without experts

    bool gemma_setup() {
        const int E = eng->n_embd, NL = eng->n_layer;
        size_t floats = (size_t)NL * E;
        float* base = nullptr;
        float** const pp[] = {&base};
        if (!gemma_mem.alloc("/dev/shm/qwen_gemma_router.bin", &floats, pp, 1, hipExtHostRegisterCoarseGrained)) {
            qwen_set_error("Gemma-4 router gammas: allocation failed");
            return false;
        }
        router_w.assign(NL, nullptr);
        const float inv = 1.0f / sqrtf((float)E);
        for (int l = 0; l < NL; ++l) {
            const LayerWeights& L = eng->layers[l];
            if (!L.moe) continue;
            float* w = base + (size_t)l * E;
            for (int i = 0; i < E; ++i) w[i] = L.router_scale[i] * inv;
            router_w[l] = w;
        }
        return true;
    }

    // Attention for T tokens from s.xn (marked) into s.proj
    bool gemma_attention(const LayerWeights& L, int pos0, int T) {
        const int HQ = eng->n_head, DQ = HQ * L.head_dim;
        if (L.v_from_k ? !matvec_multi({{&L.Wq, s.qfull}, {&L.Wk, s.k}}, s.xn, T, P_ATTN)
                       : !matvec_multi({{&L.Wq, s.qfull}, {&L.Wk, s.k}, {&L.Wv, s.v}}, s.xn, T, P_ATTN))
            return false;
        // the sliding layers' ring (one block of swa_ring positions, block stride 0: position p in slot p % R)
        inference_kv_cache_t* kvc = L.kv_ring ? eng->kvc_swa : eng->kvc;
        __half* K = kv_cache_k_row(kvc, L.kv_slot, 0);
        __half* V = kv_cache_v_row(kvc, L.kv_slot, 0);
        if (!K || !V) { qwen_set_error("no KV cache for KV layer %d", L.kv_slot); return false; }
        attention_opts o = {};
        o.head_dim = L.head_dim;
        o.kv_stride = kv_cache_d_kv(kvc);
        o.rot_pairs = L.rot_pairs;
        o.v_norm = 1;
        o.window = L.window;
        o.scale = 1.0f;
        const qmatvec_q8* q8 = q8_for(DQ, T);
        if (!check(attention_forward_opts(s.qfull, s.k, L.v_from_k ? s.k : s.v, L.q_norm, L.k_norm, K, V,
                                          kv_cache_block_shift(kvc), L.kv_ring ? 0 : kv_cache_block_stride(kvc), HQ,
                                          L.n_head_kv,
                                          L.n_rot, pos0, T, L.rope_base, (float)eng->eps, s.q, s.part_ml, s.part_acc,
                                          s.attn, &o, q8, 0),
                   "attention kernels"))
            return false;
        mark(s.attn, DQ, T, q8);
        // GEMMA_DUMP_LAYER=<layer>: write Q (normed, rotated) and the attention output of that layer as raw floats
        // to $GEMMA_DUMP_DIR/{q,attn}.f32 (debugging against llama.cpp's Qcur_pos / kqv_out)
        static const char* dump = getenv("GEMMA_DUMP_LAYER");
        if (dump && atoi(dump) == (int)(&L - eng->layers.data())) {
            if (!sync()) return false;
            const char* dir = getenv("GEMMA_DUMP_DIR") ? getenv("GEMMA_DUMP_DIR") : ".";
            const std::pair<const char*, const float*> outs[] = {{"q", s.q}, {"attn", s.attn}};
            for (const auto& o2 : outs) {
                const std::string path = std::string(dir) + "/" + o2.first + ".f32";
                if (FILE* f = fopen(path.c_str(), "wb")) {
                    fwrite(o2.second, sizeof(float), (size_t)T * DQ, f);
                    fclose(f);
                }
            }
        }
        return matvec(L.Wo, s.attn, s.proj, T, P_ATTN);
    }

    // The layer after attention: s.x (layer input), s.proj (attention output) -> s.x (layer output); s.x_alt holds
    // the post-attention residual
    bool gemma_ffn(int l, const LayerWeights& L, int T) {
        const int E = eng->n_embd, FF = eng->n_ff, NE = eng->n_expert, NU = eng->n_expert_used, FE = eng->n_ff_exp;
        const float eps = (float)eng->eps;
        const qmatvec_q8* qe = q8_for(E, T);
        if (!check(gpu_add_normed(s.x, s.proj, L.post_attn_norm, nullptr, s.x_alt, E, T, eps, 0), "add_normed"))
            return false;
        // dense MLP -> s.proj
        const qmatvec_q8* qf = q8_for(FF, T, true);
        if (!check(gpu_add_rmsnorm(s.x_alt, nullptr, s.x_alt, L.ffn_norm, s.xn, E, T, eps, qe, 0), "rmsnorm"))
            return false;
        mark(s.xn, E, T, qe);
        if (!matvec_pair(L.W_gate, L.W_up, s.xn, s.ffg, s.ffu, T, P_FFN) ||
            !check(gpu_geglu(s.ffg, s.ffu, s.act, T, FF, FF, FF, qf, 0), "geglu"))
            return false;
        mark(s.act, FF, T, qf);
        if (!matvec(L.W_down, s.act, s.proj, T, P_FFN)) return false;
        const float* h = s.proj;  // what post_ffw_norm normalizes: the MLP output, or the sum of both normed
        if (L.moe) {
            // router (float: the router is F32) on its own norm of x' -> ids / wts
            if (!check(gpu_add_rmsnorm(s.x_alt, nullptr, s.x_alt, router_w[l], s.sh_out, E, T, eps, nullptr, 0),
                       "rmsnorm (router)") ||
                !matvec(L.W_router, s.sh_out, s.r_logits, T, P_FFN) ||
                !check(gpu_moe_router(s.r_logits, s.sh_out, nullptr, s.ids, s.wts, s.sg, NE, NU, E, T, 0, 0),
                       "moe_router"))
                return false;
            const bool grouped = T >= 8;
            if (grouped)
                s.g_tile = std::min(qmatvec_id_batch_tile(L.We_gate_up.type, 2 * FE, E),
                                    qmatvec_id_batch_tile(L.We_down.type, E, FE));
            if (grouped && !check(gpu_moe_group(s.ids, T, NU, NE, s.g_tile, s.g_tok, s.g_pair, s.g_ntiles, s.g_tile_e,
                                                s.g_tile_start, s.g_tile_n, 0),
                                  "moe_group"))
                return false;
            // experts on x' normed by pre_ffw_norm_2: fused gate|up, GeGLU on the halves, down, combined -> s.sh_out
            if (!check(gpu_add_rmsnorm(s.x_alt, nullptr, s.x_alt, L.pre_ffw_norm2, s.xn, E, T, eps, qe, 0),
                       "rmsnorm (experts)"))
                return false;
            mark(s.xn, E, T, qe);
            if (!matvec_id(L.We_gate_up, s.xn, 0, s.moe_gu, 2 * FE, T, grouped, P_FFN)) return false;
            const qmatvec_q8* qx = q8_for(FE, T * NU, true);
            if (!check(gpu_geglu(s.moe_gu, s.moe_gu + FE, s.moe_act, T * NU, FE, 2 * FE, 2 * FE, qx, 0),
                       "geglu (experts)"))
                return false;
            mark(s.moe_act, FE, T * NU, qx);
            if (!matvec_id(L.We_down, s.moe_act, 1, s.moe_down, E, T, grouped, P_FFN) ||
                !check(gpu_moe_combine_scaled(s.moe_down, s.wts, nullptr, nullptr, s.ids, L.exp_scale, s.sh_out, NU,
                                              E, T, 0),
                       "moe_combine"))
                return false;
            // h = RMSNorm(mlp) * post_ffw_norm_1 + RMSNorm(moe) * post_ffw_norm_2, in s.attn (free after attn_output)
            if (!check(gpu_add_rmsnorm(s.proj, nullptr, s.proj, L.post_ffw_norm1, s.attn, E, T, eps, nullptr, 0),
                       "rmsnorm (mlp)") ||
                !check(gpu_add_normed(s.attn, s.sh_out, L.post_ffw_norm2, nullptr, s.attn, E, T, eps, 0),
                       "add_normed (experts)"))
                return false;
            h = s.attn;
        }
        return check(gpu_add_normed(s.x_alt, h, L.post_ffw_norm, L.out_scale, s.x, E, T, eps, 0), "add_normed (ffn)");
    }

    // The sliding-window ring's high-water mark: positions below ring_hi - swa_ring have been overwritten. A
    // sequence that starts again at 0 starts it again (later slots hold old positions no token reads).
    int ring_hi = 0;

    int forward_gemma(const int* tokens, int T, int pos0, float* logits_out) {
        const int E = eng->n_embd, V = eng->vocab_size;
        const float eps = (float)eng->eps;
        ring_hi = pos0 == 0 ? T : std::max(ring_hi, pos0 + T);
        prof.start();
        for (int t = 0; t < T; ++t) {
            if (tokens[t] < 0 || tokens[t] >= V) { qwen_set_error("token %d out of range", tokens[t]); return -1; }
            float* xt = s.x + (size_t)t * E;
            if (dbg_res_in) {  // qwen_model_run_layers: this residual instead of the embeddings
                memcpy(xt, dbg_res_in + (size_t)t * E, sizeof(float) * E);
                continue;
            }
            if (qt_to_float(eng->token_embd.type, eng->token_embd.row(tokens[t]), xt, E) != 0) {
                qwen_set_error("cannot decode token_embd type %d", eng->token_embd.type);
                return -1;
            }
            for (int i = 0; i < E; ++i) xt[i] *= eng->embd_scale;
        }
        prof.lap(P_EMBED);
        const bool per_layer = layer_cb || qwen_debug();
        const int l0 = dbg_res_in ? dbg_l0 : 0, l1 = dbg_res_in ? dbg_l1 : eng->n_layer;
        for (int l = l0; l < l1; ++l) {
            const LayerWeights& L = eng->layers[l];
            const qmatvec_q8* qe = q8_for(E, T);
            if (!check(gpu_add_rmsnorm(s.x, nullptr, s.x, L.attn_norm, s.xn, E, T, eps, qe, 0), "rmsnorm (attn)"))
                return -1;
            mark(s.xn, E, T, qe);
            if (!gemma_attention(L, pos0, T) || !section(P_ATTN) || !gemma_ffn(l, L, T) || !section(P_FFN)) return -1;
            if (per_layer) {
                if (!sync()) return -1;
                for (int t = 0; t < T; ++t) {
                    const float* xt = s.x + (size_t)t * E;
                    if (layer_cb) layer_cb(l, pos0 + t, xt, E, layer_cb_user);
                    if (qwen_debug()) {
                        double ss = 0.0;
                        for (int i = 0; i < E; ++i) ss += (double)xt[i] * xt[i];
                        fprintf(stderr, "[gemma] pos %d blk %2d %s rms(x)=%.5g x[0..3]=%.5g %.5g %.5g %.5g\n", pos0 + t, l,
                                L.window ? "swa " : "full", sqrt(ss / E), xt[0], xt[1], xt[2], xt[3]);
                    }
                }
                prof.start();
            }
        }
        if (dbg_res_in) return sync() ? 0 : -1;  // no LM head
        const qmatvec_q8* qe = q8_for(E, T);
        if (!check(gpu_add_rmsnorm(s.x, nullptr, s.x, eng->output_norm, s.xn, E, T, eps, qe, 0), "rmsnorm (output)"))
            return -1;
        mark(s.xn, E, T, qe);
        if (lm_head(T, nullptr) != 0) return -1;
        if (eng->logit_softcap > 0.0f &&
            (!check(gpu_softcap(s.logits, V, eng->logit_softcap, 0), "softcap") || !sync()))
            return -1;
        if (logits_out) memcpy(logits_out, s.logits, sizeof(float) * V);
        return 0;
    }

    // ------------------------------------------------------------ hyper-connections (qwen4exp)

    // Combine the block output cur (none: just normalize) into the residual streams with the stream weights in
    // s.inj, then the mixer's per-stream RMSNorm with gamma w into s.hxn (quantized for the down / inject matvecs)
    bool hc_norm(const float* cur, const float* w, int T) {
        const int HCD = eng->n_hc * eng->n_embd;
        const qmatvec_q8* q = q8_for(HCD, T);
        if (!check(gpu_hc_norm(s.res, s.res_alt, cur, s.inj, w, s.hxn, eng->n_embd, eng->n_hc, T, (float)eng->eps, q, 0),
                   "hc_norm"))
            return false;
        if (cur) std::swap(s.res, s.res_alt);
        mark(s.hxn, HCD, T, q);
        return true;
    }

    // The rest of a mixer, from s.hxn: the block input in s.xn (quantized), and s.inj when M has an inject
    bool hc_mix(const HcMix& M, int T, ProfCat c) {
        const int E = eng->n_embd, HC = eng->n_hc, R = eng->hc_rank;
        const auto pad = hc_up_pad.find(&M);
        const QuantWeight& U = pad != hc_up_pad.end() ? pad->second : M.up;
        const qmatvec_q8* ql = q8_for(U.K, T, true);
        if (!(M.inject.data ? matvec_pair(M.down, M.inject, s.hxn, s.hlo, s.inj, T, c)
                            : matvec(M.down, s.hxn, s.hlo, T, c)))
            return false;
        // SiLU, padding and quantizing in the matvec (one token; a 2..4-token version, every workgroup quantizing
        // all rows, was slower than hc_lowrank + the few-token matvec)
        if (T == 1 && ql && qmatvec_q8_lr_supported(U.type, U.K)) {
            prof.bytes[c] += (double)U.row_bytes * U.N;
            if (!check(qmatvec_launch_q8_lr(U.type, U.data, U.row_bytes, s.hlo, R, U.K, 1.0f / (float)HC, s.hgate,
                                            U.N, 0),
                       "hc up (fused)"))
                return false;
        } else {
            if (!check(gpu_hc_lowrank(s.hlo, s.hlo_pad, R, U.K, T, HC, ql, 0), "hc_lowrank")) return false;
            mark(s.hlo_pad, U.K, T, ql);
            if (!matvec(U, s.hlo_pad, s.hgate, T, c)) return false;
        }
        const qmatvec_q8* q = q8_for(E, T);
        if (!check(gpu_hc_mix(s.hxn, s.hgate, s.xn, E, HC, T, q, 0), "hc_mix")) return false;
        mark(s.xn, E, T, q);
        return true;
    }

    // The PLE rows of T tokens at pos0 (CPU): each token's n-gram hash rows, decoded into s.ple_emb, and the
    // token history in rs.ple_hist advanced. As llama.cpp's llm_graph_input_ple::set_input: positions before the
    // sequence start read as EOS, and an EOS among a token's predecessors hides the ones before it.
    bool ple_rows(const int* tokens, int T, int pos0) {
        const int NG = eng->ple_ngram, PER = eng->ple_per_ngram, PD = eng->ple_dim, E = eng->n_embd;
        const int64_t eos = eng->ple_eos;
        int32_t* hist = rs.ple_hist;  // NG - 1 tokens before this batch, most recent last
        if (pos0 == 0)
            for (int i = 0; i < NG - 1; ++i) hist[i] = (int32_t)eos;
        for (int t = 0; t < T; ++t) {
            int64_t ctx[8];
            ctx[0] = tokens[t];
            bool cut = false;
            for (int k = 1; k < NG; ++k) {
                const int j = t - k;
                ctx[k] = cut ? eos : j >= 0 ? tokens[j] : hist[NG - 1 + j];
                if (ctx[k] == eos) cut = true;
            }
            for (int n = 2; n <= NG; ++n) {
                uint64_t mixed = (uint64_t)ctx[0] * eng->ple_mult[0];
                for (int k = 1; k < n; ++k) mixed ^= (uint64_t)ctx[k] * eng->ple_mult[k];
                for (int g = 0; g < PER; ++g) {
                    const int h = (n - 2) * PER + g;
                    const uint64_t row = mixed % eng->ple_vocab[h] + eng->ple_offset[h];
                    if (qt_to_float(eng->ple_table.type, eng->ple_table.row((int)row), s.ple_emb + (size_t)t * E + h * PD,
                                    PD) != 0) {
                        qwen_set_error("cannot decode per_layer_token_embd type %d", eng->ple_table.type);
                        return false;
                    }
                }
            }
        }
        // the last NG - 1 tokens of (history, batch); a verify pass also keeps them after its first token
        std::vector<int32_t> all(hist, hist + NG - 1);
        all.insert(all.end(), tokens, tokens + T);
        if (spec_mid)  // the history after token j of a verify pass
            for (int j = 0; j + 1 < T && j < SPEC_MAX_DRAFTS; ++j)
                if (rs_mid[j].ple_hist) memcpy(rs_mid[j].ple_hist, all.data() + 1 + j, sizeof(int32_t) * (NG - 1));
        memcpy(hist, all.data() + all.size() - (NG - 1), sizeof(int32_t) * (NG - 1));
        return true;
    }

    // PLE on layer l: residual streams += gated value + SiLU(dilated conv of its grouped norm), from s.ple_emb
    bool ple(int l, const LayerWeights& L, int T) {
        const int E = eng->n_embd, HC = eng->n_hc;
        const deltanet_mid_states mp = mids(&RecurrentState::ple, l, T);
        return produced(s.ple_emb, E, T) &&
               matvec_pair(L.ple_key, L.ple_value, s.ple_emb, s.ple_key, s.ple_val, T, P_LIN) &&
               check(gpu_ple_gate(s.ple_key, s.ple_val, s.res, L.ple_norm_key, L.ple_norm_query, L.ple_norm_conv,
                                  s.ple_gated, s.ple_normed, E, HC, T, (float)eng->eps, 0),
                     "ple_gate") &&
               check(gpu_ple_conv(s.ple_normed, s.ple_gated, s.res, rs.ple[l], L.ple_conv1d, HC * E, T, eng->ple_conv,
                                  eng->ple_ngram, spec_mid ? &mp : nullptr, 0),
                     "ple_conv");
    }

    // forward() for hyper-connection models: the residual is n_hc streams; each block reads a mixer's output and
    // its result is combined back into every stream (llama.cpp qwen4exp graph)
    // Tests (qwen_model_run_layers): start from res_in instead of the embeddings, run layers [l0, l1) only, no
    // LM head
    const float* dbg_res_in = nullptr;
    int dbg_l0 = 0, dbg_l1 = -1;

    int forward_hc(const int* tokens, int T, int pos0, float* logits_out) {
        const int E = eng->n_embd, HC = eng->n_hc, HCD = HC * E;
        const int l0 = dbg_res_in ? dbg_l0 : 0, l1 = dbg_res_in ? dbg_l1 : eng->n_layer;
        kvc_cur = eng->kvc;
        prof.start();
        if (eng->ple_table.data && pos0 > 0 && !sync()) return -1;  // a queued state restore may set ple_hist
        for (int t = 0; t < T; ++t) {
            if (tokens[t] < 0 || tokens[t] >= eng->vocab_size) { qwen_set_error("token %d out of range", tokens[t]); return -1; }
            float* r = s.res + (size_t)t * HCD;  // every stream starts as the embedding
            if (qt_to_float(eng->token_embd.type, eng->token_embd.row(tokens[t]), r, E) != 0) {
                qwen_set_error("cannot decode token_embd type %d", eng->token_embd.type);
                return -1;
            }
            for (int c = 1; c < HC; ++c) memcpy(r + (size_t)c * E, r, sizeof(float) * E);
        }
        if (dbg_res_in) memcpy(s.res, dbg_res_in, sizeof(float) * (size_t)T * HCD);
        // The reset (a GPU memset) also clears ple_hist, which the CPU writes next: wait for it
        if (pos0 == 0 && (!check(rs.reset(), "recurrent state reset") || (eng->ple_table.data && !sync()))) return -1;
        if (eng->ple_table.data && !ple_rows(tokens, T, pos0)) return -1;
        prof.lap(P_EMBED);

        const bool per_layer = layer_cb || qwen_debug();
        for (int l = l0; l < l1; ++l) {
            const LayerWeights& L = eng->layers[l];
            const bool attn = L.kind == LAYER_FULL_ATTN;
            const ProfCat pc = attn ? P_ATTN : P_LIN;
            if (L.ple && !ple(l, L, T)) return -1;
            if ((l == l0 || L.ple) && !hc_norm(nullptr, L.hc_attn.norm, T)) return -1;
            if (!hc_mix(L.hc_attn, T, pc) || !(attn ? full_attention(L, pos0, T) : linear_attention(l, L, T)) ||
                !section(pc))
                return -1;
            const float* next = l + 1 < eng->n_layer ? eng->layers[l + 1].hc_attn.norm : eng->hc_head.norm;
            if (!hc_norm(s.proj, L.hc_ffn.norm, T) || !hc_mix(L.hc_ffn, T, P_FFN) || !ffn_core(L, T) ||
                !hc_norm(s.proj, next, T) || !section(P_FFN))
                return -1;
            if (per_layer) {
                if (!sync()) return -1;
                for (int t = 0; t < T; ++t) {
                    const float* xt = s.res + (size_t)t * HCD;
                    if (layer_cb) layer_cb(l, pos0 + t, xt, HCD, layer_cb_user);
                    if (qwen_debug()) {
                        double ss = 0.0;
                        for (int i = 0; i < HCD; ++i) ss += (double)xt[i] * xt[i];
                        fprintf(stderr, "[qwen] pos %d blk %2d %-6s rms(res)=%.5g res[0..3]=%.5g %.5g %.5g %.5g\n",
                                pos0 + t, l, attn ? "attn" : "linear", sqrt(ss / HCD), xt[0], xt[1], xt[2], xt[3]);
                    }
                }
                prof.start();
            }
        }
        if (dbg_res_in) return sync() ? 0 : -1;
        const bool track = eng->mtp && mtp_track(pos0, T);
        if (!hc_mix(eng->hc_head, T, P_HEAD) || lm_head(T, logits_out) != 0) return -1;
        return track && !mtp_process(tokens, T, pos0) ? -1 : 0;
    }

    // ------------------------------------------------------------ MTP draft head (qwen4exp nextn sidecar)
    //
    // The head's row at position p is (token p, the main model's residual streams at p - 1). Every main forward
    // keeps its streams (mtp_h: row 0 = position pos0 - 1, carried from the previous forward; rows 1 .. T = this
    // batch) and runs the head on its rows from mtp_pos on (no LM head), so the head's KV cache follows the
    // sequence; mtp_draft adds the row of the next, not yet decoded token and returns the head's argmax.
    inference_kv_cache_t* kvc_cur = nullptr;     // the KV cache full_attention uses (the model's or the head's)
    ShmRegion mtp_mem, mtp_pad_mem;
    float *mtp_h = nullptr, *mtp_carry = nullptr, *mtp_cat = nullptr, *mtp_logits = nullptr, *spec_logits = nullptr;
    float* mtp_chain = nullptr;                 // the head's output streams, the input of its next chained draft
    bool spec_mid = false;                      // a verify pass: keep the states after its first tokens in rs_mid
    int spec_n = 0;                             // tokens of the last verify pass (0: none)
    RecurrentState rs_mid[SPEC_MAX_DRAFTS];     // rs_mid[j]: the state after token j of a verify pass

    // The rollback points of layer l's state array `a` for a pass of T tokens (none outside a verify pass)
    deltanet_mid_states mids(std::vector<float*> RecurrentState::*a, int l, int T) const {
        deltanet_mid_states m = {{nullptr, nullptr, nullptr}};
        if (spec_mid)
            for (int j = 0; j + 1 < T && j < SPEC_MAX_DRAFTS; ++j) m.p[j] = (rs_mid[j].*a)[l];
        return m;
    }
    int mtp_carry_pos = -1;                     // position of the streams in mtp_carry (-1: none)
    int mtp_pos = 0;                            // the head's KV cache holds rows 0 .. mtp_pos - 1
    int mtp_h_pos0 = -1, mtp_h_n = 0;           // mtp_h: positions mtp_h_pos0 - 1 .. mtp_h_pos0 + mtp_h_n - 1
    const MtpHead* mtp_ready = nullptr;          // the head the buffers / padded copies were made for

    void mtp_free() {
        mtp_mem.free();
        mtp_pad_mem.free();
        if (mtp_ready) {
            hc_up_pad.erase(&mtp_ready->L.hc_attn);
            hc_up_pad.erase(&mtp_ready->L.hc_ffn);
            hc_up_pad.erase(&mtp_ready->hc_head);
        }
        mtp_ready = nullptr;
        mtp_h = mtp_carry = mtp_cat = mtp_logits = spec_logits = mtp_chain = nullptr;
        for (RecurrentState& r : rs_mid) r.free();
        spec_n = 0;
        mtp_carry_pos = -1;
        mtp_pos = 0;
        mtp_h_n = 0;
    }

    bool mtp_setup() {
        if (mtp_ready == eng->mtp) return true;
        mtp_free();
        const MtpHead* h = eng->mtp;
        const int E = eng->n_embd, HCD = eng->n_hc * E;
        const size_t b = (size_t)s.B;
        const size_t sizes[] = {(b + 1) * HCD, (size_t)HCD, b * HCD * 2, (size_t)eng->vocab_size,
                                (SPEC_MAX_DRAFTS + 1) * (size_t)eng->vocab_size, (size_t)HCD};
        float** const ptrs[] = {&mtp_h, &mtp_carry, &mtp_cat, &mtp_logits, &spec_logits, &mtp_chain};
        bool mids_ok = true;
        for (RecurrentState& r : rs_mid) mids_ok = mids_ok && (r.mem.base || !rs.mem.base || r.alloc(eng));
        if (!mtp_mem.alloc("/dev/shm/qwen_mtp.bin", sizes, ptrs, 6, hipExtHostRegisterCoarseGrained) || !mids_ok ||
            !pad_hc_up({&h->L.hc_attn, &h->L.hc_ffn, &h->hc_head}, mtp_pad_mem, "/dev/shm/qwen_mtp_hc_up_pad.bin")) {
            qwen_set_error("MTP head: buffers");
            mtp_free();
            return false;
        }
        mtp_ready = h;
        return true;
    }

    // Before the head of a main forward of T tokens at pos0: keep its streams after the last layer for the head
    bool mtp_track(int pos0, int T) {
        if (!mtp_setup()) return false;
        const int HCD = eng->n_hc * eng->n_embd;
        const size_t row = sizeof(float) * HCD;
        // row 0: the streams of position pos0 - 1 (zeros at the start, or when they are not known)
        if (pos0 > 0 && mtp_carry_pos == pos0 - 1) {
            if (!check(hipMemcpyAsync(mtp_h, mtp_carry, row, hipMemcpyDeviceToDevice, 0), "MTP streams")) return false;
        } else if (!check(hipMemsetAsync(mtp_h, 0, row, 0), "MTP streams")) {
            return false;
        }
        if (!check(hipMemcpyAsync(mtp_h + HCD, s.res, row * T, hipMemcpyDeviceToDevice, 0), "MTP streams") ||
            !check(hipMemcpyAsync(mtp_carry, s.res + (size_t)(T - 1) * HCD, row, hipMemcpyDeviceToDevice, 0),
                   "MTP streams"))
            return false;
        mtp_h_pos0 = pos0;
        mtp_h_n = T;
        mtp_carry_pos = pos0 + T - 1;
        if (mtp_pos > pos0) mtp_pos = std::max(pos0, std::min(mtp_pos, pos0 + T));  // rows from pos0 on change
        return true;
    }

    // After a main forward: the head's rows mtp_pos .. pos0 + T - 1 (all of them that this batch covers)
    bool mtp_process(const int* tokens, int T, int pos0) {
        const int p0 = std::max(mtp_pos, pos0), n = pos0 + T - p0;
        if (n <= 0) return true;
        // the sync: the next forward's CPU work writes buffers these kernels read
        if (!mtp_forward(tokens + (p0 - pos0), mtp_h + (size_t)(p0 - pos0) * eng->n_hc * eng->n_embd, n, p0, false) ||
            !sync())
            return false;
        mtp_pos = pos0 + T;
        return true;
    }

    // The head on T rows (tokens, streams of the positions before them) at positions pos0 ..; with logits, its LM
    // head on the last row into mtp_logits (and a sync). The rows must not exceed s.B.
    bool mtp_forward(const int* tokens, const float* h, int T, int pos0, bool logits) {
        const MtpHead& M = *eng->mtp;
        const int E = eng->n_embd, HC = eng->n_hc;
        const float eps = (float)eng->eps;
        if (T > s.B) { qwen_set_error("MTP head: %d rows (max %d)", T, s.B); return false; }
        if (!M.kvc) { qwen_set_error("MTP head: no KV cache"); return false; }
        if (pos0 + T > kv_cache_n_ctx(M.kvc)) { qwen_set_error("MTP head: position %d out of range", pos0 + T - 1); return false; }
        // e: RMSNorm(embedding) * enorm (rows decoded on the CPU into s.x; the GPU is done with s.x here)
        for (int t = 0; t < T; ++t)
            if (tokens[t] < 0 || tokens[t] >= eng->vocab_size ||
                qt_to_float(M.token_embd.type, M.token_embd.row(tokens[t]), s.x + (size_t)t * E, E) != 0) {
                qwen_set_error("MTP head: token %d", tokens[t]);
                return false;
            }
        kvc_cur = M.kvc;
        const bool ok =
            check(gpu_add_rmsnorm(s.x, nullptr, s.x, M.enorm, s.xn, E, T, eps, nullptr, 0), "MTP enorm") &&
            // h: per-stream RMSNorm * hnorm, then [e | h] per stream through eh_proj: the block's residual streams
            check(gpu_hc_norm(h, nullptr, nullptr, nullptr, M.hnorm, s.hxn, E, HC, T, eps, nullptr, 0), "MTP hnorm") &&
            check(gpu_mtp_concat(s.xn, s.hxn, mtp_cat, E, HC, T, 0), "MTP concat") &&
            matvec(M.eh_proj, mtp_cat, s.res, T * HC, P_HEAD) &&  // Q8_0: the float path
            // the block, as in forward_hc
            hc_norm(nullptr, M.L.hc_attn.norm, T) && hc_mix(M.L.hc_attn, T, P_HEAD) &&
            full_attention(M.L, pos0, T) && hc_norm(s.proj, M.L.hc_ffn.norm, T) && hc_mix(M.L.hc_ffn, T, P_HEAD) &&
            ffn_core(M.L, T) && hc_norm(s.proj, M.hc_head.norm, T) &&
            (!logits || (hc_mix(M.hc_head, T, P_HEAD) && lm_head_with(M.output, mtp_logits, T, nullptr, false) == 0));
        kvc_cur = eng->kvc;
        return ok;
    }

    // Speculative verify pass: tokens[0] at pos (the next token) and n - 1 drafts after it in one forward; the
    // logits after each (verify_logits) and the states after the first n - 1 kept for rollback()
    int forward_verify(const int* tokens, int n, int pos) {
        if (!eng->mtp || !mtp_setup()) { qwen_set_error("no MTP head"); return -1; }
        if (n < 2 || n > SPEC_MAX_DRAFTS + 1) { qwen_set_error("verify pass of %d tokens", n); return -1; }
        spec_mid = true;
        const int r = forward(tokens, n, pos, nullptr);
        spec_mid = false;
        if (r != 0) return r;
        spec_pos = pos;
        spec_n = n;
        return 0;
    }
    int spec_pos = -1;

    // The logits after token i of the last verify pass (until the next forward)
    const float* verify_logits(int i) const {
        if (spec_n < 2 || i < 0 || i >= spec_n) return nullptr;
        return i == spec_n - 1 ? s.logits : spec_logits + (size_t)i * eng->vocab_size;
    }

    // After forward_verify at pos accepted only its first `keep` tokens (1 <= keep < n): back to the state after the
    // token at pos + keep - 1 (the KV cache needs nothing: the later rows are overwritten by the next tokens)
    int rollback(int keep) {
        if (spec_pos < 0 || keep < 1 || keep >= spec_n) { qwen_set_error("rollback without a verify pass"); return -1; }
        const RecurrentState& mid = rs_mid[keep - 1];
        if (rs.mem.base && !check(hipMemcpyAsync(rs.mem.base, mid.mem.base, rs.mem.total, hipMemcpyDeviceToDevice, 0),
                                  "rollback"))
            return -1;
        const int HCD = eng->n_hc * eng->n_embd;
        if (!check(hipMemcpyAsync(mtp_carry, mtp_h + (size_t)keep * HCD, sizeof(float) * HCD, hipMemcpyDeviceToDevice,
                                  0),
                   "rollback"))
            return -1;
        mtp_carry_pos = spec_pos + keep - 1;
        mtp_pos = std::min(mtp_pos, spec_pos + keep);
        mtp_h_n = keep;
        spec_pos = -1;
        spec_n = 0;
        return sync() ? 0 : -1;
    }

    // The head's drafts for positions pos + 1 .. pos + k, given the token at pos (not decoded yet): its argmax,
    // then (k > 1) the head again on that draft with its own output streams standing in for the model's, and so
    // on. Returns how many (0 when there is no draft: no head, or the streams of pos - 1 are not known).
    int mtp_drafts(int token, int pos, int* drafts, int k) {
        if (!eng->mtp || !mtp_setup() || mtp_carry_pos != pos - 1 || pos < 1 || k < 1) return 0;
        k = std::min(k, SPEC_MAX_DRAFTS);
        // rows before pos the head has not seen (after a jump in the sequence) stay as they are: older rows only
        // make its drafts worse, never the output
        mtp_pos = pos;
        const float* h = mtp_carry;
        int tok = token, n = 0;
        for (; n < k; ++n) {
            if (pos + n + 2 > kv_cache_n_ctx(eng->kvc) || !mtp_forward(&tok, h, 1, pos + n, true)) break;
            int best = 0;
            for (int v = 1; v < eng->vocab_size; ++v)
                if (mtp_logits[v] > mtp_logits[best]) best = v;
            drafts[n] = best;
            if (n + 1 < k &&
                !check(hipMemcpyAsync(mtp_chain, s.res, sizeof(float) * eng->n_hc * eng->n_embd,
                                      hipMemcpyDeviceToDevice, 0),
                       "MTP chain"))
                break;
            h = mtp_chain;
            tok = best;
        }
        // only row pos has the model's streams: the head recomputes the rows after it once they are decoded
        mtp_pos = pos + 1;
        return n;
    }

    // The FFN of a hyper-connection block, from s.xn to s.proj: MoE, or a dense SwiGLU
    bool ffn_core(const LayerWeights& L, int T) {
        if (L.moe) return moe_core(L, T);
        const int FF = eng->n_ff;
        const qmatvec_q8* qf = q8_for(FF, T, true);
        if (!matvec_pair(L.W_gate, L.W_up, s.xn, s.ffg, s.ffu, T, P_FFN) ||
            !check(gpu_swiglu(s.ffg, s.ffu, s.act, FF * T, qf, 0), "swiglu"))
            return false;
        mark(s.act, FF, T, qf);
        return matvec(L.W_down, s.act, s.proj, T, P_FFN);
    }

    // Use (or stop using) the Q8_0 copy of the LM head; builds it the first time. Only for F32 / F16 / BF16
    // heads: returns 1 (nothing done) for a head that is already quantized, -1 on error.
    int set_lm_head_q8(bool on) {
        const QuantWeight& H = eng->output;
        if (!on) { head_q8 = false; return 0; }
        if (!qmatvec_q8_0s_source_supported(H.type, H.K)) return 1;
        if (!head_q) {
            float *q = nullptr, *d = nullptr;
            const size_t sizes[] = {(size_t)H.N * H.K / 4, (size_t)H.N * (H.K / 32) / 2 + 1};
            float** const ptrs[] = {&q, &d};
            if (!head_mem.alloc("/dev/shm/qwen_lm_head_q8.bin", sizes, ptrs, 2, hipExtHostRegisterCoarseGrained)) {
                qwen_set_error("LM head Q8_0 copy: allocation failed");
                return -1;
            }
            head_q = (int8_t*)q;
            head_d = (uint16_t*)d;
            if (!check(qmatvec_quantize_rows_q8_0(H.type, H.data, H.row_bytes, H.N, H.K, head_q, head_d, 0),
                       "LM head Q8_0 copy") ||
                !sync()) {
                head_mem.free();
                head_q = nullptr;
                head_d = nullptr;
                return -1;
            }
        }
        head_q8 = true;
        return 0;
    }

    // A whole prompt (or any n tokens of the sequence) from position pos0, in micro-batches of s.B tokens.
    // Leaves the logits of the last token in s.logits (and logits_out, if given).
    int prefill(const int* tokens, int n, int pos0, float* logits_out) {
        if (n <= 0) { qwen_set_error("empty prefill"); return -1; }
        for (int done = 0; done < n;) {
            const int T = std::min(s.B, n - done);
            if (forward(tokens + done, T, pos0 + done, done + T == n ? logits_out : nullptr) != 0) return -1;
            done += T;
        }
        return 0;
    }
};

extern "C" {

const char* qwen_model_last_error(void) { return g_qwen_error; }

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

int qwen_model_set_lm_head_q8(qwen_model_t* model, int enable) {
    return model ? model->set_lm_head_q8(enable != 0) : -1;
}

int qwen_model_is_recurrent(const qwen_model_t* model) {
    return model && model->rs.mem.base ? 1 : 0;
}

int qwen_model_ring_reuse_ok(const qwen_model_t* model, int start) {
    if (!model || model->eng->swa_ring <= 0 || start <= 0) return 1;
    return std::max(0, start - model->eng->swa_window) >= model->ring_hi - model->eng->swa_ring ? 1 : 0;
}

void* qwen_model_state_memory(qwen_model_t* model, size_t* bytes) {
    if (bytes) *bytes = model && model->rs.mem.base ? model->rs.mem.total : 0;
    return model ? model->rs.mem.base : nullptr;
}

int qwen_model_sync(qwen_model_t* model) { return model && model->sync() ? 0 : -1; }

int qwen_model_state_save(qwen_model_t* model, int slot) {
    return model && model->check(model->rs.save(slot), "recurrent state checkpoint") ? 0 : -1;
}

int qwen_model_state_restore(qwen_model_t* model, int slot) {
    return model && model->check(model->rs.restore(slot), "recurrent state restore") ? 0 : -1;
}

void qwen_model_set_layer_callback(qwen_model_t* model, qwen_layer_cb cb, void* user) {
    if (!model) return;
    model->layer_cb = cb;
    model->layer_cb_user = user;
}

int qwen_model_forward(qwen_model_t* model, int token_id, int pos, float* logits) {
    if (!model) return -1;
    return model->forward(&token_id, 1, pos, logits);
}

int qwen_model_prefill(qwen_model_t* model, const int* tokens, int n, int pos0, float* logits) {
    if (!model || !tokens) return -1;
    return model->prefill(tokens, n, pos0, logits);
}

int qwen_model_run_layers(qwen_model_t* model, const int* tokens, int n, const float* res_in, int l0, int l1) {
    if (!model || !tokens || !res_in || (model->eng->n_hc <= 0 && !model->eng->gemma4) || l0 < 0 ||
        l1 > model->eng->n_layer || l0 >= l1 ||
        n > model->s.B) {
        qwen_set_error("qwen_model_run_layers: bad arguments");
        return -1;
    }
    model->dbg_res_in = res_in;
    model->dbg_l0 = l0;
    model->dbg_l1 = l1;
    const int r = model->forward(tokens, n, 0, nullptr);
    model->dbg_res_in = nullptr;
    return r;
}

void qwen_model_mtp_reset(qwen_model_t* model) {
    if (model) model->mtp_free();
}

int qwen_model_mtp_draft(qwen_model_t* model, int token, int pos) {
    int d = -1;
    return model && model->mtp_drafts(token, pos, &d, 1) == 1 ? d : -1;
}

int qwen_model_mtp_drafts(qwen_model_t* model, int token, int pos, int* drafts, int k) {
    return model && drafts ? model->mtp_drafts(token, pos, drafts, k) : 0;
}

int qwen_model_forward_verify(qwen_model_t* model, const int* tokens, int n, int pos) {
    return model && tokens ? model->forward_verify(tokens, n, pos) : -1;
}

const float* qwen_model_verify_logits(const qwen_model_t* model, int i) {
    return model ? model->verify_logits(i) : nullptr;
}

int qwen_model_rollback(qwen_model_t* model, int keep) {
    return model ? model->rollback(keep) : -1;
}

int qwen_model_max_batch(const qwen_model_t* model) {
    return model ? model->s.B : 0;
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
