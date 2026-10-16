// qwen_model.cpp -- Qwen3.5 single-token forward pass (Phase 20 Step 3c).
//
// Every projection (attention Q/K/V/O, FFN gate/up/down, LM head) is one qmatvec_launch on the raw GGUF
// blocks, read in place from the pinned file mapping. Activations live in one /dev/shm scratch buffer,
// pinned once, so the GPU reads and writes them without copies. The small per-token vector ops (norms,
// RoPE, single-query attention, SwiGLU) run on the CPU via qwen_ops for now.
//
// Status: full-attention layers follow the llama.cpp qwen35 graph (gated Q, q/k RMSNorm, partial NeoX
// RoPE, GQA, sigmoid output gate) but are not yet verified end to end (Step 4). Gated DeltaNet layers
// are identity (Step 5), so the logits are not meaningful yet.

#include "qwen_model.h"
#include "inference.h"
#include "inference_internal.h"
#include "kv_cache.h"
#include "qmatvec.h"
#include "quant_types.h"
#include "qwen_ops.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

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

// Per-token activations: one /dev/shm file, mmap(MAP_SHARED), hipHostRegister'ed once (project core pattern)
struct Scratch {
    float* x = nullptr;       // residual stream [E]
    float* xn = nullptr;      // normed input to the projections [E]
    float* qfull = nullptr;   // attn_q output, [Q | gate] per head [2 * HQ * HD]
    float* q = nullptr;       // Q after norm + RoPE [HQ * HD]
    float* gate = nullptr;    // attention output gate [HQ * HD]
    float* k = nullptr;       // [HKV * HD]
    float* v = nullptr;       // [HKV * HD]
    float* attn = nullptr;    // gated attention output, input to attn_output [HQ * HD]
    float* proj = nullptr;    // attn_output / ffn_down output [E]
    float* ffg = nullptr;     // ffn_gate output [FF]
    float* ffu = nullptr;     // ffn_up output [FF]
    float* act = nullptr;     // SiLU(gate) * up, input to ffn_down [FF]
    float* logits = nullptr;  // [vocab]

    void* base = nullptr;
    size_t total = 0;

    bool alloc(int E, int HQ, int HKV, int HD, int FF, int V) {
        const size_t sizes[] = {(size_t)E, (size_t)E, (size_t)2 * HQ * HD, (size_t)HQ * HD, (size_t)HQ * HD,
                                (size_t)HKV * HD, (size_t)HKV * HD, (size_t)HQ * HD, (size_t)E,
                                (size_t)FF, (size_t)FF, (size_t)FF, (size_t)V};
        float** ptrs[] = {&x, &xn, &qfull, &q, &gate, &k, &v, &attn, &proj, &ffg, &ffu, &act, &logits};
        size_t off[sizeof(sizes) / sizeof(sizes[0])];
        total = 0;
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
            off[i] = total;
            total += (sizes[i] * sizeof(float) + 255) & ~(size_t)255;
        }
        const char* path = "/dev/shm/qwen_scratch.bin";
        int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) return false;
        const bool sized = ftruncate(fd, (off_t)total) == 0;
        void* p = sized ? mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
        close(fd);
        unlink(path);
        if (p == MAP_FAILED) return false;
        if (hipHostRegister(p, total, hipHostRegisterMapped) != hipSuccess) {
            munmap(p, total);
            return false;
        }
        base = p;
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) *ptrs[i] = (float*)((char*)p + off[i]);
        return true;
    }

    void free() {
        if (!base) return;
        (void)hipHostUnregister(base);
        munmap(base, total);
        base = nullptr;
    }
};

struct QwenModel {
    inference_model* eng = nullptr;
    Scratch s;

    ~QwenModel() {
        s.free();
        qmatvec_free();
    }

    bool init(inference_model* model) {
        eng = model;
        if (qmatvec_init("matvec_q.co") != 0) {
            qwen_set_error("cannot load matvec_q.co (run from build/co)");
            return false;
        }
        if (!s.alloc(eng->n_embd, eng->n_head, eng->n_head_kv, eng->head_dim, eng->n_ff, eng->vocab_size)) {
            qwen_set_error("scratch allocation failed");
            return false;
        }
        return true;
    }

    // y = W x on the GPU. x and y must be in pinned memory (the scratch buffer).
    bool matvec(const QuantWeight& W, const float* x, float* y) {
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

    // Gated full attention for one token at position pos; adds its output to the residual
    bool full_attention(int l, const LayerWeights& L, int pos) {
        const int E = eng->n_embd, HD = eng->head_dim, HQ = eng->n_head, HKV = eng->n_head_kv;
        const int DKV = HKV * HD;
        const float eps = (float)eng->eps;

        qwen_rmsnorm(s.x, L.attn_norm, s.xn, E, eps);
        if (!matvec(L.Wq, s.xn, s.qfull) || !matvec(L.Wk, s.xn, s.k) || !matvec(L.Wv, s.xn, s.v) || !sync())
            return false;

        // attn_q rows are [Q (HD) | gate (HD)] per head
        for (int h = 0; h < HQ; ++h) {
            float* q = s.q + (size_t)h * HD;
            memcpy(q, s.qfull + (size_t)h * 2 * HD, HD * sizeof(float));
            memcpy(s.gate + (size_t)h * HD, s.qfull + (size_t)h * 2 * HD + HD, HD * sizeof(float));
            qwen_rmsnorm(q, L.q_norm, q, HD, eps);
            qwen_rope_neox(q, eng->n_rot, pos, eng->rope_freq_base);
        }
        for (int g = 0; g < HKV; ++g) {
            float* k = s.k + (size_t)g * HD;
            qwen_rmsnorm(k, L.k_norm, k, HD, eps);
            qwen_rope_neox(k, eng->n_rot, pos, eng->rope_freq_base);
        }

        __half* krow = kv_cache_k_row(eng->kvc, l, pos);
        __half* vrow = kv_cache_v_row(eng->kvc, l, pos);
        if (!krow || !vrow) { qwen_set_error("KV cache row %d of layer %d out of range", pos, l); return false; }
        for (int i = 0; i < DKV; ++i) {
            krow[i] = __float2half(s.k[i]);
            vrow[i] = __float2half(s.v[i]);
        }

        qwen_attention(s.q, kv_cache_k_row(eng->kvc, l, 0), kv_cache_v_row(eng->kvc, l, 0), DKV, pos + 1,
                       HQ, HKV, HD, 1.0f / sqrtf((float)HD), s.attn);
        for (int i = 0; i < HQ * HD; ++i) s.attn[i] *= qwen_sigmoid(s.gate[i]);

        if (!matvec(L.Wo, s.attn, s.proj) || !sync()) return false;
        for (int i = 0; i < E; ++i) s.x[i] += s.proj[i];
        return true;
    }

    // SwiGLU FFN; adds its output to the residual
    bool ffn(const LayerWeights& L) {
        const int E = eng->n_embd, FF = eng->n_ff;
        qwen_rmsnorm(s.x, L.ffn_norm, s.xn, E, (float)eng->eps);
        if (!matvec(L.W_gate, s.xn, s.ffg) || !matvec(L.W_up, s.xn, s.ffu) || !sync()) return false;
        for (int i = 0; i < FF; ++i) s.act[i] = qwen_silu(s.ffg[i]) * s.ffu[i];
        if (!matvec(L.W_down, s.act, s.proj) || !sync()) return false;
        for (int i = 0; i < E; ++i) s.x[i] += s.proj[i];
        return true;
    }

    int forward(int token_id, int pos, float* logits_out) {
        const int E = eng->n_embd;
        if (token_id < 0 || token_id >= eng->vocab_size) { qwen_set_error("token %d out of range", token_id); return -1; }
        if (!eng->kvc || pos < 0 || pos >= kv_cache_n_ctx(eng->kvc)) { qwen_set_error("position %d out of range", pos); return -1; }

        // Embedding: decode one row of token_embd on the CPU
        if (qt_to_float(eng->token_embd.type, eng->token_embd.row(token_id), s.x, E) != 0) {
            qwen_set_error("cannot decode token_embd type %d", eng->token_embd.type);
            return -1;
        }

        for (int l = 0; l < eng->n_layer; ++l) {
            const LayerWeights& L = eng->layers[l];
            // TODO(Phase 20 Step 5): Gated DeltaNet token mixer; linear layers are identity until then
            if (L.kind == LAYER_FULL_ATTN && !full_attention(l, L, pos)) return -1;
            if (!ffn(L)) return -1;
            if (qwen_debug()) {
                double ss = 0.0;
                for (int i = 0; i < E; ++i) ss += (double)s.x[i] * s.x[i];
                fprintf(stderr, "[qwen] pos %d blk %2d %-6s rms(x)=%.5g x[0..3]=%.5g %.5g %.5g %.5g\n", pos, l,
                        L.kind == LAYER_FULL_ATTN ? "attn" : "linear", sqrt(ss / E), s.x[0], s.x[1], s.x[2], s.x[3]);
            }
        }

        qwen_rmsnorm(s.x, eng->output_norm, s.xn, E, (float)eng->eps);
        if (!matvec(eng->output, s.xn, s.logits) || !sync()) return -1;
        memcpy(logits_out, s.logits, (size_t)eng->vocab_size * sizeof(float));
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

int qwen_model_forward(qwen_model_t* model, int token_id, int pos, float* logits) {
    if (!model || !logits) return -1;
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
