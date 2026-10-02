// qwen_model.cpp -- Phase 4: Qwen 3 dense transformer forward pass host orchestration (minimal skeleton).
// Loads all kernels, allocates unified pinned buffers, runs token-by-token forward pass.

#include "qwen_model.h"
#include "inference.h"
#include "gguf_mmap.h"
#include "kv_cache.h"
#include "fp4_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// Internal inference_model definition (mirrors inference.cpp)
struct inference_model {
    gguf_model_t* gguf = nullptr;
    std::string model_name;
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int n_ctx = 0;
    double eps = 1e-5;
    std::vector<__half> embed_table;
    void* d_embed_table = nullptr;
    struct LayerWeights {
        struct QuantWeight {
            void* d_packed = nullptr;
            void* d_scales = nullptr;
        } Wq, Wk, Wv, Wo, Wgate, Wup, Wdown;
        void* d_attn_ln = nullptr;
        void* d_mlp_ln = nullptr;
    };
    std::vector<LayerWeights> layers;
    struct QuantWeight {
        void* d_packed = nullptr;
        void* d_scales = nullptr;
    } lm_head;
    void* d_lm_head_packed = nullptr;
    void* d_lm_head_scales = nullptr;
    void* d_final_ln = nullptr;
    inference_kv_cache_t* kvc = nullptr;
};

using namespace rocmfp4;

// ==================== Kernel Module Handles (global) ====================
static hipModule_t mod_gemm_act_fp4 = nullptr, mod_swiglu = nullptr, mod_fp16_gemm = nullptr;
static hipModule_t mod_rope = nullptr, mod_softmax = nullptr, mod_layernorm = nullptr;
static hipModule_t mod_kv_store = nullptr, mod_embedding = nullptr;

static hipFunction_t fn_gemm_act_fp4 = nullptr, fn_swiglu = nullptr, fn_fp16_gemm = nullptr;
static hipFunction_t fn_rope = nullptr, fn_softmax = nullptr, fn_layernorm = nullptr;
static hipFunction_t fn_kv_store = nullptr, fn_embedding = nullptr;

static bool load_module(const char* co, const char* sym, hipModule_t& m, hipFunction_t& f) {
    int fd = open(co, O_RDONLY); if (fd < 0) return false;
    struct stat st; fstat(fd, &st);
    void* data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    bool ok = (hipModuleLoadData(&m, data) == hipSuccess) &&
              (hipModuleGetFunction(&f, m, sym) == hipSuccess);
    munmap(data, st.st_size); close(fd); return ok;
}

static bool load_all_kernels() {
    hipInit(0);
    return load_module("gemm_act_fp4.co", "gemm_act_fp4", mod_gemm_act_fp4, fn_gemm_act_fp4) &&
           load_module("swiglu.co", "swiglu_fp4", mod_swiglu, fn_swiglu) &&
           load_module("fp16_gemm.co", "fp16_gemm", mod_fp16_gemm, fn_fp16_gemm) &&
           load_module("rope_embedding.co", "rope_rotate_half", mod_rope, fn_rope) &&
           load_module("softmax_masked.co", "softmax_masked", mod_softmax, fn_softmax) &&
           load_module("layernorm.co", "rmsnorm", mod_layernorm, fn_layernorm) &&
           load_module("kv_cache_store.co", "kv_store", mod_kv_store, fn_kv_store) &&
           load_module("embedding_lookup.co", "embedding_lookup", mod_embedding, fn_embedding);
}

static void unload_all_kernels() {
    if (mod_gemm_act_fp4) hipModuleUnload(mod_gemm_act_fp4);
    if (mod_swiglu) hipModuleUnload(mod_swiglu);
    if (mod_fp16_gemm) hipModuleUnload(mod_fp16_gemm);
    if (mod_rope) hipModuleUnload(mod_rope);
    if (mod_softmax) hipModuleUnload(mod_softmax);
    if (mod_layernorm) hipModuleUnload(mod_layernorm);
    if (mod_kv_store) hipModuleUnload(mod_kv_store);
    if (mod_embedding) hipModuleUnload(mod_embedding);
}

// ==================== Per-Layer Scratch Buffers ====================
struct LayerScratch {
    __half* h = nullptr; void* d_h = nullptr;
    float* q_fp32 = nullptr; void* d_q = nullptr;
    float* k_fp32 = nullptr; void* d_k = nullptr;
    float* v_fp32 = nullptr; void* d_v = nullptr;
    __half* qh = nullptr; void* d_qh = nullptr;
    __half* kh = nullptr; void* d_kh = nullptr;
    float* rope_cos = nullptr; void* d_rc = nullptr;
    float* rope_sin = nullptr; void* d_rs = nullptr;
    __half* rks = nullptr; void* d_rks = nullptr;
    float* scores = nullptr; void* d_sc = nullptr;
    __half* probs_fp16 = nullptr; void* d_pf = nullptr;
    float* oh = nullptr; void* d_oh = nullptr;
    __half* oin = nullptr; void* d_oin = nullptr;
    __half* h_mlp = nullptr; void* d_hmlp = nullptr;
    float* gate_fp32 = nullptr; void* d_gate = nullptr;
    float* up_fp32 = nullptr; void* d_up = nullptr;
    float* down_fp32 = nullptr; void* d_down = nullptr;
    float* vexpB = nullptr; void* d_vb2 = nullptr;

    void* base = nullptr; size_t total = 0; int fd = -1;

    bool alloc(int E, int HD, int HQ, int HKV, int CTX, int DQ, int DKV) {
        const int hd2 = HD / 2;
        size_t layout[] = {
            (size_t)E * sizeof(__half),                    // h
            (size_t)DQ * sizeof(float),                    // q_fp32
            (size_t)DKV * sizeof(float),                   // k_fp32
            (size_t)DKV * sizeof(float),                   // v_fp32
            (size_t)HQ * HD * sizeof(__half),              // qh
            (size_t)HKV * HD * sizeof(__half),             // kh
            (size_t)HQ * hd2 * sizeof(float),              // rope_cos
            (size_t)HQ * hd2 * sizeof(float),              // rope_sin
            (size_t)DKV * sizeof(__half),                  // rks
            (size_t)CTX * sizeof(float),                   // scores
            (size_t)CTX * sizeof(__half),                  // probs_fp16
            (size_t)HQ * HD * sizeof(float),               // oh
            (size_t)DQ * sizeof(__half),                   // oin
            (size_t)E * sizeof(__half),                    // h_mlp
            (size_t)DQ * sizeof(float),                    // gate_fp32
            (size_t)DKV * sizeof(float),                   // up_fp32
            (size_t)E * sizeof(float),                     // down_fp32
            (size_t)CTX * HD * sizeof(__half),             // vexpB
        };
        size_t total = 0;
        for (size_t sz : layout) { total = (total + 127) & ~127; total += sz; }
        this->total = total;

        fd = open("/dev/shm/qwen_layer_scratch.bin", O_RDWR | O_CREAT, 0666);
        if (fd < 0) return false;
        ftruncate(fd, total);
        base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) return false;

        size_t off = 0;
        auto get = [&](size_t sz, void** hptr, void** dptr) {
            off = (off + 127) & ~127;
            *hptr = (char*)base + off;
            hipHostGetDevicePointer(dptr, *hptr, 0);
            off += sz;
        };
        get(layout[0], (void**)&h, &d_h);
        get(layout[1], (void**)&q_fp32, &d_q);
        get(layout[2], (void**)&k_fp32, &d_k);
        get(layout[3], (void**)&v_fp32, &d_v);
        get(layout[4], (void**)&qh, &d_qh);
        get(layout[5], (void**)&kh, &d_kh);
        get(layout[6], (void**)&rope_cos, &d_rc);
        get(layout[7], (void**)&rope_sin, &d_rs);
        get(layout[8], (void**)&rks, &d_rks);
        get(layout[9], (void**)&scores, &d_sc);
        get(layout[10], (void**)&probs_fp16, &d_pf);
        get(layout[11], (void**)&oh, &d_oh);
        get(layout[12], (void**)&oin, &d_oin);
        get(layout[13], (void**)&h_mlp, &d_hmlp);
        get(layout[14], (void**)&gate_fp32, &d_gate);
        get(layout[15], (void**)&up_fp32, &d_up);
        get(layout[16], (void**)&down_fp32, &d_down);
        get(layout[17], (void**)&vexpB, &d_vb2);

        hipHostRegister(base, total, hipHostRegisterDefault);
        return true;
    }
    void free() {
        if (base) { hipHostUnregister(base); munmap(base, total); }
        if (fd >= 0) close(fd);
    }
};

// ==================== Qwen Model ====================
struct QwenModel {
    inference_model* eng_model = nullptr;
    std::vector<LayerScratch> layer_scratch;
    int n_layer = 28, n_embd = 4096, n_head = 32, n_head_kv = 8, head_dim = 128;
    int vocab_size = 151936, n_ctx = 4096;

    ~QwenModel() {
        for (auto& s : layer_scratch) s.free();
        unload_all_kernels();
    }

    bool init(inference_model* model) {
        eng_model = model;
        if (!load_all_kernels()) return false;

        const int E = n_embd, HD = head_dim, HQ = n_head, HKV = n_head_kv, CTX = n_ctx;
        const int DQ = HQ * HD, DKV = HKV * HD;

        layer_scratch.resize(n_layer);
        for (int i = 0; i < n_layer; ++i) {
            if (!layer_scratch[i].alloc(E, HD, HQ, HKV, CTX, DQ, DKV)) return false;
        }
        return true;
    }

    void build_rope_tables(int pos, LayerScratch& s) {
        const int HD = head_dim, HQ = n_head, hd2 = HD / 2;
        const double log_theta = log(10000.0);
        for (int h = 0; h < HQ; ++h) {
            float* cr = s.rope_cos + h * hd2;
            float* sr = s.rope_sin + h * hd2;
            for (int d = 0; d < hd2; ++d) {
                double freq = exp(-2.0 * d / (double)HD * log_theta);
                cr[d] = cosf((float)(pos * freq));
                sr[d] = sinf((float)(pos * freq));
            }
        }
    }

    // Skeleton forward - minimal to compile
    int forward(int token_id, int pos, float* logits_out) {
        (void)token_id; (void)pos; (void)logits_out;
        // TODO: Implement full forward pass
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
    return model->forward(token_id, pos, logits);
}
}