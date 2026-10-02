// inference.cpp -- Phase 4: C API implementation for zero-copy ROCmFP4 inference engine.
// Thin FFI wrappers around the C++ HIP implementation. All state in inference_model_t.

#include "inference.h"
#include "gguf_mmap.h"
#include "fp4_common.h"
#include "kv_cache.h"
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
#include <cstdarg>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

using namespace rocmfp4;

// Thread-local error buffer
static thread_local char g_last_error[1024] = {0};

static void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
}

static void clear_error() { g_last_error[0] = 0; }

// Forward declarations for kernel modules (loaded at runtime via hipModuleLoadData)
static hipModule_t g_mod_gemm_act_fp4 = nullptr;
static hipFunction_t g_fn_gemm_act_fp4 = nullptr;
static hipModule_t g_mod_swiglu = nullptr;
static hipFunction_t g_fn_swiglu = nullptr;
static hipModule_t g_mod_embedding = nullptr;
static hipFunction_t g_fn_embedding = nullptr;
static hipModule_t g_mod_rope = nullptr;
static hipFunction_t g_fn_rope = nullptr;
static hipModule_t g_mod_softmax = nullptr;
static hipFunction_t g_fn_softmax = nullptr;
static hipModule_t g_mod_fp16_gemm = nullptr;
static hipFunction_t g_fn_fp16_gemm = nullptr;
static hipModule_t g_mod_kv_store = nullptr;
static hipFunction_t g_fn_kv_store = nullptr;
static hipModule_t g_mod_layernorm = nullptr;
static hipFunction_t g_fn_layernorm = nullptr;

static bool load_module(const char* co_path, const char* sym, hipModule_t& mod, hipFunction_t& fn) {
    int fd = open(co_path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st; fstat(fd, &st);
    void* data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    bool ok = (hipModuleLoadData(&mod, data) == hipSuccess) &&
              (hipModuleGetFunction(&fn, mod, sym) == hipSuccess);
    munmap(data, st.st_size); close(fd);
    return ok;
}

static void ensure_kernels_loaded() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        load_module("gemm_act_fp4.co", "gemm_act_fp4", g_mod_gemm_act_fp4, g_fn_gemm_act_fp4);
        load_module("swiglu.co", "swiglu_fp4", g_mod_swiglu, g_fn_swiglu);
        load_module("embedding_lookup.co", "embedding_lookup", g_mod_embedding, g_fn_embedding);
        load_module("rope_embedding.co", "rope_rotate_half", g_mod_rope, g_fn_rope);
        load_module("softmax_masked.co", "softmax_masked", g_mod_softmax, g_fn_softmax);
        load_module("fp16_gemm.co", "fp16_gemm", g_mod_fp16_gemm, g_fn_fp16_gemm);
        load_module("kv_cache_store.co", "kv_store", g_mod_kv_store, g_fn_kv_store);
        load_module("layernorm.co", "rmsnorm", g_mod_layernorm, g_fn_layernorm);
        hipInit(0);
    });
}

// Quantized weight matrix for a single projection (ROCmFP4 Codebook10)
struct QuantWeight {
    int N, K;                                  // output_dim x input_dim (K multiple of 16)
    std::vector<unsigned char> packed;         // N x K/2 bytes
    std::vector<unsigned char> scales;         // N x K/16 bytes
    std::vector<float> true_v;                 // N x K effective values (for CPU ref)
    void* d_packed = nullptr;
    void* d_scales = nullptr;
};

// Per-layer weights (attention + MLP)
struct LayerWeights {
    // Attention: Q, K, V, O projections
    QuantWeight Wq, Wk, Wv, Wo;
    // MLP: gate, up, down (SwiGLU)
    QuantWeight W_gate, W_up, W_down;
    // LayerNorm scales
    std::vector<__half> attn_ln_scale;
    std::vector<__half> mlp_ln_scale;
    // Device pointers for norms
    void* d_attn_ln = nullptr;
    void* d_mlp_ln = nullptr;
};

struct inference_model {
    gguf_model_t* gguf = nullptr;              // GGUF file mapping
    std::string model_name;
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int n_ctx = 0;

    // Embedding table
    std::vector<__half> embed_table;           // [vocab_size x n_embd] FP16
    void* d_embed_table = nullptr;

    // Output projection (LM head) - Q8_0 quantized for dynamic range
    // For simplicity in Phase 4, we'll use FP4 too; Q8_0 comes in Phase 5
    QuantWeight lm_head;
    void* d_lm_head_packed = nullptr;
    void* d_lm_head_scales = nullptr;

    // Final LayerNorm
    std::vector<__half> final_ln_scale;
    void* d_final_ln = nullptr;

    // Per-layer weights
    std::vector<LayerWeights> layers;

    // KV cache manager
    inference_kv_cache_t* kvc = nullptr;

    // Scratch buffers (allocated per forward pass size)
    void* scratch_base = nullptr;
    size_t scratch_bytes = 0;

    // Tokenizer (simplified: vocab from GGUF, no BPE for Phase 4)
    std::vector<std::string> vocab_tokens;
    std::unordered_map<std::string, int> vocab_map;
};

static bool load_quant_weight(gguf_model_t* gguf, const char* prefix, QuantWeight& w) {
    // Tensors named like "blk.0.attn_q.weight", "blk.0.attn_q.scales", etc.
    std::string p_name = std::string(prefix) + ".weight";
    std::string s_name = std::string(prefix) + ".scales";

    size_t w_sz, s_sz;
    const void* w_ptr = gguf_get_tensor_ptr(gguf, p_name.c_str(), &w_sz);
    const void* s_ptr = gguf_get_tensor_ptr(gguf, s_name.c_str(), &s_sz);
    if (!w_ptr || !s_ptr) return false;

    // Parse shape from GGUF metadata (simplified: assume known layout)
    // Real implementation would read shape from tensor_info
    w.N = w_sz / (w.K / 2);  // placeholder

    w.packed.assign((unsigned char*)w_ptr, (unsigned char*)w_ptr + w_sz);
    w.scales.assign((unsigned char*)s_ptr, (unsigned char*)s_ptr + s_sz);
    w.true_v.resize((size_t)w.N * w.K);
    // Reconstruct true_v for CPU reference (dequant)
    const size_t strips = w.K / 16;
    for (int n = 0; n < w.N; ++n) {
        for (size_t s = 0; s < strips; ++s) {
            float sc = ue4m3_to_float(w.scales[(size_t)n * strips + s]);
            for (int i = 0; i < 16; ++i) {
                int k = (int)s * 16 + i;
                uint8_t wb = w.packed[(size_t)n * (w.K / 2) + (k >> 1)];
                uint8_t nib = (k & 1) ? (wb >> 4) : (wb & 0x0F);
                float code = dequant_code(nib);
                w.true_v[(size_t)n * w.K + k] = code * sc;
            }
        }
    }
    return true;
}

static bool pin_model_memory(inference_model* m) {
    // Pin embedding table
    if (!m->embed_table.empty()) {
        hipError_t e = hipHostRegister(m->embed_table.data(),
                                       m->embed_table.size() * sizeof(__half),
                                       hipHostRegisterDefault);
        if (e != hipSuccess) return false;
        hipHostGetDevicePointer(&m->d_embed_table, m->embed_table.data(), 0);
    }
    // Pin per-layer weights
    for (auto& L : m->layers) {
        auto pin = [](QuantWeight& w) -> bool {
            if (w.packed.empty()) return false;
            if (hipHostRegister(w.packed.data(), w.packed.size(), hipHostRegisterDefault) != hipSuccess) return false;
            if (hipHostRegister(w.scales.data(), w.scales.size(), hipHostRegisterDefault) != hipSuccess) return false;
            hipHostGetDevicePointer(&w.d_packed, w.packed.data(), 0);
            hipHostGetDevicePointer(&w.d_scales, w.scales.data(), 0);
            return true;
        };
        if (!pin(L.Wq) || !pin(L.Wk) || !pin(L.Wv) || !pin(L.Wo) ||
            !pin(L.W_gate) || !pin(L.W_up) || !pin(L.W_down)) return false;
        if (!L.attn_ln_scale.empty()) {
            hipHostRegister(L.attn_ln_scale.data(), L.attn_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
            hipHostGetDevicePointer(&L.d_attn_ln, L.attn_ln_scale.data(), 0);
        }
        if (!L.mlp_ln_scale.empty()) {
            hipHostRegister(L.mlp_ln_scale.data(), L.mlp_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
            hipHostGetDevicePointer(&L.d_mlp_ln, L.mlp_ln_scale.data(), 0);
        }
    }
    // Pin LM head
    if (!m->lm_head.packed.empty()) {
        hipHostRegister(m->lm_head.packed.data(), m->lm_head.packed.size(), hipHostRegisterDefault);
        hipHostRegister(m->lm_head.scales.data(), m->lm_head.scales.size(), hipHostRegisterDefault);
        hipHostGetDevicePointer(&m->d_lm_head_packed, m->lm_head.packed.data(), 0);
        hipHostGetDevicePointer(&m->d_lm_head_scales, m->lm_head.scales.data(), 0);
    }
    if (!m->final_ln_scale.empty()) {
        hipHostRegister(m->final_ln_scale.data(), m->final_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
        hipHostGetDevicePointer(&m->d_final_ln, m->final_ln_scale.data(), 0);
    }
    return true;
}

static void unpin_model_memory(inference_model* m) {
    auto unpin = [](void* ptr) {
        if (ptr) hipHostUnregister(ptr);
    };
    unpin(m->d_embed_table);
    for (auto& L : m->layers) {
        unpin(L.Wq.d_packed); unpin(L.Wq.d_scales);
        unpin(L.Wk.d_packed); unpin(L.Wk.d_scales);
        unpin(L.Wv.d_packed); unpin(L.Wv.d_scales);
        unpin(L.Wo.d_packed); unpin(L.Wo.d_scales);
        unpin(L.W_gate.d_packed); unpin(L.W_gate.d_scales);
        unpin(L.W_up.d_packed); unpin(L.W_up.d_scales);
        unpin(L.W_down.d_packed); unpin(L.W_down.d_scales);
        unpin(L.d_attn_ln); unpin(L.d_mlp_ln);
    }
    unpin(m->d_lm_head_packed); unpin(m->d_lm_head_scales);
    unpin(m->d_final_ln);
}

// Simplified tokenizer: split on whitespace, map to vocab (placeholder)
static std::vector<int> simple_tokenize(inference_model* m, const char* text) {
    std::vector<int> tokens;
    const char* p = text;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n') ++p;
        if (!*p) break;
        const char* start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') ++p;
        std::string word(start, p - start);
        auto it = m->vocab_map.find(word);
        if (it != m->vocab_map.end()) tokens.push_back(it->second);
        else tokens.push_back(0);  // UNK
    }
    if (tokens.empty()) tokens.push_back(1);  // BOS
    return tokens;
}

// Forward pass for a single token at position `pos` (0-indexed in sequence)
static int forward_pass(inference_model* m, int token_id, int pos, float* logits_out) {
    ensure_kernels_loaded();

    // Scratch buffer layout (simplified single-token version):
    // We'll allocate a large unified buffer per forward pass
    const int E = m->n_embd, HD = m->head_dim, HQ = m->n_head, HKV = m->n_head_kv;
    const int DQ = HQ * HD, DKV = HKV * HD, GRP = HQ / HKV;
    const float scale = 1.0f / sqrtf((float)HD);
    const float eps = 1e-5f;

    // For Phase 4, we implement the forward pass step by step using the verified kernels.
    // This is a skeleton showing the structure; full implementation requires careful
    // buffer management matching the attn_verify pattern.

    // 1. Embedding lookup: token_id -> x [1 x E] FP16
    __half* x;
    // ... allocate in scratch ...

    // 2. For each layer:
    //    a) RMSNorm (attn_ln) -> h
    //    b) Q/K/V projections via gemm_act_fp4 (h x Wq/Wk/Wv)
    //    c) RoPE on Q/K per head
    //    d) KV store into cache at position `pos`
    //    e) Attention: scores = Q @ K^T (fp16_gemm), softmax, out = scores @ V (fp16_gemm)
    //    f) O projection (gemm_act_fp4)
    //    g) Residual add
    //    h) RMSNorm (mlp_ln)
    //    i) SwiGLU (gate_proj + up_proj -> silu(gate)*up -> down_proj)
    //    j) Residual add

    // 3. Final RMSNorm
    // 4. LM head projection -> logits

    // TODO: Full implementation with proper buffer layout matching attn_verify
    // For now, return a placeholder error indicating not fully implemented
    set_error("Phase 4 forward pass not yet fully implemented; kernel chain skeleton only");
    return -1;
}

// ======================== C API IMPLEMENTATION ========================

inference_model_t* inference_load_model(const char* gguf_path) {
    clear_error();
    if (!gguf_path) { set_error("NULL path"); return nullptr; }

    inference_model_t* m = new inference_model();
    m->gguf = gguf_mmap_open(gguf_path);
    if (!m->gguf) { set_error("gguf_mmap_open failed"); delete m; return nullptr; }

    // Read metadata (simplified - real impl queries GGUF keys)
    m->model_name = "qwen3-unknown";
    m->n_layer = 28;    // Qwen3-7B
    m->n_embd = 4096;
    m->n_head = 32;
    m->n_head_kv = 8;
    m->head_dim = 128;
    m->vocab_size = 151936;
    m->n_ctx = 4096;

    // Load embedding table
    size_t sz;
    const void* emb_ptr = gguf_get_tensor_ptr(m->gguf, "token_embd.weight", &sz);
    if (emb_ptr && sz == (size_t)m->vocab_size * m->n_embd * sizeof(__half)) {
        m->embed_table.assign((const __half*)emb_ptr, (const __half*)emb_ptr + m->vocab_size * m->n_embd);
    }

    // Load per-layer weights (simplified - real impl iterates GGUF tensor names)
    m->layers.resize(m->n_layer);
    for (int i = 0; i < m->n_layer; ++i) {
        LayerWeights& L = m->layers[i];
        char prefix[64];
        // This is a placeholder - real code would query actual tensor names from GGUF
        // load_quant_weight(m->gguf, "blk.0.attn_q", L.Wq); etc.
    }

    // Load LM head
    // load_quant_weight(m->gguf, "output.weight", m->lm_head);

    // Load final LN
    // ...

    if (!pin_model_memory(m)) {
        set_error("hipHostRegister failed");
        unpin_model_memory(m);
        delete m;
        return nullptr;
    }

    return m;
}

void inference_free_model(inference_model_t* m) {
    if (!m) return;
    unpin_model_memory(m);
    if (m->kvc) kv_cache_free(m->kvc);
    if (m->gguf) gguf_mmap_close(m->gguf);
    delete m;
}

int32_t* inference_tokenize(inference_model_t* m, const char* text, int* out_len) {
    clear_error();
    if (!m || !text || !out_len) { set_error("Invalid args"); return nullptr; }
    std::vector<int> toks = simple_tokenize(m, text);
    int32_t* arr = new int32_t[toks.size()];
    std::memcpy(arr, toks.data(), toks.size() * sizeof(int32_t));
    *out_len = (int)toks.size();
    return arr;
}

void inference_free_tokens(int32_t* tokens, int len) {
    (void)len;
    delete[] tokens;
}

int inference_kv_cache_init(inference_model_t* m, int n_ctx) {
    clear_error();
    if (!m) { set_error("NULL model"); return -1; }
    kv_cache_config_t cfg{1, n_ctx, m->n_embd, m->head_dim, m->n_head, m->n_head_kv};
    m->kvc = kv_cache_create("/dev/shm/inference_kvc.bin", &cfg);
    if (!m->kvc) { set_error("kv_cache_create failed"); return -1; }
    m->n_ctx = n_ctx;
    return 0;
}

void inference_kv_cache_free(inference_model_t* m) {
    if (m && m->kvc) { kv_cache_free(m->kvc); m->kvc = nullptr; }
}

void inference_set_seed(inference_model_t* m, unsigned int seed) {
    (void)m; (void)seed; // TODO: set RNG state
}

int inference_decode_token(inference_model_t* m, int token_id, int* next_token_out) {
    clear_error();
    if (!m || !m->kvc || !next_token_out) { set_error("Invalid state"); return -1; }

    // For Phase 4 skeleton, we just return an error
    // Full implementation runs forward_pass() and samples from logits
    set_error("Phase 4 decode_token not yet implemented");
    return -1;
}

int inference_get_logits(inference_model_t* m, int* vocab_ids, float* logits) {
    clear_error();
    if (!m || !vocab_ids || !logits) { set_error("Invalid args"); return -1; }
    // Fill vocab_ids with 0..vocab_size-1
    for (int i = 0; i < m->vocab_size; ++i) vocab_ids[i] = i;
    // Run forward to get logits (placeholder)
    float* tmp = new float[m->vocab_size];
    int rc = forward_pass(m, 0, 0, tmp); // placeholder
    if (rc == 0) std::memcpy(logits, tmp, m->vocab_size * sizeof(float));
    delete[] tmp;
    return rc;
}

int inference_n_ctx(const inference_model_t* m) {
    return m ? m->n_ctx : 0;
}

int inference_batch_size(const inference_model_t* m) {
    (void)m; return 1;
}

const char* inference_model_name(const inference_model_t* m) {
    return m ? m->model_name.c_str() : nullptr;
}

const void* inference_get_tensor_ptr(const inference_model_t* m, const char* name, size_t* out_bytes) {
    clear_error();
    if (!m || !m->gguf || !name || !out_bytes) { set_error("Invalid args"); return nullptr; }
    size_t off, sz;
    const void* ptr = gguf_get_tensor_ptr(m->gguf, name, &sz);
    if (!ptr) { set_error("Tensor not found: %s", name); return nullptr; }
    *out_bytes = sz;
    return ptr;
}

const char* inference_get_last_error(void) {
    return g_last_error[0] ? g_last_error : nullptr;
}