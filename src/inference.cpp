// inference.cpp -- Phase 4: C API implementation for zero-copy ROCmFP4 inference engine.
// Thin FFI wrappers around the C++ HIP implementation. All state in inference_model_t.

#include "inference.h"
#include "gguf_mmap.h"
#include "fp4_common.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "qwen_model.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdarg>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <algorithm>
#include <random>

using namespace rocmfp4;

// Global RNG seed for reproducible sampling
static std::atomic<uint32_t> g_rng_seed{0};

// Thread-local error buffer
static thread_local char g_last_error[1024] = {0};

static void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
}

static void clear_error() { g_last_error[0] = 0; }

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

    // Reference counting for shared ownership
    std::atomic<int> ref_count{1};
    std::mutex mtx;

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

    // Tokenizer
    tokenizer_t* tokenizer = nullptr;
    qwen_model_t* qwen = nullptr;
};

static bool pin_model_memory(inference_model* m) {
    // Pin embedding table
    if (!m->embed_table.empty()) {
        hipError_t e = hipHostRegister(m->embed_table.data(),
                                       m->embed_table.size() * sizeof(__half),
                                       hipHostRegisterDefault);
        if (e != hipSuccess) return false;
        (void)hipHostGetDevicePointer(&m->d_embed_table, m->embed_table.data(), 0);
    }
    // Pin per-layer weights
    for (auto& L : m->layers) {
        auto pin = [](QuantWeight& w) -> bool {
            if (w.packed.empty()) return false;
            if (hipHostRegister(w.packed.data(), w.packed.size(), hipHostRegisterDefault) != hipSuccess) return false;
            if (hipHostRegister(w.scales.data(), w.scales.size(), hipHostRegisterDefault) != hipSuccess) return false;
            (void)hipHostGetDevicePointer(&w.d_packed, w.packed.data(), 0);
            (void)hipHostGetDevicePointer(&w.d_scales, w.scales.data(), 0);
            return true;
        };
        if (!pin(L.Wq) || !pin(L.Wk) || !pin(L.Wv) || !pin(L.Wo) ||
            !pin(L.W_gate) || !pin(L.W_up) || !pin(L.W_down)) return false;
        if (!L.attn_ln_scale.empty()) {
            (void)hipHostRegister(L.attn_ln_scale.data(), L.attn_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
            (void)hipHostGetDevicePointer(&L.d_attn_ln, L.attn_ln_scale.data(), 0);
        }
        if (!L.mlp_ln_scale.empty()) {
            (void)hipHostRegister(L.mlp_ln_scale.data(), L.mlp_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
            (void)hipHostGetDevicePointer(&L.d_mlp_ln, L.mlp_ln_scale.data(), 0);
        }
    }
    // Pin LM head
    if (!m->lm_head.packed.empty()) {
        (void)hipHostRegister(m->lm_head.packed.data(), m->lm_head.packed.size(), hipHostRegisterDefault);
        (void)hipHostRegister(m->lm_head.scales.data(), m->lm_head.scales.size(), hipHostRegisterDefault);
        (void)hipHostGetDevicePointer(&m->d_lm_head_packed, m->lm_head.packed.data(), 0);
        (void)hipHostGetDevicePointer(&m->d_lm_head_scales, m->lm_head.scales.data(), 0);
    }
    if (!m->final_ln_scale.empty()) {
        (void)hipHostRegister(m->final_ln_scale.data(), m->final_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
        (void)hipHostGetDevicePointer(&m->d_final_ln, m->final_ln_scale.data(), 0);
    }
    return true;
}

static void unpin_model_memory(inference_model* m) {
    auto unpin = [](void* ptr) {
        if (ptr) (void)hipHostUnregister(ptr);
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



// Helper: find tensor index by name
static int find_tensor_index(gguf_model_t* gguf, const char* name) {
    int64_t n = gguf_n_tensors(gguf);
    for (int64_t i = 0; i < n; ++i) {
        const char* tn = gguf_tensor_name(gguf, i);
        if (tn && strcmp(tn, name) == 0) return (int)i;
    }
    return -1;
}

// Check if a tensor exists by name in the GGUF file
static bool has_tensor(gguf_model_t* gguf, const char* name) {
    int64_t n = gguf_n_tensors(gguf);
    for (int64_t i = 0; i < n; ++i) {
        if (strcmp(gguf_tensor_name(gguf, i), name) == 0) return true;
    }
    return false;
}

// Load quantized weight by name from GGUF (for non-QKV tensors)
static bool load_quant_weight_from_gguf(gguf_model_t* gguf, const char* weight_name, const char* scales_name, QuantWeight& w) {
    size_t w_sz, s_sz;
    const void* w_ptr = gguf_get_tensor_ptr(gguf, weight_name, &w_sz);
    const void* s_ptr = gguf_get_tensor_ptr(gguf, scales_name, &s_sz);
    if (!w_ptr || !s_ptr) {
        fprintf(stderr, "Missing tensor %s or %s\n", weight_name, scales_name);
        return false;
    }
    // Get tensor metadata for shape
    int idx = find_tensor_index(gguf, weight_name);
    if (idx < 0) {
        fprintf(stderr, "Weight tensor %s not found in index\n", weight_name);
        return false;
    }
    // Get dimensions: assume 2D [out_dim, in_dim]
    int64_t* ne = gguf_tensor_ne(gguf, idx);
    if (!ne) return false;
    int64_t out_dim = ne[0];
    int64_t in_dim = ne[1];
    w.N = (int)out_dim;
    w.K = (int)in_dim;
    // Copy packed bytes and scales
    w.packed.assign((const unsigned char*)w_ptr, (const unsigned char*)w_ptr + w_sz);
    w.scales.assign((const unsigned char*)s_ptr, (const unsigned char*)s_ptr + s_sz);
    // Build true_v for CPU reference (dequant)
    w.true_v.resize((size_t)w.N * w.K);
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

// Load all model weights from GGUF into the inference_model struct
static bool load_model_weights(inference_model* m) {
    m->layers.resize(m->n_layer);
    for (int i = 0; i < m->n_layer; ++i) {
        LayerWeights& L = m->layers[i];
        char wname[128], sname[128];

        // --- Attention Q,K,V,O projections ---
        // Phase 15.4: Detect SSM-only layers in hybrid architecture and skip them.
        // Check if ANY attention tensors exist for this layer. If not, it's an SSM layer.
        bool is_ssm_layer = false;
        {
            char ssm_check[128];
            snprintf(ssm_check, sizeof(ssm_check), "blk.%d.ssm_alpha", i);
            if (has_tensor(m->gguf, ssm_check)) {
                // This is an SSM layer — skip entirely (Phase 15.4)
                is_ssm_layer = true;
                fprintf(stderr, "[load_model_weights] blk.%d: SSM layer detected, skipping\n", i);
            }
        }
        
        if (is_ssm_layer) continue;  // Skip SSM-only layers entirely
        
        // Check if we have a combined QKV tensor (Qwen3.5-4B pattern) or separate ones (Qwen3-7B)
        snprintf(wname, sizeof(wname), "blk.%d.attn_qkv.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_qkv.scales", i);
        
        size_t qkv_w_sz = 0;
        const void* qkv_ptr = gguf_get_tensor_ptr(m->gguf, wname, &qkv_w_sz);
        
        if (qkv_ptr && qkv_w_sz > 0) {
            // Combined QKV tensor found — load and split into Wq/Wk/Wv
            fprintf(stderr, "[load_model_weights] blk.%d: using combined attn_qkv weight [%ld x %ld]\n", 
                    i, (long)m->layers[i].Wq.N, (long)(m->n_head * m->head_dim + 2 * m->n_head_kv * m->head_dim));
            
            int qidx = find_tensor_index(m->gguf, wname);
            if (qidx < 0) return false;
            int64_t* ne = gguf_tensor_ne(m->gguf, qidx);
            if (!ne) return false;
            
            int64_t E = ne[0];              // embedding length (rows)
            int64_t DQ = (int64_t)m->n_head * m->head_dim;   // Q columns
            int64_t DKV = (int64_t)m->n_head_kv * m->head_dim; // K/V columns
            
            size_t total_cols = DQ + 2 * DKV;
            size_t packed_bytes_per_row = total_cols >> 1;  // 2 nibbles per byte
            
            L.Wq.N = (int)E; L.Wq.K = (int)DQ;
            L.Wk.N = (int)E; L.Wk.K = (int)DKV;
            L.Wv.N = (int)E; L.Wv.K = (int)DKV;
            
            // Load scales tensor  
            size_t s_sz;
            const void* s_ptr = gguf_get_tensor_ptr(m->gguf, sname, &s_sz);
            if (!s_ptr) {
                fprintf(stderr, "blk.%d: combined qkv scales tensor not found\n", i);
                return false;
            }
            
            const unsigned char* pk = (const unsigned char*)qkv_ptr;
            
            // Allocate packed arrays for each projection
            size_t q_packed_bytes = (size_t)E * ((size_t)DQ >> 1);
            size_t k_packed_bytes = (size_t)E * ((size_t)DKV >> 1);
            size_t v_packed_bytes = (size_t)E * ((size_t)DKV >> 1);
            
            L.Wq.packed.resize(q_packed_bytes);
            L.Wk.packed.resize(k_packed_bytes);
            L.Wv.packed.resize(v_packed_bytes);
            
            // Extract packed data row by row
            for (int64_t row = 0; row < E; ++row) {
                const unsigned char* src_row = pk + row * packed_bytes_per_row;
                
                // Q: columns [0 .. DQ-1] → packed cols [0 .. DQ/2-1]
                for (int c = 0; c < (int)DQ - 1; c += 2) {
                    size_t dst_off = (size_t)(c >> 1);
                    size_t src_off = (size_t)(c >> 1);
                    L.Wq.packed[(size_t)row * q_packed_bytes / E + dst_off] = src_row[src_off];
                }
                // Handle odd column at end of Q if DQ is odd
                if (DQ & 1) {
                    int last_c = (int)(DQ - 1);
                    size_t src_off = (size_t)(last_c >> 1);
                    size_t dst_off = (size_t)(last_c >> 1);
                    unsigned char b = src_row[src_off];
                    L.Wq.packed[(size_t)row * q_packed_bytes / E + dst_off] = (unsigned char)(b & 0x0F);
                }
                
                // K: columns [DQ .. DQ+DKV-1] → packed cols [0 .. DKV/2-1]
                for (int c = 0; c < (int)DKV - 1; c += 2) {
                    int global_col = (int)DQ + c;
                    size_t src_off = (size_t)(global_col >> 1);
                    size_t dst_off = (size_t)(c >> 1);
                    L.Wk.packed[(size_t)row * k_packed_bytes / E + dst_off] = src_row[src_off];
                }
                if (DKV & 1) {
                    int last_c = (int)(DKV - 1);
                    int global_col = (int)DQ + last_c;
                    size_t src_off = (size_t)(global_col >> 1);
                    size_t dst_off = (size_t)(last_c >> 1);
                    unsigned char b = src_row[src_off];
                    L.Wk.packed[(size_t)row * k_packed_bytes / E + dst_off] = (unsigned char)(b & 0x0F);
                }
                
                // V: columns [DQ+DKV .. DQ+2*DKV-1] → packed cols [0 .. DKV/2-1]  
                for (int c = 0; c < (int)DKV - 1; c += 2) {
                    int global_col = (int)DQ + (int)DKV + c;
                    size_t src_off = (size_t)(global_col >> 1);
                    size_t dst_off = (size_t)(c >> 1);
                    L.Wv.packed[(size_t)row * v_packed_bytes / E + dst_off] = src_row[src_off];
                }
                if (DKV & 1) {
                    int last_c = (int)(DKV - 1);
                    int global_col = (int)DQ + (int)DKV + last_c;
                    size_t src_off = (size_t)(global_col >> 1);
                    size_t dst_off = (size_t)(last_c >> 1);
                    unsigned char b = src_row[src_off];
                    L.Wv.packed[(size_t)row * v_packed_bytes / E + dst_off] = (unsigned char)(b & 0x0F);
                }
            }
            
            // Load scales — try per-projection first, then combined fallback  
            char sname_q[128], sname_k[128], sname_v[128];
            snprintf(sname_q, sizeof(sname_q), "blk.%d.attn_q.scales", i);
            snprintf(sname_k, sizeof(sname_k), "blk.%d.attn_k.scales", i);
            snprintf(sname_v, sizeof(sname_v), "blk.%d.attn_v.scales", i);
            
            size_t sq_sz;
            const void* sq_ptr = gguf_get_tensor_ptr(m->gguf, sname_q, &sq_sz);
            
            if (sq_ptr && sq_sz > 0) {
                // Separate scales for each projection (Qwen3-7B style)
                const void* sk_p = gguf_get_tensor_ptr(m->gguf, sname_k, &sq_sz);
                const void* sv_p = gguf_get_tensor_ptr(m->gguf, sname_v, &sq_sz);
                
                L.Wq.scales.assign((const unsigned char*)sq_ptr, (const unsigned char*)sq_ptr + sq_sz);
                L.Wk.scales.assign((const unsigned char*)sk_p, (const unsigned char*)sk_p + sq_sz);
                L.Wv.scales.assign((const unsigned char*)sv_p, (const unsigned char*)sv_p + sq_sz);
            } else {
                // Combined scales tensor — split by row index and scale column offset
                int sc_idx = find_tensor_index(m->gguf, sname);
                if (sc_idx < 0) return false;
                
                int64_t* ne_sc = gguf_tensor_ne(m->gguf, sc_idx);
                if (!ne_sc) return false;
                size_t total_scale_cols = (size_t)ne_sc[1];
                
                // Scales per row for Q/K/V projections: ceil(DQ/16), ceil(DKV/16), ceil(DKV/16)
                size_t q_scales_per_row = ((size_t)DQ + 15) >> 4;
                size_t k_scales_per_row = ((size_t)DKV + 15) >> 4;
                size_t v_scales_per_row = ((size_t)DKV + 15) >> 4;
                
                L.Wq.scales.resize((size_t)E * q_scales_per_row);
                L.Wk.scales.resize((size_t)E * k_scales_per_row);
                L.Wv.scales.resize((size_t)E * v_scales_per_row);
                
                const unsigned char* src_sc = (const unsigned char*)s_ptr;
                size_t scales_per_row_total = total_scale_cols;
                
                for (int64_t row = 0; row < E; ++row) {
                    size_t src_row_off = row * scales_per_row_total;
                    
                    // Q scales: first q_scales_per_row entries in this row
                    for (size_t s = 0; s < q_scales_per_row; ++s) {
                        L.Wq.scales[row * q_scales_per_row + s] = src_sc[src_row_off + s];
                    }
                    
                    // K scales: next k_scales_per_row entries 
                    int k_start = (int)q_scales_per_row;
                    for (size_t s = 0; s < k_scales_per_row; ++s) {
                        if ((size_t)(k_start + s) < scales_per_row_total) {
                            L.Wk.scales[row * k_scales_per_row + s] = src_sc[src_row_off + (size_t)k_start + s];
                        }
                    }
                    
                    // V scales: next v_scales_per_row entries
                    int v_start = (int)(q_scales_per_row + k_scales_per_row);
                    for (size_t s = 0; s < v_scales_per_row; ++s) {
                        if ((size_t)(v_start + s) < scales_per_row_total) {
                            L.Wv.scales[row * v_scales_per_row + s] = src_sc[src_row_off + (size_t)v_start + s];
                        }
                    }
                }
            }
            
        } else {
            // Separate Q/K/V tensors — original path (backward compatible with Qwen3-7B)
            snprintf(wname, sizeof(wname), "blk.%d.attn_q.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.attn_q.scales", i);
            if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.Wq)) return false;
            
            snprintf(wname, sizeof(wname), "blk.%d.attn_k.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.attn_k.scales", i);
            if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.Wk)) return false;
            
            snprintf(wname, sizeof(wname), "blk.%d.attn_v.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.attn_v.scales", i);
            if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.Wv)) return false;
        }

        // Attention O projection (always separate)
        snprintf(wname, sizeof(wname), "blk.%d.attn_o.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_o.scales", i);
        if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.Wo)) return false;
        // MLP gate, up, down
        snprintf(wname, sizeof(wname), "blk.%d.ffn_gate.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_gate.scales", i);
        if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.W_gate)) return false;
        snprintf(wname, sizeof(wname), "blk.%d.ffn_up.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_up.scales", i);
        if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.W_up)) return false;
        snprintf(wname, sizeof(wname), "blk.%d.ffn_down.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_down.scales", i);
        if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.W_down)) return false;
        // LayerNorm scales
        size_t ln_sz;
        const void* ln_ptr = gguf_get_tensor_ptr(m->gguf, 
            (std::string("blk.") + std::to_string(i) + ".attn_norm.weight").c_str(), &ln_sz);
        if (ln_ptr && ln_sz == (size_t)m->n_embd * sizeof(__half)) {
            L.attn_ln_scale.assign((const __half*)ln_ptr, (const __half*)ln_ptr + m->n_embd);
        }
        ln_ptr = gguf_get_tensor_ptr(m->gguf,
            (std::string("blk.") + std::to_string(i) + ".ffn_norm.weight").c_str(), &ln_sz);
        if (ln_ptr && ln_sz == (size_t)m->n_embd * sizeof(__half)) {
            L.mlp_ln_scale.assign((const __half*)ln_ptr, (const __half*)ln_ptr + m->n_embd);
        }
    }
    return true;
}

// ======================== C API IMPLEMENTATION ========================

inference_model_t* inference_load_model(const char* gguf_path) {
    clear_error();
    if (!gguf_path) { set_error("NULL path"); return nullptr; }

    inference_model_t* m = new inference_model();
    m->gguf = gguf_mmap_open(gguf_path);
    if (!m->gguf) { set_error("gguf_mmap_open failed"); delete m; return nullptr; }

    // Read model metadata from GGUF (Phase 15.1)
    int64_t n_layer = gguf_block_count(m->gguf);
    if (n_layer <= 0) { set_error("block_count not found in GGUF"); delete m; return nullptr; }
    
    int64_t n_embd = gguf_embedding_length(m->gguf);
    if (n_embd <= 0) { set_error("embedding_length not found in GGUF"); delete m; return nullptr; }
    
    int64_t n_head = gguf_head_count(m->gguf);
    int64_t n_head_kv = gguf_head_count_kv(m->gguf);
    
    // head_dim = embedding_length / head_count (derived, not stored)
    int64_t hd = n_embd / n_head;
    
    // vocab_size: prefer tokenizer.ggml.vocab_size or array length
    int64_t vs = gguf_vocab_size(m->gguf);
    if (vs <= 0) {
        const char** tok_arr = nullptr; size_t tok_cnt = 0;
        if (gguf_kv_get_string_array(m->gguf, "tokenizer.ggml.tokens", &tok_arr, &tok_cnt) == 0) {
            vs = (int64_t)tok_cnt;
        }
    }
    if (vs <= 0) { set_error("vocab_size not found in GGUF"); delete m; return nullptr; }
    
    int64_t ctx_len = gguf_context_length(m->gguf);
    if (ctx_len <= 0) ctx_len = 4096;  // fallback default
    
    // Read architecture string from GGUF
    const char* arch = gguf_arch(m->gguf);
    m->model_name = arch ? arch : "qwen3-unknown";
    
    m->n_layer = (int)n_layer;
    m->n_embd = (int)n_embd;
    m->n_head = (int)n_head;
    m->n_head_kv = (int)n_head_kv;
    m->head_dim = (int)hd;
    m->vocab_size = (int)vs;
    m->n_ctx = (int)ctx_len;

    // Load embedding table
    size_t sz;
    const void* emb_ptr = gguf_get_tensor_ptr(m->gguf, "token_embd.weight", &sz);
    if (emb_ptr && sz == (size_t)m->vocab_size * m->n_embd * sizeof(__half)) {
        m->embed_table.assign((const __half*)emb_ptr, (const __half*)emb_ptr + m->vocab_size * m->n_embd);
    }

    // Load per-layer weights from GGUF tensors
    if (!load_model_weights(m)) {
        set_error("Failed to load model weights");
        unpin_model_memory(m);
        delete m;
        return nullptr;
    }

    // Load LM head
    if (!load_quant_weight_from_gguf(m->gguf, "output.weight", "output.scales", m->lm_head)) {
        set_error("Failed to load LM head");
        unpin_model_memory(m);
        delete m;
        return nullptr;
    }

    // Load final LN scale
    const void* final_ln_ptr = gguf_get_tensor_ptr(m->gguf, "output_norm.weight", &sz);
    if (final_ln_ptr && sz == (size_t)m->n_embd * sizeof(__half)) {
        m->final_ln_scale.assign((const __half*)final_ln_ptr, (const __half*)final_ln_ptr + m->n_embd);
    }

    // Create tokenizer from GGUF metadata (Phase 5/8)
    m->tokenizer = tokenizer_load_from_gguf(m->gguf);

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
    if (m->tokenizer) tokenizer_free(m->tokenizer);
    if (m->qwen) qwen_model_free(m->qwen);
    unpin_model_memory(m);
    if (m->kvc) kv_cache_free(m->kvc);
    if (m->gguf) gguf_mmap_close(m->gguf);
    delete m;
}

int32_t* inference_tokenize(inference_model_t* m, const char* text, int* out_len) {
    clear_error();
    if (!m || !text || !out_len) { set_error("Invalid args"); return nullptr; }
    if (!m->tokenizer) { set_error("No tokenizer available"); return nullptr; }
    // allocate max possible tokens (len+2)
    int max_toks = (int)strlen(text) + 2;
    std::vector<int> buf(max_toks);
    int n = tokenizer_encode(m->tokenizer, text, buf.data(), max_toks);
    if (n < 0) { set_error("tokenizer_encode failed"); return nullptr; }
    int32_t* arr = new int32_t[n];
    std::memcpy(arr, buf.data(), n * sizeof(int32_t));
    *out_len = n;
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
    // Create Qwen model for forward passes
    m->qwen = qwen_model_create(m);
    if (!m->qwen) { set_error("qwen_model_create failed"); return -1; }
    return 0;
}

void inference_kv_cache_free(inference_model_t* m) {
    if (m && m->kvc) { kv_cache_free(m->kvc); m->kvc = nullptr; }
}

void inference_set_seed(inference_model_t* m, unsigned int seed) {
    (void)m;
    g_rng_seed.store(seed, std::memory_order_relaxed);
}

int inference_decode_token(inference_model_t* m, int token_id, int* next_token_out) {
    clear_error();
    if (!m || !m->kvc || !m->qwen || !next_token_out) { set_error("Invalid state"); return -1; }

    // Need a position counter; store in model (simple static)
    static thread_local int pos = 0;
    float* logits = new float[m->vocab_size];
    int rc = qwen_model_forward(m->qwen, token_id, pos, logits);
    if (rc != 0) { delete[] logits; set_error("qwen_model_forward failed"); return -1; }

    // Greedy sampling
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < m->vocab_size; ++i) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    *next_token_out = best;
    ++pos;
    delete[] logits;
    return 0;
}

int inference_get_logits(inference_model_t* m, int* vocab_ids, float* logits) {
    clear_error();
    if (!m || !m->qwen || !vocab_ids || !logits) { set_error("Invalid args"); return -1; }
    // Fill vocab_ids with 0..vocab_size-1
    for (int i = 0; i < m->vocab_size; ++i) vocab_ids[i] = i;
    // Need a dummy token_id and position 0 for logits
    static thread_local int pos = 0;
    int rc = qwen_model_forward(m->qwen, 0, pos, logits);
    if (rc == 0) {
        for (int i = 0; i < m->vocab_size; ++i) vocab_ids[i] = i;
    }
    return rc;
}

// Sample from logits with temperature, top-k, top-p.
// vocab_ids: array of vocab indices (size vocab_size)
// logits: raw logits array (size vocab_size)
// Returns sampled token ID, or -1 on error.
int inference_sample_logits(inference_model_t* m, float temperature, int top_k, float top_p,
                           int* vocab_ids, float* logits, int* out_token) {
    clear_error();
    if (!m || !vocab_ids || !logits || !out_token) { set_error("Invalid args"); return -1; }
    int V = m->vocab_size;
    // Copy logits to local vector for manipulation
    std::vector<float> l(logits, logits + V);
    // Temperature scaling
    if (temperature > 0.0f) {
        for (int i = 0; i < V; ++i) l[i] /= temperature;
    }
    // Softmax to get probabilities
    float max_l = l[0];
    for (int i = 1; i < V; ++i) if (l[i] > max_l) max_l = l[i];
    float sum = 0.0f;
    for (int i = 0; i < V; ++i) { l[i] = expf(l[i] - max_l); sum += l[i]; }
    for (int i = 0; i < V; ++i) l[i] /= sum;

    // Top-k
    if (top_k > 0 && top_k < V) {
        std::vector<std::pair<float,int>> idx;
        idx.reserve(V);
        for (int i = 0; i < V; ++i) idx.emplace_back(l[i], i);
        std::nth_element(idx.begin(), idx.begin() + top_k, idx.end(),
                         [](const auto& a, const auto& b){ return a.first > b.first; });
        float cutoff = idx[top_k].first;
        for (int i = 0; i < V; ++i) if (l[i] < cutoff) l[i] = 0.0f;
        // renormalize
        sum = 0.0f; for (int i = 0; i < V; ++i) sum += l[i];
        if (sum > 0) for (int i = 0; i < V; ++i) l[i] /= sum;
    }
    // Top-p (nucleus)
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<std::pair<float,int>> idx;
        idx.reserve(V);
        for (int i = 0; i < V; ++i) idx.emplace_back(l[i], i);
        std::sort(idx.begin(), idx.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        float cum = 0.0f;
        int cutoff_idx = V;
        for (int i = 0; i < V; ++i) {
            cum += idx[i].first;
            if (cum >= top_p) { cutoff_idx = i + 1; break; }
        }
        for (int i = cutoff_idx; i < V; ++i) l[idx[i].second] = 0.0f;
        sum = 0.0f; for (int i = 0; i < V; ++i) sum += l[i];
        if (sum > 0) for (int i = 0; i < V; ++i) l[i] /= sum;
    }

    // Sample from distribution
    static thread_local std::mt19937 rng([](){
        uint32_t seed = g_rng_seed.load(std::memory_order_relaxed);
        if (seed != 0) return std::mt19937(seed);
        return std::mt19937((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }());
    std::discrete_distribution<int> dist(l.begin(), l.end());
    int sampled = dist(rng);
    *out_token = vocab_ids[sampled];
    return 0;
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

int inference_vocab_size(const inference_model_t* m) {
    return m ? m->vocab_size : 0;
}

const void* inference_get_tensor_ptr(const inference_model_t* m, const char* name, size_t* out_bytes) {
    clear_error();
    if (!m || !m->gguf || !name || !out_bytes) { set_error("Invalid args"); return nullptr; }
    size_t sz;
    const void* ptr = gguf_get_tensor_ptr(m->gguf, name, &sz);
    if (!ptr) { set_error("Tensor not found: %s", name); return nullptr; }
    *out_bytes = sz;
    return ptr;
}

const char* inference_get_last_error(void) {
    return g_last_error[0] ? g_last_error : nullptr;
}

// Error string conversion
const char* inference_strerror(inference_error_t err) {
    switch (err) {
        case INFERENCE_OK: return "Success";
        case INFERENCE_ERR_INVALID_ARG: return "Invalid argument";
        case INFERENCE_ERR_OUT_OF_MEMORY: return "Out of memory";
        case INFERENCE_ERR_MODEL_LOAD: return "Model load failed";
        case INFERENCE_ERR_KV_CACHE: return "KV cache error";
        case INFERENCE_ERR_TOKENIZER: return "Tokenizer error";
        case INFERENCE_ERR_KERNEL_LAUNCH: return "Kernel launch failed";
        case INFERENCE_ERR_INVALID_STATE: return "Invalid state";
        case INFERENCE_ERR_NOT_IMPLEMENTED: return "Not implemented";
        default: return "Unknown error";
    }
}

// Reference counting
void inference_model_retain(inference_model_t* m) {
    if (m) m->ref_count.fetch_add(1, std::memory_order_relaxed);
}

void inference_model_release(inference_model_t* m) {
    if (!m) return;
    int old = m->ref_count.fetch_sub(1, std::memory_order_acq_rel);
    if (old == 1) {
        inference_free_model(m);
    }
}
// Batched forward: runs forward pass for a batch of tokens to fill KV cache.
int inference_forward_batch(inference_model_t* model,
                            const int32_t* tokens,
                            const int* positions,
                            const int* seq_lens,
                            int batch_size,
                            int seq_len) {
    clear_error();
    (void)tokens; (void)positions; (void)seq_lens; (void)batch_size; (void)seq_len;
    if (!model || !model->qwen) { set_error("Invalid arguments"); return INFERENCE_ERR_INVALID_ARG; }
    // Actual batched launch is in qwen_model.cpp; this stub validates inputs.
    return INFERENCE_OK;
}

// Generation loop helper
int inference_generate(inference_model_t* model,
                       const int32_t* prompt_ids,
                       int prompt_len,
                       int max_new_tokens,
                       float temperature,
                       int top_k,
                       float top_p,
                       int32_t* out_ids,
                       int* out_len) {
    clear_error();
    if (!model || !prompt_ids || !out_ids || !out_len || prompt_len <= 0 || max_new_tokens <= 0) {
        set_error("Invalid arguments");
        return INFERENCE_ERR_INVALID_ARG;
    }
    // Prefill prompt
    for (int i = 0; i < prompt_len; ++i) {
        int next = 0;
        int rc = inference_decode_token(model, prompt_ids[i], &next);
        if (rc != 0) return rc;
    }
    // Generate new tokens
    int generated = 0;
    float* logits = new float[model->vocab_size];
    int* vocab_ids = new int[model->vocab_size];
    for (int i = 0; i < model->vocab_size; ++i) vocab_ids[i] = i;
    for (int step = 0; step < max_new_tokens; ++step) {
        int rc = inference_get_logits(model, vocab_ids, logits);
        if (rc != 0) { delete[] logits; delete[] vocab_ids; return rc; }
        int next = 0;
        rc = inference_sample_logits(model, temperature, top_k, top_p, vocab_ids, logits, &next);
        if (rc != 0) { delete[] logits; delete[] vocab_ids; return rc; }
        out_ids[generated++] = next;
        // feed next token back
        rc = inference_decode_token(model, next, &next);
        if (rc != 0) { delete[] logits; delete[] vocab_ids; return rc; }
        if (next == tokenizer_eos_id(model->tokenizer)) break; // stop on EOS
    }
    *out_len = generated;
    delete[] logits;
    delete[] vocab_ids;
    return INFERENCE_OK;
}

