// inference_internal.h -- Complete type definitions shared between
// inference.cpp and qwen_model.cpp. Both translation units must see
// IDENTICAL struct layouts (defined here, not duplicated).

#ifndef INFERENCE_INTERNAL_H
#define INFERENCE_INTERNAL_H

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>

#include "inference.h"       // opaque handle typedefs
#include "gguf_mmap.h"        // gguf_model_t* forward decl
#include "kv_cache.h"         // inference_kv_cache_t* forward decl
#include "tokenizer.h"        // tokenizer_t typedef

// Forward declarations for types defined later or in other TUs
struct tokenizer;            // BPE tokenizer (defined in tokenizer.h as struct tokenizer)
struct QwenModel;            // Qwen model handle (opaque in qwen_model.h)

// ============================================================
// COMPLETE type definitions - single source of truth for both TUs
// ============================================================

struct QuantWeight {
    int N = 0, K = 0;                                  // output_dim x input_dim (K multiple of 16)
    std::vector<unsigned char> packed;                 // N x K/2 bytes
    std::vector<unsigned char> scales;                // N x K/16 bytes (UE4M3 encoding)
    std::vector<float> scales_f;                       // decoded floats (Q4_0/Q4_1/Q8_0)
    std::vector<float> true_v;                         // effective values (CPU ref)
    void* d_packed = nullptr;                          // device ptr for packed weights
    void* d_scales = nullptr;                          // device ptr for UE4M3 scales
    void* d_scales_f = nullptr;                        // device ptr for decoded float scales
    int8_t quant_type = 0;                             // GGUF tensor type (2=Q4_0, 3=Q4_1, 7=Q8_0, 13=TYPE13, 100=TYPE100)
};

// Block kind, classified once at load time from the tensors present in the GGUF (Phase 20 Step 2).
// The forward pass switches on this; nothing else may derive layer types from the index.
enum LayerKind : int8_t {
    LAYER_FULL_ATTN   = 0,  // attn_q/attn_k/attn_v/attn_output (+ q/k norm, gated Q)
    LAYER_LINEAR_ATTN = 1,  // Gated DeltaNet: attn_qkv, attn_gate, ssm_* tensors
};

struct LayerWeights {
    LayerKind kind = LAYER_FULL_ATTN;
    QuantWeight Wq, Wk, Wv, Wo;                       // Attention projections
    QuantWeight W_gate, W_up, W_down;                 // MLP (SwiGLU)
    std::vector<__half> attn_ln_scale;               // LayerNorm scale (attention)
    std::vector<__half> mlp_ln_scale;                // LayerNorm scale (MLP)
    void* d_attn_ln = nullptr;                        // device ptr for attn norm
    void* d_mlp_ln = nullptr;                         // device ptr for mlp norm
};

// The complete inference_model struct - MUST match exactly between all TUs.
// This replaces the old local definition in inference.cpp and qwen_model.cpp.
// It is aliased as `inference_model` to maintain backward compatibility with
// existing code that references ::inference_model.
struct inference_model {
    gguf_model_t* gguf = nullptr;                      // GGUF file mapping
    std::string model_name;                             // architecture name from GGUF
    int n_layer = 0;                                    // decoder layers run in the stack (block_count - n_nextn)
    int n_nextn = 0;                                    // trailing MTP (nextn) blocks, excluded from the stack
    int n_embd = 0;                                     // embedding dimension
    int n_head = 0;                                     // attention heads
    int n_head_kv = 0;                                  // KV cache heads
    int head_dim = 0;                                   // per-head dimension (attention.key_length)
    int n_ff = 0;                                       // FFN width (feed_forward_length)
    int n_rot = 0;                                      // rotary dims per head (rope.dimension_count)
    float rope_freq_base = 10000.0f;                    // rope.freq_base
    int vocab_size = 0;                                 // vocabulary size
    int n_ctx = 0;                                      // context window size

    std::atomic<int> ref_count{1};                      // reference counting
    std::mutex mtx;                                     // thread safety mutex

    std::vector<__half> embed_table;                   // [vocab_size x n_embd] FP16 token embeddings
    void* d_embed_table = nullptr;                     // device ptr for pinned embed table

    QuantWeight lm_head;                                // output projection weights
    void* d_lm_head_packed = nullptr;                  // device ptr for LM head packed weights
    void* d_lm_head_scales = nullptr;                  // device ptr for LM head scales (UE4M3)
    void* d_lm_head_scales_f = nullptr;                // device ptr for LM head decoded floats

    std::vector<__half> final_ln_scale;               // final LayerNorm scale vector
    void* d_final_ln = nullptr;                        // device ptr for final LN

    std::vector<LayerWeights> layers;                  // per-layer attention + MLP weights

    inference_kv_cache_t* kvc = nullptr;               // KV cache manager handle
    void* scratch_base = nullptr;                      // temporary scratch buffer base
    size_t scratch_bytes = 0;                          // scratch buffer size in bytes

    tokenizer_t* tokenizer = nullptr;                  // BPE tokenizer instance
    QwenModel* qwen = nullptr;                      // Qwen model handle (set during kv_init)

    double eps = 1e-5;                                  // normalization epsilon
    int full_attention_interval = 0;                   // informational only; use LayerWeights::kind

    void* d_embed_table_copy = nullptr;                // GPU copy of embedding table (set during kv_init)
};

#endif // INFERENCE_INTERNAL_H
