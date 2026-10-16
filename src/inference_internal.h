// inference_internal.h -- Complete type definitions shared between
// inference.cpp and qwen_model.cpp. Both translation units must see
// IDENTICAL struct layouts (defined here, not duplicated).

#ifndef INFERENCE_INTERNAL_H
#define INFERENCE_INTERNAL_H

#include <atomic>
#include <mutex>
#include <string>
#include <vector>
#include <cstddef>
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

// One weight tensor, read in place from the pinned GGUF mapping (Phase 20 Step 3c). No repacking:
// qmatvec_launch() decodes the native ggml blocks on the GPU, qt_to_float() on the CPU.
struct QuantWeight {
    int type = -1;                                     // ggml type id (quant_types.h)
    int N = 0;                                         // rows = output dim = ne[1]
    int K = 0;                                         // values per row = input dim = ne[0] (contiguous)
    size_t row_bytes = 0;                              // qt_row_size(type, K)
    const void* data = nullptr;                        // row 0, inside the GGUF mapping
    const char* row(int n) const { return (const char*)data + (size_t)n * row_bytes; }
};

// Block kind, classified once at load time from the tensors present in the GGUF (Phase 20 Step 2).
// The forward pass switches on this; nothing else may derive layer types from the index.
enum LayerKind : int8_t {
    LAYER_FULL_ATTN   = 0,  // attn_q/attn_k/attn_v/attn_output (+ q/k norm, gated Q)
    LAYER_LINEAR_ATTN = 1,  // Gated DeltaNet: attn_qkv, attn_gate, ssm_* tensors
};

struct LayerWeights {
    LayerKind kind = LAYER_FULL_ATTN;
    QuantWeight Wq, Wk, Wv, Wo;                       // full attention only; Wq rows are [Q | gate] per head
    QuantWeight W_gate, W_up, W_down;                 // MLP (SwiGLU)
    const float* attn_norm = nullptr;                 // [n_embd] F32, pre-mixer RMSNorm gamma
    const float* ffn_norm = nullptr;                  // [n_embd] F32, post_attention_norm (pre-FFN)
    const float* q_norm = nullptr;                    // [head_dim] F32, full attention only
    const float* k_norm = nullptr;                    // [head_dim] F32, full attention only
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

    const void* map_base = nullptr;                    // GGUF mapping, pinned once read-only
    size_t map_size = 0;
    bool map_pinned = false;

    QuantWeight token_embd;                            // [vocab x n_embd]; rows decoded on the CPU per token
    QuantWeight output;                                // LM head; same tensor as token_embd when tied
    const float* output_norm = nullptr;                // [n_embd] F32 final RMSNorm gamma

    std::vector<LayerWeights> layers;                  // per-layer attention + MLP weights

    inference_kv_cache_t* kvc = nullptr;               // KV cache manager handle
    void* scratch_base = nullptr;                      // temporary scratch buffer base
    size_t scratch_bytes = 0;                          // scratch buffer size in bytes

    tokenizer_t* tokenizer = nullptr;                  // BPE tokenizer instance
    QwenModel* qwen = nullptr;                      // Qwen model handle (set during kv_init)

    double eps = 1e-5;                                  // normalization epsilon
    int full_attention_interval = 0;                   // informational only; use LayerWeights::kind
};

#endif // INFERENCE_INTERNAL_H
