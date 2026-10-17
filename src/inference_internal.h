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
    int kv_slot = -1;                                 // full attention: this layer's KV cache layer; else -1
    QuantWeight Wq, Wk, Wv, Wo;                       // full attention only; Wq rows are [Q | gate] per head
    QuantWeight W_gate, W_up, W_down;                 // MLP (SwiGLU); the shared expert when moe
    const float* attn_norm = nullptr;                 // [n_embd] F32, pre-mixer RMSNorm gamma
    const float* ffn_norm = nullptr;                  // [n_embd] F32, post_attention_norm (pre-FFN)
    const float* q_norm = nullptr;                    // [head_dim] F32, full attention only
    const float* k_norm = nullptr;                    // [head_dim] F32, full attention only

    // Gated DeltaNet (LAYER_LINEAR_ATTN only, Phase 20 Step 5)
    QuantWeight W_qkv;                                // attn_qkv: [Q (n_group*d_state) | K (same) | V (d_inner)]
    QuantWeight W_z;                                  // attn_gate: output gate z [d_inner]
    QuantWeight W_alpha, W_beta;                      // ssm_alpha / ssm_beta: [dt_rank] per token
    QuantWeight W_ssm_out;                            // ssm_out: d_inner -> n_embd
    const float* ssm_conv1d = nullptr;                // [conv_dim][d_conv] F32, d_conv contiguous per channel
    const float* ssm_dt = nullptr;                    // [dt_rank] F32, ssm_dt.bias
    const float* ssm_a = nullptr;                     // [dt_rank] F32, -exp(A_log)
    const float* ssm_norm = nullptr;                  // [d_state] F32, per-head gated RMSNorm gamma

    // Mixture of experts (qwen35moe, Phase 20 Step 6): replaces the dense FFN when moe
    bool moe = false;
    QuantWeight W_router;                             // ffn_gate_inp: F32 [n_expert x n_embd], router logits
    QuantWeight We_gate, We_up;                       // ffn_{gate,up}_exps: [n_expert * n_ff_exp x n_embd]
    QuantWeight We_down;                              // ffn_down_exps: [n_expert * n_embd x n_ff_exp]
    const float* shexp_gate = nullptr;                // ffn_gate_inp_shexp: [n_embd] F32, shared-expert gate
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
    int n_ff = 0;                                       // FFN width (feed_forward_length); 0 for MoE models
    int n_expert = 0;                                   // MoE: experts per layer (expert_count)
    int n_expert_used = 0;                              // MoE: experts per token (expert_used_count)
    int n_ff_exp = 0;                                   // MoE: expert FFN width (expert_feed_forward_length)
    int n_ff_shexp = 0;                                 // MoE: shared expert FFN width
    int n_rot = 0;                                      // rotary dims per head (rope.dimension_count)
    float rope_freq_base = 10000.0f;                    // rope.freq_base
    int vocab_size = 0;                                 // vocabulary size
    int n_ctx = 0;                                      // context window size

    // Gated DeltaNet hparams (ssm.*). V heads = dt_rank, Q/K heads = n_group, head size = d_state.
    int ssm_d_conv = 0;                                 // conv kernel width (4)
    int ssm_d_state = 0;                                // head size of q/k/v and of the square state (128)
    int ssm_n_group = 0;                                // Q/K heads (16)
    int ssm_dt_rank = 0;                                // V heads (32)
    int ssm_d_inner = 0;                                // V width = dt_rank * d_state (4096)

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

    int n_past = 0;                                     // tokens decoded in the current sequence (next position)
    uint64_t seed = 0;                                  // inference_generate sampler seed

    double eps = 1e-5;                                  // normalization epsilon
    int full_attention_interval = 0;                   // informational only; use LayerWeights::kind
    int n_kv_layers = 0;                                // full-attention layers, i.e. KV cache layers
};

#endif // INFERENCE_INTERNAL_H
