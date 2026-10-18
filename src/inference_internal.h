// inference_internal.h -- Complete type definitions shared between
// inference.cpp and qwen_model.cpp. Both translation units must see
// IDENTICAL struct layouts (defined here, not duplicated).

#ifndef INFERENCE_INTERNAL_H
#define INFERENCE_INTERNAL_H

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>

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

// Hyper-connection mixer (qwen4exp): the residual is n_hc parallel streams of n_embd. In place of every RMSNorm,
// a mixer normalizes each stream (one RMSNorm per stream, gamma over all n_hc * n_embd values), gates the result
// with sigmoid(up(SiLU(down(xn) / n_hc))) and averages the streams into the block input; inject (N = n_hc)
// gives the per-stream weights 2 * sigmoid(inject(xn) / n_hc) of the block output added back to each stream.
// The final mixer (output_hc_*) has no inject and replaces output_norm. llama.cpp qwen4exp build_hc_mix.
struct HcMix {
    const float* norm = nullptr;                      // [n_hc * n_embd] F32
    QuantWeight down;                                 // [hc_rank x n_hc * n_embd]
    QuantWeight up;                                   // [n_hc * n_embd x hc_rank]
    QuantWeight inject;                               // [n_hc x n_hc * n_embd]; none for the final mixer
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

    // Hyper-connections (qwen4exp): mixers before the token mixer and before the FFN (attn_norm / ffn_norm unset)
    HcMix hc_attn, hc_ffn;
    // PLE n-gram hash embedding (qwen4exp, on the layers in ple.layers): applied to the residual streams before
    // this layer's mixer (llama.cpp qwen4exp build_ple)
    bool ple = false;
    QuantWeight ple_key;                              // [n_hc * n_embd x n_embd] (input: the gathered rows)
    QuantWeight ple_value;                            // [n_embd x n_embd]
    const float* ple_norm_key = nullptr;              // [n_hc * n_embd] F32 each
    const float* ple_norm_query = nullptr;
    const float* ple_norm_conv = nullptr;
    const uint16_t* ple_conv1d = nullptr;             // F16 [n_hc * n_embd][ple_conv_kernel], taps contiguous
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

    // qwen4exp (Qwen3.8-Flash-Next). n_hc == 0: a plain residual stream (qwen35 / qwen35moe).
    int n_hc = 0;                                       // hyper_connection.count: parallel residual streams
    int hc_rank = 0;                                    // hyper_connection.low_rank
    HcMix hc_head;                                      // output_hc_*: the final mixer, in place of output_norm
    bool gdn_sigmoid_gate = false;                      // DeltaNet output gate sigmoid(z) instead of SiLU(z)
    // PLE n-gram hash embeddings: a token's row of head h is (t0*m0 ^ t1*m1 ^ .. ^ t_{n-1}*m_{n-1}) % vocab[h] +
    // offset[h] for its n-gram of the n - 1 previous tokens (n = 2 .. ple_ngram, ple_per_ngram heads each); the
    // rows (ple_dim values each) of all heads, concatenated, are the PLE input
    int ple_ngram = 0, ple_per_ngram = 0, ple_conv = 0, ple_dim = 0, ple_eos = -1;
    std::vector<uint64_t> ple_mult, ple_offset, ple_vocab;
    QuantWeight ple_table;                              // per_layer_token_embd [rows x ple_dim], rows read on the CPU

    inference_kv_cache_t* kvc = nullptr;               // KV cache manager handle
    void* scratch_base = nullptr;                      // temporary scratch buffer base
    size_t scratch_bytes = 0;                          // scratch buffer size in bytes

    tokenizer_t* tokenizer = nullptr;                  // BPE tokenizer instance
    QwenModel* qwen = nullptr;                      // Qwen model handle (set during kv_init)

    int n_past = 0;                                     // tokens decoded in the current sequence (next position)
    std::vector<int32_t> seq;                           // those tokens (seq.size() == n_past)
    struct Checkpoint { int pos, slot; uint64_t age; };
    std::vector<Checkpoint> ckpts;                      // saved recurrent states, by position in seq
    uint64_t ckpt_clock = 0;
    int last_reused = 0, last_prefilled = 0;            // inference_last_prompt_stats
    // KV block store (Phase 22b, inference_store_open): blocks published as <chain hash>.kvb, recurrent state
    // snapshots at block boundaries as <chain hash>.rst
    struct Hash { uint8_t b[32]; bool operator==(const Hash& o) const { return memcmp(b, o.b, 32) == 0; } };
    std::string store_dir;                              // empty: no store
    int store_dirfd = -1;
    int store_snapshot_every = 0;                       // snapshot every N block boundaries (0: only prompt ends)
    int store_snapshot_at = -1;                         // block boundary of the current prompt's end: snapshot too
    Hash store_root{};                                  // chain hash before block 0 (model and layout identity)
    std::vector<Hash> chain;                            // chain[i]: hash after block i of seq
    size_t published = 0;                               // full blocks of seq that are in the store
    long long st_published = 0, st_attached = 0, st_snap_saved = 0, st_snap_restored = 0, st_tokens = 0;
    std::unordered_map<std::string, std::string> chat_templates;  // inference_chat_template, by name
    int lm_head_q8 = 0;                                 // inference_set_lm_head_q8: use an int8 copy of the LM head
    uint64_t seed = 0;                                  // inference_generate sampler seed

    double eps = 1e-5;                                  // normalization epsilon
    int full_attention_interval = 0;                   // informational only; use LayerWeights::kind
    int n_kv_layers = 0;                                // full-attention layers, i.e. KV cache layers
};

#endif // INFERENCE_INTERNAL_H
