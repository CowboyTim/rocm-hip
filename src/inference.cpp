// inference.cpp -- Phase 4: C API implementation for zero-copy ROCmFP4 inference engine.
// Thin FFI wrappers around the C++ HIP implementation. All state in inference_model_t.

#include "inference.h"
#include "inference_internal.h"
#include "gguf_mmap.h"
#include "quant_types.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "sampler.h"
#include "qwen_model.h"
#include <hip/hip_runtime.h>

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

// Global RNG seed for reproducible sampling

// Thread-local error buffer
static thread_local char g_last_error[1024] = {0};

static void set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
}

static void clear_error() { g_last_error[0] = 0; }

// All internal types (QuantWeight, LayerWeights, inference_model) are
// defined in inference_internal.h which is included above.

// Pin the whole GGUF mapping once, read-only. Every weight and norm the kernels read lives inside it,
// and on UMA the registered host pointer is the device pointer, so nothing is copied.
static bool pin_model_memory(inference_model* m) {
    void* base = (void*)gguf_map_base(m->gguf, &m->map_size);
    if (!base) return false;
    hipError_t e = hipHostRegister(base, m->map_size, hipHostRegisterReadOnly);
    if (e != hipSuccess) {
        fprintf(stderr, "[pin_model] hipHostRegister(GGUF map, %zu bytes, ReadOnly) failed: %s\n",
                m->map_size, hipGetErrorString(e));
        return false;
    }
    m->map_base = base;
    m->map_pinned = true;
    return true;
}

static void unpin_model_memory(inference_model* m) {
    if (m->map_pinned) (void)hipHostUnregister((void*)m->map_base);
    m->map_pinned = false;
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

// ======================== WEIGHT LOADING (Phase 20 Step 3c) ========================
// Weights are not copied or repacked. Each QuantWeight points at its tensor inside the GGUF mapping,
// which is pinned once (read-only) so qmatvec kernels read the native ggml blocks in place.

// Bind a 2-D weight tensor. Missing tensors fail only when required.
static bool bind_weight(gguf_model_t* gguf, const char* name, QuantWeight& w, bool required) {
    const int idx = find_tensor_index(gguf, name);
    if (idx < 0) {
        if (required) set_error("missing tensor %s", name);
        return !required;
    }
    const int type = gguf_tensor_type(gguf, idx);
    const int64_t* ne = gguf_tensor_ne(gguf, idx);
    const int nd = gguf_tensor_dims(gguf, idx);
    if (!qt_get_traits(type)) { set_error("%s: unsupported tensor type %d", name, type); return false; }
    // A 3-D tensor (MoE experts, ne = {K, rows per expert, n_expert}) is bound as one matrix of all experts'
    // rows: expert e is the row range [e * ne[1], (e + 1) * ne[1])
    if (!ne || nd < 1 || nd > 3) { set_error("%s: expected a 2-D or 3-D tensor (n_dims=%d)", name, nd); return false; }
    const int64_t K = ne[0], N = (nd >= 2 ? ne[1] : 1) * (nd == 3 ? ne[2] : 1);
    const size_t rb = qt_row_size(type, K);
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(gguf, name, &bytes);
    if (!p || rb == 0 || bytes != rb * (size_t)N) {
        set_error("%s: size %zu does not match %lld rows x %zu bytes (type %d)", name, bytes, (long long)N, rb, type);
        return false;
    }
    w.type = type;
    w.N = (int)N;
    w.K = (int)K;
    w.row_bytes = rb;
    w.data = p;
    return true;
}

// Bind an F32 norm vector of n values. Missing tensors fail only when required.
static bool bind_norm(gguf_model_t* gguf, const char* name, int n, const float*& out, bool required) {
    const int idx = find_tensor_index(gguf, name);
    if (idx < 0) {
        if (required) set_error("missing tensor %s", name);
        return !required;
    }
    const int64_t* ne = gguf_tensor_ne(gguf, idx);
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(gguf, name, &bytes);
    if (gguf_tensor_type(gguf, idx) != QT_F32 || !ne || ne[0] != n || !p || bytes != (size_t)n * sizeof(float)) {
        set_error("%s: expected F32[%d] (type %d, %zu bytes)", name, n, gguf_tensor_type(gguf, idx), bytes);
        return false;
    }
    out = (const float*)p;
    return true;
}

// Bind an F32 tensor of ne0 x ne1 values (ne0 contiguous), e.g. ssm_conv1d [d_conv x conv_dim]
static bool bind_f32(gguf_model_t* gguf, const char* name, int ne0, int ne1, const float*& out) {
    const int idx = find_tensor_index(gguf, name);
    if (idx < 0) { set_error("missing tensor %s", name); return false; }
    const int64_t* ne = gguf_tensor_ne(gguf, idx);
    const int nd = gguf_tensor_dims(gguf, idx);
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(gguf, name, &bytes);
    if (gguf_tensor_type(gguf, idx) != QT_F32 || !ne || ne[0] != ne0 || (nd > 1 ? ne[1] : 1) != ne1 || !p ||
        bytes != (size_t)ne0 * ne1 * sizeof(float)) {
        set_error("%s: expected F32[%d x %d] (type %d, %zu bytes)", name, ne0, ne1, gguf_tensor_type(gguf, idx), bytes);
        return false;
    }
    out = (const float*)p;
    return true;
}

static bool check_shape(const char* what, int layer, const QuantWeight& w, int N, int K) {
    if (w.N == N && w.K == K) return true;
    set_error("blk.%d.%s: shape %dx%d, expected %dx%d (rows x cols)", layer, what, w.N, w.K, N, K);
    return false;
}

// Bind every per-layer tensor, classify each block, and check shapes against the hyperparameters
static bool load_model_weights(inference_model* m) {
    const int E = m->n_embd, HD = m->head_dim, HQ = m->n_head, HKV = m->n_head_kv, FF = m->n_ff;
    m->layers.resize(m->n_layer);
    char name[128];
    auto nm = [&](int i, const char* t) { snprintf(name, sizeof(name), "blk.%d.%s", i, t); return name; };

    for (int i = 0; i < m->n_layer; ++i) {
        LayerWeights& L = m->layers[i];

        // --- Classify the block by the tensors it carries (Phase 20 Step 2) ---
        // ssm_alpha.weight => Gated DeltaNet linear attention; attn_q.weight => full attention. MTP (nextn) blocks
        // sit past n_layer and are never loaded here.
        const bool linear = find_tensor_index(m->gguf, nm(i, "ssm_alpha.weight")) >= 0;
        const bool full = find_tensor_index(m->gguf, nm(i, "attn_q.weight")) >= 0;
        if (linear == full) {
            set_error("blk.%d: cannot classify (ssm_alpha=%d attn_q=%d)", i, linear, full);
            return false;
        }
        L.kind = linear ? LAYER_LINEAR_ATTN : LAYER_FULL_ATTN;
        if (full) L.kv_slot = m->n_kv_layers++;  // only attention layers get KV cache space

        if (!bind_norm(m->gguf, nm(i, "attn_norm.weight"), E, L.attn_norm, true)) return false;
        if (!bind_norm(m->gguf, nm(i, "post_attention_norm.weight"), E, L.ffn_norm, true)) return false;

        if (L.kind == LAYER_LINEAR_ATTN) {
            const int DS = m->ssm_d_state, NG = m->ssm_n_group, NV = m->ssm_dt_rank, DI = m->ssm_d_inner;
            const int conv_dim = 2 * NG * DS + DI;
            if (m->ssm_d_conv < 2 || DS <= 0 || NG <= 0 || NV <= 0 || NV % NG != 0 || DI != NV * DS) {
                set_error("blk.%d: bad ssm hparams (d_conv %d d_state %d n_group %d dt_rank %d d_inner %d)", i,
                          m->ssm_d_conv, DS, NG, NV, DI);
                return false;
            }
            if (!bind_weight(m->gguf, nm(i, "attn_qkv.weight"), L.W_qkv, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "attn_gate.weight"), L.W_z, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ssm_alpha.weight"), L.W_alpha, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ssm_beta.weight"), L.W_beta, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ssm_out.weight"), L.W_ssm_out, true)) return false;
            if (!bind_f32(m->gguf, nm(i, "ssm_conv1d.weight"), m->ssm_d_conv, conv_dim, L.ssm_conv1d)) return false;
            if (!bind_norm(m->gguf, nm(i, "ssm_dt.bias"), NV, L.ssm_dt, true)) return false;
            if (!bind_norm(m->gguf, nm(i, "ssm_a"), NV, L.ssm_a, true)) return false;
            if (!bind_norm(m->gguf, nm(i, "ssm_norm.weight"), DS, L.ssm_norm, true)) return false;
            if (!check_shape("attn_qkv", i, L.W_qkv, conv_dim, E) ||
                !check_shape("attn_gate", i, L.W_z, DI, E) ||
                !check_shape("ssm_alpha", i, L.W_alpha, NV, E) ||
                !check_shape("ssm_beta", i, L.W_beta, NV, E) ||
                !check_shape("ssm_out", i, L.W_ssm_out, E, DI)) return false;
        } else {
            if (!bind_weight(m->gguf, nm(i, "attn_q.weight"), L.Wq, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "attn_k.weight"), L.Wk, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "attn_v.weight"), L.Wv, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "attn_output.weight"), L.Wo, true)) return false;
            if (!bind_norm(m->gguf, nm(i, "attn_q_norm.weight"), HD, L.q_norm, true)) return false;
            if (!bind_norm(m->gguf, nm(i, "attn_k_norm.weight"), HD, L.k_norm, true)) return false;
            // Gated attention: attn_q holds Q and the output gate, head_dim each, interleaved per head
            if (!check_shape("attn_q", i, L.Wq, 2 * HQ * HD, E) ||
                !check_shape("attn_k", i, L.Wk, HKV * HD, E) ||
                !check_shape("attn_v", i, L.Wv, HKV * HD, E) ||
                !check_shape("attn_output", i, L.Wo, E, HQ * HD)) return false;
        }

        // FFN: MoE (router ffn_gate_inp + 3-D expert tensors + a sigmoid-gated shared expert) or dense SwiGLU
        L.moe = find_tensor_index(m->gguf, nm(i, "ffn_gate_inp.weight")) >= 0;
        if (L.moe) {
            const int NE = m->n_expert, FE = m->n_ff_exp, FS = m->n_ff_shexp;
            if (NE <= 0 || m->n_expert_used <= 0 || m->n_expert_used > NE || FE <= 0 || FS <= 0) {
                set_error("blk.%d: bad MoE hparams (experts %d used %d ff_exp %d ff_shexp %d)", i, NE,
                          m->n_expert_used, FE, FS);
                return false;
            }
            if (!bind_weight(m->gguf, nm(i, "ffn_gate_inp.weight"), L.W_router, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_gate_exps.weight"), L.We_gate, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_up_exps.weight"), L.We_up, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_down_exps.weight"), L.We_down, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_gate_shexp.weight"), L.W_gate, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_up_shexp.weight"), L.W_up, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_down_shexp.weight"), L.W_down, true)) return false;
            if (!bind_norm(m->gguf, nm(i, "ffn_gate_inp_shexp.weight"), E, L.shexp_gate, true)) return false;
            if (!check_shape("ffn_gate_inp", i, L.W_router, NE, E) ||
                !check_shape("ffn_gate_exps", i, L.We_gate, NE * FE, E) ||
                !check_shape("ffn_up_exps", i, L.We_up, NE * FE, E) ||
                !check_shape("ffn_down_exps", i, L.We_down, NE * E, FE) ||
                !check_shape("ffn_gate_shexp", i, L.W_gate, FS, E) ||
                !check_shape("ffn_up_shexp", i, L.W_up, FS, E) ||
                !check_shape("ffn_down_shexp", i, L.W_down, E, FS)) return false;
        } else {
            if (!bind_weight(m->gguf, nm(i, "ffn_gate.weight"), L.W_gate, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_up.weight"), L.W_up, true)) return false;
            if (!bind_weight(m->gguf, nm(i, "ffn_down.weight"), L.W_down, true)) return false;
            if (!check_shape("ffn_gate", i, L.W_gate, FF, E) ||
                !check_shape("ffn_up", i, L.W_up, FF, E) ||
                !check_shape("ffn_down", i, L.W_down, E, FF)) return false;
        }
    }
    return true;
}

// ======================== C API IMPLEMENTATION ========================

inference_model_t* inference_load_model(const char* gguf_path) {
    clear_error();
    if (!gguf_path) { set_error("NULL path"); return nullptr; }

    fprintf(stderr, "[LOAD] Starting load_model for %s\n", gguf_path); fflush(stderr);
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

    // head_dim: attention.key_length when present (Qwen3.5: 256, not n_embd/n_head = 160)
    int64_t hd = gguf_arch_int(m->gguf, "attention.key_length", n_head > 0 ? n_embd / n_head : 0);

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

    // block_count includes the trailing MTP (nextn) blocks; they are not part of the decoder stack
    m->n_nextn = (int)gguf_arch_int(m->gguf, "nextn_predict_layers", 0);
    m->n_layer = (int)n_layer - m->n_nextn;
    m->n_embd = (int)n_embd;
    m->n_head = (int)n_head;
    m->n_head_kv = (int)n_head_kv;
    m->head_dim = (int)hd;
    m->vocab_size = (int)vs;
    m->n_ff = (int)gguf_arch_int(m->gguf, "feed_forward_length", 0);
    m->n_expert = (int)gguf_arch_int(m->gguf, "expert_count", 0);
    m->n_expert_used = (int)gguf_arch_int(m->gguf, "expert_used_count", 0);
    m->n_ff_exp = (int)gguf_arch_int(m->gguf, "expert_feed_forward_length", 0);
    m->n_ff_shexp = (int)gguf_arch_int(m->gguf, "expert_shared_feed_forward_length", 0);
    m->n_rot = (int)gguf_arch_int(m->gguf, "rope.dimension_count", hd);
    m->rope_freq_base = gguf_arch_float(m->gguf, "rope.freq_base", 10000.0f);
    m->eps = gguf_arch_float(m->gguf, "attention.layer_norm_rms_epsilon", 1e-6f);
    m->full_attention_interval = (int)gguf_arch_int(m->gguf, "full_attention_interval", 0);
    m->ssm_d_conv = (int)gguf_arch_int(m->gguf, "ssm.conv_kernel", 0);
    m->ssm_d_state = (int)gguf_arch_int(m->gguf, "ssm.state_size", 0);
    m->ssm_n_group = (int)gguf_arch_int(m->gguf, "ssm.group_count", 0);
    m->ssm_dt_rank = (int)gguf_arch_int(m->gguf, "ssm.time_step_rank", 0);
    m->ssm_d_inner = (int)gguf_arch_int(m->gguf, "ssm.inner_size", 0);
    fprintf(stderr, "[LOAD] arch=%s blocks=%lld (decoder %d + nextn %d) E=%d heads=%d/%d head_dim=%d ff=%d "
            "n_rot=%d rope_base=%g eps=%g\n", arch ? arch : "?", (long long)n_layer, m->n_layer, m->n_nextn,
            m->n_embd, m->n_head, m->n_head_kv, m->head_dim, m->n_ff, m->n_rot, m->rope_freq_base, m->eps);

    m->n_ctx = (int)ctx_len;

    // Token embeddings: any decodable type; rows are decoded per token in the forward pass
    if (!bind_weight(m->gguf, "token_embd.weight", m->token_embd, true) ||
        m->token_embd.K != m->n_embd || m->token_embd.N != m->vocab_size) {
        if (m->token_embd.data)
            set_error("token_embd: shape %dx%d, expected %dx%d", m->token_embd.N, m->token_embd.K,
                      m->vocab_size, m->n_embd);
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }

    // Per-layer weights and norms
    if (!load_model_weights(m)) {
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }

    // LM head: output.weight when present, else tied to token_embd
    if (!bind_weight(m->gguf, "output.weight", m->output, false) ||
        !bind_norm(m->gguf, "output_norm.weight", m->n_embd, m->output_norm, true)) {
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }
    if (!m->output.data) m->output = m->token_embd;
    if (m->output.K != m->n_embd || m->output.N != m->vocab_size) {
        set_error("output: shape %dx%d, expected %dx%d", m->output.N, m->output.K, m->vocab_size, m->n_embd);
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }
    fprintf(stderr, "[LOAD] token_embd type %d, LM head type %d%s\n", m->token_embd.type, m->output.type,
            m->output.data == m->token_embd.data ? " (tied)" : "");

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
    fprintf(stderr, "[CLEANUP] Starting model free...\n"); fflush(stderr);
    if (m->tokenizer) { fprintf(stderr, "[CLEANUP] Freeing tokenizer\n"); fflush(stderr); tokenizer_free(m->tokenizer); }
    // IMPORTANT: unpin must happen BEFORE qwen_model_free since unpin walks layer weights
    // whose device pointers point into the layer scratch buffers we're about to free.
    fprintf(stderr, "[CLEANUP] Unpinning model memory\n"); fflush(stderr);
    unpin_model_memory(m);
    if (m->kvc) { fprintf(stderr, "[CLEANUP] Freeing KV cache\n"); fflush(stderr); kv_cache_free(m->kvc); }
    if (m->qwen) { fprintf(stderr, "[CLEANUP] Freeing qwen model\n"); fflush(stderr); qwen_model_free(m->qwen); }
    if (m->gguf) { fprintf(stderr, "[CLEANUP] Closing GGUF\n"); fflush(stderr); gguf_mmap_close(m->gguf); }
    fprintf(stderr, "[CLEANUP] Deleting model struct\n"); fflush(stderr);
    delete m;
    fprintf(stderr, "[CLEANUP] Model free complete\n"); fflush(stderr);
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
    // One KV layer per full-attention layer (LayerWeights::kv_slot); DeltaNet layers keep recurrent state instead
    kv_cache_config_t cfg{m->n_kv_layers > 0 ? m->n_kv_layers : 1, n_ctx, m->n_embd, m->head_dim, m->n_head,
                          m->n_head_kv};
    m->kvc = kv_cache_create("/dev/shm/inference_kvc.bin", &cfg);
    if (!m->kvc) { set_error("kv_cache_create failed"); return -1; }
    m->n_ctx = n_ctx;
    m->n_past = 0;
    // Create Qwen model for forward passes
    fprintf(stderr, "[KV_INIT] About to create qwen model\n"); fflush(stderr);
    m->qwen = qwen_model_create(m);
    if (!m->qwen) { set_error("qwen_model_create failed"); return -1; }
    fprintf(stderr, "[KV_INIT] Qwen model created at %p\n", (void*)m->qwen);
    if (m->lm_head_q8 && qwen_model_set_lm_head_q8(m->qwen, 1) < 0) {
        set_error("LM head Q8_0 copy failed (details on stderr)");
        return -1;
    }
    return 0;
}

int inference_set_lm_head_q8(inference_model_t* m, int enable) {
    if (!m) return INFERENCE_ERR_INVALID_ARG;
    const int t = m->output.type;
    if (enable && t != QT_F32 && t != QT_F16 && t != QT_BF16) return 1;  // already quantized
    m->lm_head_q8 = enable != 0;
    if (m->qwen && qwen_model_set_lm_head_q8(m->qwen, m->lm_head_q8) < 0) {
        set_error("LM head Q8_0 copy failed (details on stderr)");
        m->lm_head_q8 = 0;
        return INFERENCE_ERR_OUT_OF_MEMORY;
    }
    return 0;
}

void inference_kv_cache_free(inference_model_t* m) {
    if (m && m->kvc) { kv_cache_free(m->kvc); m->kvc = nullptr; }
}

qwen_model_t* inference_get_qwen_model(inference_model_t* m) {
    if (!m) return nullptr;
    return m->qwen;
}

void inference_set_seed(inference_model_t* m, unsigned int seed) {
    if (m) m->seed = seed;
}

void inference_reset(inference_model_t* m) {
    if (m) m->n_past = 0;
}

int inference_n_past(const inference_model_t* m) {
    return m ? m->n_past : 0;
}

// One forward pass for token at position n_past (the next position of the current sequence)
static int forward_next(inference_model_t* m, int token_id) {
    if (!m || !m->kvc || !m->qwen) { set_error("Invalid state (call inference_kv_cache_init first)"); return -1; }
    if (m->n_past >= m->n_ctx) { set_error("context full (%d tokens)", m->n_ctx); return INFERENCE_ERR_CONTEXT_FULL; }
    if (qwen_model_forward(m->qwen, token_id, m->n_past, nullptr) != 0) { set_error("qwen_model_forward failed"); return -1; }
    ++m->n_past;
    return 0;
}

int inference_prefill(inference_model_t* m, const int32_t* tokens, int n) {
    clear_error();
    if (!m || !m->kvc || !m->qwen || !tokens || n <= 0) { set_error("Invalid state or arguments"); return -1; }
    if (m->n_past + n > m->n_ctx) { set_error("context full (%d + %d > %d tokens)", m->n_past, n, m->n_ctx); return INFERENCE_ERR_CONTEXT_FULL; }
    if (qwen_model_prefill(m->qwen, (const int*)tokens, n, m->n_past, nullptr) != 0) { set_error("qwen_model_prefill failed"); return -1; }
    m->n_past += n;
    return 0;
}

int inference_decode_token(inference_model_t* m, int token_id, int* next_token_out) {
    clear_error();
    if (!next_token_out) { set_error("Invalid args"); return -1; }
    const int rc = forward_next(m, token_id);
    if (rc != 0) return rc;
    const float* logits = qwen_model_logits(m->qwen);
    int best = 0;
    for (int i = 1; i < m->vocab_size; ++i)
        if (logits[i] > logits[best]) best = i;
    *next_token_out = best;
    return 0;
}

int inference_get_logits(inference_model_t* m, int* vocab_ids, float* logits) {
    clear_error();
    if (!m || !m->qwen || !vocab_ids || !logits) { set_error("Invalid args"); return -1; }
    if (m->n_past == 0) { set_error("no token decoded yet"); return -1; }
    memcpy(logits, qwen_model_logits(m->qwen), (size_t)m->vocab_size * sizeof(float));
    for (int i = 0; i < m->vocab_size; ++i) vocab_ids[i] = i;
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

int inference_generate_ex(inference_model_t* model, const int32_t* prompt_ids, int prompt_len, int max_new_tokens,
                          const sampler_params* params, int32_t* out_ids, int* out_len) {
    clear_error();
    if (!model || !prompt_ids || !out_ids || !out_len || prompt_len <= 0 || max_new_tokens <= 0) {
        set_error("Invalid arguments");
        return INFERENCE_ERR_INVALID_ARG;
    }
    *out_len = 0;
    sampler_params prm = params ? *params : sampler_default_params();
    sampler_t* smp = sampler_create(&prm);
    const int eos = model->tokenizer ? tokenizer_eos_id(model->tokenizer) : -1;

    // New sequence: prefill the prompt in batches, then sample from the logits in place (no copy)
    model->n_past = 0;
    int rc = inference_prefill(model, prompt_ids, prompt_len);
    for (int i = 0; i < prompt_len; ++i) sampler_accept(smp, prompt_ids[i]);
    for (int step = 0; step < max_new_tokens && rc == 0; ++step) {
        const int next = sampler_sample(smp, qwen_model_logits(model->qwen), model->vocab_size);
        if (next == eos) break;
        out_ids[(*out_len)++] = next;
        sampler_accept(smp, next);
        if (step + 1 < max_new_tokens) rc = forward_next(model, next);
    }
    sampler_free(smp);
    return rc == INFERENCE_ERR_CONTEXT_FULL ? INFERENCE_OK : rc;  // a full context just ends generation
}

// ---- Streaming generation

inference_gen_options inference_gen_default_options(void) {
    inference_gen_options o;
    memset(&o, 0, sizeof(o));
    o.sampler = sampler_default_params();
    o.max_new_tokens = 256;
    return o;
}

int inference_token_is_eog(const inference_model_t* m, int32_t token) {
    return m && m->tokenizer ? tokenizer_is_eog(m->tokenizer, token) : 0;
}

int inference_detokenize(const inference_model_t* m, const int32_t* ids, int n, char* out, int max_len) {
    if (!m || !m->tokenizer) { set_error("no tokenizer"); return INFERENCE_ERR_INVALID_STATE; }
    const int len = tokenizer_detokenize(m->tokenizer, (const int*)ids, n, out, max_len);
    if (len < 0) { set_error("Invalid arguments"); return INFERENCE_ERR_INVALID_ARG; }
    return len;
}

// Bytes at the end of s that begin a UTF-8 character still missing continuation bytes
static size_t utf8_incomplete_tail(const std::string& s) {
    for (size_t k = 1; k <= 4 && k <= s.size(); ++k) {
        const unsigned char c = (unsigned char)s[s.size() - k];
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: look further back for the lead byte
        const size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        return need > k ? k : 0;
    }
    return 0;
}

// Length of the longest suffix of s that is a proper prefix of one of the stop strings (it may still become one)
static size_t stop_prefix_tail(const std::string& s, const std::vector<std::string>& stops) {
    size_t best = 0;
    for (const std::string& st : stops)
        for (size_t len = std::min(st.size() - 1, s.size()); len > best; --len)
            if (s.compare(s.size() - len, len, st, 0, len) == 0) {
                best = len;
                break;
            }
    return best;
}

int inference_generate_stream(inference_model_t* m, const int32_t* prompt_ids, int n_prompt,
                              const inference_gen_options* options, int32_t* out_ids, int max_out, int* n_out,
                              inference_stop_reason* reason) {
    clear_error();
    if (n_out) *n_out = 0;
    if (reason) *reason = INFERENCE_STOP_NONE;
    if (!m || !m->qwen || !m->tokenizer || n_prompt < 0 || (n_prompt > 0 && !prompt_ids) || max_out < 0 ||
        (max_out > 0 && !out_ids)) {
        set_error("Invalid state or arguments");
        return INFERENCE_ERR_INVALID_ARG;
    }
    const inference_gen_options o = options ? *options : inference_gen_default_options();
    if (n_prompt > 0) {
        const int rc = inference_prefill(m, prompt_ids, n_prompt);
        if (rc != 0) return rc;
    } else if (m->n_past == 0) {
        set_error("nothing to continue: empty prompt and an empty sequence");
        return INFERENCE_ERR_INVALID_ARG;
    }
    std::vector<std::string> stops;
    for (int i = 0; i < o.n_stop_strings; ++i)
        if (o.stop_strings && o.stop_strings[i] && o.stop_strings[i][0]) stops.emplace_back(o.stop_strings[i]);

    sampler_t* smp = sampler_create(&o.sampler);
    for (int i = 0; i < n_prompt; ++i) sampler_accept(smp, prompt_ids[i]);
    std::string pending;  // generated text not delivered yet
    std::vector<char> buf(256);
    inference_stop_reason stop = INFERENCE_STOP_NONE;
    int emitted = 0, rc = 0;
    while (stop == INFERENCE_STOP_NONE) {
        if (emitted >= o.max_new_tokens) { stop = INFERENCE_STOP_MAX_TOKENS; break; }
        const int next = sampler_sample(smp, qwen_model_logits(m->qwen), m->vocab_size);
        if (!o.ignore_eog && tokenizer_is_eog(m->tokenizer, next)) { stop = INFERENCE_STOP_EOG; break; }
        sampler_accept(smp, next);
        if (emitted < max_out) out_ids[emitted] = next;
        ++emitted;

        int len = tokenizer_detokenize(m->tokenizer, &next, 1, buf.data(), (int)buf.size());
        if (len >= (int)buf.size()) {
            buf.resize((size_t)len + 1);
            len = tokenizer_detokenize(m->tokenizer, &next, 1, buf.data(), (int)buf.size());
        }
        pending.append(buf.data(), (size_t)std::max(len, 0));

        // Deliver what can no longer become part of a stop string or of a split UTF-8 character
        size_t ready, cut = std::string::npos;
        for (const std::string& st : stops) cut = std::min(cut, pending.find(st));
        if (cut != std::string::npos) {
            ready = cut;
            stop = INFERENCE_STOP_STRING;
        } else {
            ready = pending.size() - std::max(utf8_incomplete_tail(pending), stop_prefix_tail(pending, stops));
        }
        if (o.on_token && o.on_token(o.user, next, pending.data(), (int)ready) != 0 && stop == INFERENCE_STOP_NONE)
            stop = INFERENCE_STOP_CALLBACK;
        pending.erase(0, stop == INFERENCE_STOP_STRING ? pending.size() : ready);

        // Every emitted token joins the sequence, so a later call continues right after it
        rc = forward_next(m, next);
        if (rc == INFERENCE_ERR_CONTEXT_FULL) {
            rc = 0;
            if (stop == INFERENCE_STOP_NONE) stop = INFERENCE_STOP_CONTEXT_FULL;
        }
        if (rc != 0) break;
    }
    if (rc == 0 && o.on_token && !pending.empty()) (void)o.on_token(o.user, -1, pending.data(), (int)pending.size());
    sampler_free(smp);
    if (n_out) *n_out = emitted;
    if (reason) *reason = stop;
    return rc;
}

int inference_generate(inference_model_t* model, const int32_t* prompt_ids, int prompt_len, int max_new_tokens,
                       float temperature, int top_k, float top_p, int32_t* out_ids, int* out_len) {
    sampler_params prm = sampler_default_params();
    prm.temperature = temperature;
    prm.top_k = top_k;
    prm.top_p = top_p;
    prm.min_p = 0.0f;
    prm.seed = model ? model->seed : 0;
    return inference_generate_ex(model, prompt_ids, prompt_len, max_new_tokens, &prm, out_ids, out_len);
}

tokenizer_t* inference_get_tokenizer(const inference_model_t* m) {
    if (!m) return nullptr;
    return m->tokenizer;
}
