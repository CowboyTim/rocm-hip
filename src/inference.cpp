// inference.cpp -- Phase 4: C API implementation for zero-copy ROCmFP4 inference engine.
// Thin FFI wrappers around the C++ HIP implementation. All state in inference_model_t.

#include "inference_log.h"
#include "inference.h"
#include "inference_internal.h"
#include "gguf_mmap.h"
#include "quant_types.h"
#include "kv_cache.h"
#include "tokenizer.h"
#include "sampler.h"
#include "qwen_model.h"
#include "sha256.h"
#include <hip/hip_runtime.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdarg>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
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

static void seq_reset(inference_model_t* m);
static void free_seqs(inference_model_t* m);

// All internal types (QuantWeight, LayerWeights, inference_model) are
// defined in inference_internal.h which is included above.

// Pin the GGUF mappings once, read-only (every file of a split model). Every weight and norm the kernels read
// lives inside them, and on UMA the registered host pointer is the device pointer, so nothing is copied. Tensors
// only the CPU reads are left out: the token embeddings (when the LM head is not tied to them) and the PLE table
// (qwen4exp: 35.8 GiB, 16 rows per token), so a model bigger than what may be pinned can still run.
static bool pin_gguf(gguf_model_t* gguf, std::vector<const QuantWeight*> cpu_only_w,
                     std::vector<std::pair<void*, size_t>>& pinned, size_t& pinned_bytes, size_t& unpinned_bytes) {
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    std::vector<std::pair<const char*, const char*>> skip;  // whole pages inside CPU-only tensors
    for (const QuantWeight* w : cpu_only_w) {
        if (!w || !w->data) continue;
        const uintptr_t b = ((uintptr_t)w->data + page - 1) / page * page;
        const uintptr_t e = ((uintptr_t)w->data + w->row_bytes * (size_t)w->N) / page * page;
        if (e > b) skip.push_back({(const char*)b, (const char*)e});
    }
    std::sort(skip.begin(), skip.end());
    pinned_bytes = unpinned_bytes = 0;
    for (int i = 0; i < gguf_n_maps(gguf); ++i) {
        size_t size = 0;
        const char* base = (const char*)gguf_map_at(gguf, i, &size);
        const char* cur = base;
        auto pin = [&](const char* end) {
            if (end <= cur) return true;
            hipError_t e = hipHostRegister((void*)cur, (size_t)(end - cur), hipHostRegisterReadOnly);
            if (e != hipSuccess) {
                INFER_LOG("[pin_model] hipHostRegister(GGUF map %d, %zu bytes, ReadOnly) failed: %s\n", i,
                          (size_t)(end - cur), hipGetErrorString(e));
                return false;
            }
            pinned.push_back({(void*)cur, (size_t)(end - cur)});
            pinned_bytes += (size_t)(end - cur);
            return true;
        };
        for (const auto& r : skip) {
            if (r.first < base || r.second > base + size) continue;
            if (!pin(r.first)) return false;
            unpinned_bytes += (size_t)(r.second - r.first);
            cur = r.second;
        }
        if (!pin(base + size)) return false;
    }
    INFER_LOG("[pin_model] %d file(s): %zu MB registered, %zu MB left to the CPU\n", gguf_n_maps(gguf),
              pinned_bytes >> 20, unpinned_bytes >> 20);
    return true;
}

static void unpin(std::vector<std::pair<void*, size_t>>& pinned) {
    for (const auto& r : pinned) (void)hipHostUnregister(r.first);
    pinned.clear();
}

static bool pin_model_memory(inference_model* m) {
    if (m->stage.base) {  // every tensor the GPU reads is in the stage: pin it, leave the GGUF mapping to the CPU
        const hipError_t e = hipHostRegister(m->stage.base, m->stage.bytes, hipHostRegisterReadOnly);
        if (e != hipSuccess) {
            INFER_LOG("[pin_model] hipHostRegister(stage, %zu bytes) failed: %s\n", m->stage.bytes, hipGetErrorString(e));
            return false;
        }
        m->pinned.push_back({m->stage.base, m->stage.bytes});
        m->pinned_bytes = m->stage.bytes;
        INFER_LOG("[pin_model] stage: %zu MB registered\n", m->pinned_bytes >> 20);
        return true;
    }
    std::vector<const QuantWeight*> cpu_only = {&m->ple_table};
    if (m->output.data != m->token_embd.data) cpu_only.push_back(&m->token_embd);
    return pin_gguf(m->gguf, cpu_only, m->pinned, m->pinned_bytes, m->unpinned_bytes);
}

static void unpin_model_memory(inference_model* m) {
    unpin(m->pinned);
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

// Bind a hyper-connection mixer <prefix>_{norm,down,up,inject}.weight (inject only with_inject)
static bool bind_hc_mix(inference_model* m, gguf_model_t* g, const char* prefix, HcMix& h, bool with_inject) {
    const int HCD = m->n_hc * m->n_embd, R = m->hc_rank;
    char name[160];
    auto nm = [&](const char* t) { snprintf(name, sizeof(name), "%s_%s.weight", prefix, t); return name; };
    if (!bind_norm(g, nm("norm"), HCD, h.norm, true) || !bind_weight(g, nm("down"), h.down, true) ||
        !bind_weight(g, nm("up"), h.up, true) ||
        (with_inject && !bind_weight(g, nm("inject"), h.inject, true)))
        return false;
    if (h.down.N != R || h.down.K != HCD || h.up.N != HCD || h.up.K != R ||
        (with_inject && (h.inject.N != m->n_hc || h.inject.K != HCD))) {
        set_error("%s: hyper-connection shapes do not match n_hc %d x n_embd %d, rank %d", prefix, m->n_hc, m->n_embd,
                  R);
        return false;
    }
    return true;
}

// A numeric metadata array as uint64 values (U32 / I32 / U64 / I64 elements). Returns false if absent.
static bool kv_u64_array(gguf_model_t* gguf, const char* key, std::vector<uint64_t>& out) {
    int32_t type = -1;
    const void* data = nullptr;
    size_t n = 0;
    if (gguf_kv_get_array(gguf, key, &type, &data, &n) != 0) return false;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        switch (type) {
            case 4: out[i] = ((const uint32_t*)data)[i]; break;
            case 5: out[i] = (uint64_t)(int64_t)((const int32_t*)data)[i]; break;
            case 10: case 11: memcpy(&out[i], (const char*)data + 8 * i, 8); break;
            default: return false;
        }
    }
    return true;
}

// qwen4exp hyperparameters (hyper-connections, PLE) and its global tensors; nothing for other architectures
static bool load_qwen4exp(inference_model* m, std::vector<int>& ple_layers) {
    const std::string a = m->model_name + ".";
    m->n_hc = (int)gguf_arch_int(m->gguf, "hyper_connection.count", 0);
    if (m->n_hc <= 0) return true;
    m->hc_rank = (int)gguf_arch_int(m->gguf, "hyper_connection.low_rank", 0);
    m->gdn_sigmoid_gate = true;
    if (m->hc_rank <= 0) { set_error("hyper_connection.low_rank missing"); return false; }
    if (!bind_hc_mix(m, m->gguf, "output_hc", m->hc_head, false)) return false;
    // QSA sparse attention is not implemented: only dense attention (every compress ratio 0, as released)
    std::vector<uint64_t> ratios;
    if (kv_u64_array(m->gguf, (a + "attention.compress_ratios").c_str(), ratios))
        for (uint64_t r : ratios)
            if (r != 0) { set_error("QSA sparse attention (compress ratio %llu) is not supported", (unsigned long long)r); return false; }
    std::vector<uint64_t> pl;
    if (!kv_u64_array(m->gguf, (a + "ple.layers").c_str(), pl) || pl.empty()) return true;
    for (uint64_t l : pl) ple_layers.push_back((int)l);
    m->ple_ngram = (int)gguf_arch_int(m->gguf, "ple.ngram_size", 0);
    m->ple_per_ngram = (int)gguf_arch_int(m->gguf, "ple.heads_per_ngram", 0);
    m->ple_conv = (int)gguf_arch_int(m->gguf, "ple.conv_kernel", 0);
    m->ple_eos = (int)gguf_arch_int(m->gguf, "ple.eos_token_id", -1);
    m->ple_dim = (int)gguf_arch_int(m->gguf, "embedding_length_per_layer_input", 0);
    const int heads = (m->ple_ngram - 1) * m->ple_per_ngram;
    if (m->ple_ngram < 2 || m->ple_ngram > 8 || m->ple_per_ngram <= 0 || m->ple_conv < 1 || m->ple_eos < 0 ||
        heads * m->ple_dim != m->n_embd || !kv_u64_array(m->gguf, (a + "ple.layer_multipliers").c_str(), m->ple_mult) ||
        !kv_u64_array(m->gguf, (a + "ple.head_offsets").c_str(), m->ple_offset) ||
        !kv_u64_array(m->gguf, (a + "ple.head_vocab_sizes").c_str(), m->ple_vocab) ||
        (int)m->ple_mult.size() < m->ple_ngram || (int)m->ple_offset.size() != heads ||
        (int)m->ple_vocab.size() != heads) {
        set_error("bad PLE hparams (ngram %d, heads/ngram %d, dim %d, conv %d)", m->ple_ngram, m->ple_per_ngram,
                  m->ple_dim, m->ple_conv);
        return false;
    }
    if (!bind_weight(m->gguf, "per_layer_token_embd.weight", m->ple_table, true)) return false;
    for (int h = 0; h < heads; ++h)
        if (m->ple_table.K != m->ple_dim || m->ple_vocab[h] == 0 ||
            m->ple_offset[h] + m->ple_vocab[h] > (uint64_t)m->ple_table.N) {
            set_error("per_layer_token_embd: %d rows of %d do not hold PLE head %d", m->ple_table.N, m->ple_table.K, h);
            return false;
        }
    return true;
}

// Bind block i of GGUF g (the model's own file, or an MTP sidecar) into L: classify it, bind its tensors and
// check their shapes against the hyperparameters. A full-attention block takes the next KV cache layer.
static bool load_layer(inference_model* m, gguf_model_t* g, int i, LayerWeights& L, const std::vector<int>& ple_layers) {
    const int E = m->n_embd, HD = m->head_dim, HQ = m->n_head, HKV = m->n_head_kv, FF = m->n_ff;
    char name[128];
    auto nm = [&](int i, const char* t) { snprintf(name, sizeof(name), "blk.%d.%s", i, t); return name; };


    // --- Classify the block by the tensors it carries (Phase 20 Step 2) ---
    // ssm_alpha.weight => Gated DeltaNet linear attention; attn_q.weight => full attention. MTP (nextn) blocks
    // sit past n_layer and are never loaded here.
    const bool linear = find_tensor_index(g, nm(i, "ssm_alpha.weight")) >= 0;
    const bool full = find_tensor_index(g, nm(i, "attn_q.weight")) >= 0;
    if (linear == full) {
        set_error("blk.%d: cannot classify (ssm_alpha=%d attn_q=%d)", i, linear, full);
        return false;
    }
    L.kind = linear ? LAYER_LINEAR_ATTN : LAYER_FULL_ATTN;
    if (full) L.kv_slot = m->n_kv_layers++;  // only attention layers get KV cache space

    if (m->n_hc > 0) {  // hyper-connection mixers in place of the norms
        if (!bind_hc_mix(m, g, nm(i, "hc_attn"), L.hc_attn, true) || !bind_hc_mix(m, g, nm(i, "hc_ffn"), L.hc_ffn, true))
            return false;
    } else {
        if (!bind_norm(g, nm(i, "attn_norm.weight"), E, L.attn_norm, true)) return false;
        // the FFN's norm: post_attention_norm (Qwen3.5 and later) or ffn_norm (Qwen3)
        const bool post = find_tensor_index(g, nm(i, "post_attention_norm.weight")) >= 0;
        if (!bind_norm(g, nm(i, post ? "post_attention_norm.weight" : "ffn_norm.weight"), E, L.ffn_norm, true))
            return false;
    }
    L.ple = std::find(ple_layers.begin(), ple_layers.end(), i) != ple_layers.end();
    if (L.ple) {
        const int HCD = m->n_hc * E;
        if (!bind_weight(g, nm(i, "ple_key.weight"), L.ple_key, true) ||
            !bind_weight(g, nm(i, "ple_value.weight"), L.ple_value, true) ||
            !bind_norm(g, nm(i, "ple_norm_key.weight"), HCD, L.ple_norm_key, true) ||
            !bind_norm(g, nm(i, "ple_norm_query.weight"), HCD, L.ple_norm_query, true) ||
            !bind_norm(g, nm(i, "ple_norm_conv.weight"), HCD, L.ple_norm_conv, true) ||
            !check_shape("ple_key", i, L.ple_key, HCD, E) || !check_shape("ple_value", i, L.ple_value, E, E))
            return false;
        const int idx = find_tensor_index(g, nm(i, "ple_conv1d.weight"));
        const int64_t* ne = idx >= 0 ? gguf_tensor_ne(g, idx) : nullptr;
        size_t bytes = 0;
        const void* p = idx >= 0 ? gguf_get_tensor_ptr(g, name, &bytes) : nullptr;
        if (!p || gguf_tensor_type(g, idx) != QT_F16 || ne[0] != m->ple_conv || ne[1] != HCD ||
            bytes != (size_t)m->ple_conv * HCD * 2) {
            set_error("blk.%d.ple_conv1d: expected F16[%d x %d]", i, m->ple_conv, HCD);
            return false;
        }
        L.ple_conv1d = (const uint16_t*)p;
    }

    if (L.kind == LAYER_LINEAR_ATTN) {
        const int DS = m->ssm_d_state, NG = m->ssm_n_group, NV = m->ssm_dt_rank, DI = m->ssm_d_inner;
        const int conv_dim = 2 * NG * DS + DI;
        if (m->ssm_d_conv < 2 || DS <= 0 || NG <= 0 || NV <= 0 || NV % NG != 0 || DI != NV * DS) {
            set_error("blk.%d: bad ssm hparams (d_conv %d d_state %d n_group %d dt_rank %d d_inner %d)", i,
                      m->ssm_d_conv, DS, NG, NV, DI);
            return false;
        }
        if (!bind_weight(g, nm(i, "attn_qkv.weight"), L.W_qkv, true)) return false;
        if (!bind_weight(g, nm(i, "attn_gate.weight"), L.W_z, true)) return false;
        if (!bind_weight(g, nm(i, "ssm_alpha.weight"), L.W_alpha, true)) return false;
        if (!bind_weight(g, nm(i, "ssm_beta.weight"), L.W_beta, true)) return false;
        if (!bind_weight(g, nm(i, "ssm_out.weight"), L.W_ssm_out, true)) return false;
        if (!bind_f32(g, nm(i, "ssm_conv1d.weight"), m->ssm_d_conv, conv_dim, L.ssm_conv1d)) return false;
        if (!bind_norm(g, nm(i, "ssm_dt.bias"), NV, L.ssm_dt, true)) return false;
        if (!bind_norm(g, nm(i, "ssm_a"), NV, L.ssm_a, true)) return false;
        if (!bind_norm(g, nm(i, "ssm_norm.weight"), DS, L.ssm_norm, true)) return false;
        if (!check_shape("attn_qkv", i, L.W_qkv, conv_dim, E) ||
            !check_shape("attn_gate", i, L.W_z, DI, E) ||
            !check_shape("ssm_alpha", i, L.W_alpha, NV, E) ||
            !check_shape("ssm_beta", i, L.W_beta, NV, E) ||
            !check_shape("ssm_out", i, L.W_ssm_out, E, DI)) return false;
    } else {
        if (!bind_weight(g, nm(i, "attn_q.weight"), L.Wq, true)) return false;
        if (!bind_weight(g, nm(i, "attn_k.weight"), L.Wk, true)) return false;
        if (!bind_weight(g, nm(i, "attn_v.weight"), L.Wv, true)) return false;
        if (!bind_weight(g, nm(i, "attn_output.weight"), L.Wo, true)) return false;
        if (!bind_norm(g, nm(i, "attn_q_norm.weight"), HD, L.q_norm, true)) return false;
        if (!bind_norm(g, nm(i, "attn_k_norm.weight"), HD, L.k_norm, true)) return false;
        // Gated attention (Qwen3.5 and later): attn_q holds Q and the output gate, head_dim each, interleaved per
        // head; Qwen3: Q only. The first attention layer decides, the others must agree.
        if (L.kv_slot == 0) m->attn_gated = L.Wq.N == 2 * HQ * HD;
        if (!check_shape("attn_q", i, L.Wq, (m->attn_gated ? 2 : 1) * HQ * HD, E) ||
            !check_shape("attn_k", i, L.Wk, HKV * HD, E) ||
            !check_shape("attn_v", i, L.Wv, HKV * HD, E) ||
            !check_shape("attn_output", i, L.Wo, E, HQ * HD)) return false;
    }

    // FFN: MoE (router ffn_gate_inp + 3-D expert tensors + a sigmoid-gated shared expert) or dense SwiGLU
    L.moe = find_tensor_index(g, nm(i, "ffn_gate_inp.weight")) >= 0;
    if (L.moe) {
        const int NE = m->n_expert, FE = m->n_ff_exp, FS = m->n_ff_shexp;
        if (NE <= 0 || m->n_expert_used <= 0 || m->n_expert_used > NE || FE <= 0 || FS <= 0) {
            set_error("blk.%d: bad MoE hparams (experts %d used %d ff_exp %d ff_shexp %d)", i, NE,
                      m->n_expert_used, FE, FS);
            return false;
        }
        if (!bind_weight(g, nm(i, "ffn_gate_inp.weight"), L.W_router, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_gate_exps.weight"), L.We_gate, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_up_exps.weight"), L.We_up, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_down_exps.weight"), L.We_down, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_gate_shexp.weight"), L.W_gate, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_up_shexp.weight"), L.W_up, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_down_shexp.weight"), L.W_down, true)) return false;
        if (!bind_norm(g, nm(i, "ffn_gate_inp_shexp.weight"), E, L.shexp_gate, true)) return false;
        if (!check_shape("ffn_gate_inp", i, L.W_router, NE, E) ||
            !check_shape("ffn_gate_exps", i, L.We_gate, NE * FE, E) ||
            !check_shape("ffn_up_exps", i, L.We_up, NE * FE, E) ||
            !check_shape("ffn_down_exps", i, L.We_down, NE * E, FE) ||
            !check_shape("ffn_gate_shexp", i, L.W_gate, FS, E) ||
            !check_shape("ffn_up_shexp", i, L.W_up, FS, E) ||
            !check_shape("ffn_down_shexp", i, L.W_down, E, FS)) return false;
    } else {
        if (!bind_weight(g, nm(i, "ffn_gate.weight"), L.W_gate, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_up.weight"), L.W_up, true)) return false;
        if (!bind_weight(g, nm(i, "ffn_down.weight"), L.W_down, true)) return false;
        if (!check_shape("ffn_gate", i, L.W_gate, FF, E) ||
            !check_shape("ffn_up", i, L.W_up, FF, E) ||
            !check_shape("ffn_down", i, L.W_down, E, FF)) return false;
    }
    return true;
}

// A small integer metadata array (BOOL / U8 / I32 / U32 elements), e.g. Gemma-4's per-layer head counts and
// sliding-window pattern. Returns false if absent or of another type.
static bool kv_int_array(gguf_model_t* gguf, const char* key, std::vector<int>& out) {
    int32_t type = -1;
    const void* data = nullptr;
    size_t n = 0;
    if (gguf_kv_get_array(gguf, key, &type, &data, &n) != 0) return false;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        switch (type) {
            case 0: case 7: out[i] = ((const uint8_t*)data)[i]; break;  // U8, BOOL
            case 4: out[i] = (int)((const uint32_t*)data)[i]; break;
            case 5: out[i] = ((const int32_t*)data)[i]; break;
            default: return false;
        }
    }
    return true;
}

// Gemma-4's per-layer attention shapes (llama.cpp gemma4.cpp): sliding-window layers (head_dim key_length_swa, RoPE
// over rope.dimension_count_swa dims at base rope.freq_base_swa) and global ones (key_length, rope.dimension_count,
// rope.freq_base, rope_freqs), and the counts that vary per layer
struct Gemma4Shape {
    std::vector<int> swa, n_kv;
    int hd = 0, hd_swa = 0, rot = 0, rot_swa = 0, window = 0, rot_pairs = 0;
    float base = 0.0f, base_swa = 0.0f;
};

static bool load_gemma4(inference_model* m, Gemma4Shape& g) {
    const std::string a = m->model_name + ".";
    if (!kv_int_array(m->gguf, (a + "attention.sliding_window_pattern").c_str(), g.swa) ||
        !kv_int_array(m->gguf, (a + "attention.head_count_kv").c_str(), g.n_kv) || (int)g.swa.size() != m->n_layer ||
        (int)g.n_kv.size() != m->n_layer) {
        set_error("gemma4: attention.sliding_window_pattern / head_count_kv must have %d entries", m->n_layer);
        return false;
    }
    g.hd = (int)gguf_arch_int(m->gguf, "attention.key_length", 0);
    g.hd_swa = (int)gguf_arch_int(m->gguf, "attention.key_length_swa", 0);
    g.rot = (int)gguf_arch_int(m->gguf, "rope.dimension_count", g.hd);
    g.rot_swa = (int)gguf_arch_int(m->gguf, "rope.dimension_count_swa", g.hd_swa);
    g.window = (int)gguf_arch_int(m->gguf, "attention.sliding_window", 0);
    g.base = gguf_arch_float(m->gguf, "rope.freq_base", 10000.0f);
    g.base_swa = gguf_arch_float(m->gguf, "rope.freq_base_swa", 10000.0f);
    if (gguf_arch_int(m->gguf, "attention.value_length", g.hd) != g.hd ||
        gguf_arch_int(m->gguf, "attention.value_length_swa", g.hd_swa) != g.hd_swa || g.hd <= 0 || g.hd_swa <= 0 ||
        g.window <= 0) {
        set_error("gemma4: needs key_length == value_length (%d, swa %d) and a sliding window", g.hd, g.hd_swa);
        return false;
    }
    if (gguf_arch_int(m->gguf, "embedding_length_per_layer_input", 0) != 0 ||
        gguf_arch_int(m->gguf, "attention.shared_kv_layers", 0) != 0) {
        set_error("gemma4: per-layer inputs and shared KV layers (E2B / E4B) are not supported");
        return false;
    }
    // rope_freqs (global layers): factor 1 for the first pairs, ~1e30 (no rotation) for the rest
    const float* ff = nullptr;
    if (!bind_norm(m->gguf, "rope_freqs.weight", g.rot / 2, ff, false)) return false;
    g.rot_pairs = g.rot / 2;
    if (ff) {
        g.rot_pairs = 0;
        while (g.rot_pairs < g.rot / 2 && ff[g.rot_pairs] == 1.0f) ++g.rot_pairs;
        for (int i = g.rot_pairs; i < g.rot / 2; ++i)
            if (ff[i] < 1e20f) {
                set_error("rope_freqs: factor %g at pair %d (only 1 then 'no rotation' is supported)", ff[i], i);
                return false;
            }
    }
    m->gemma4 = true;
    m->embd_scale = sqrtf((float)m->n_embd);
    m->logit_softcap = gguf_arch_float(m->gguf, "final_logit_softcapping", 0.0f);
    // scratch for the largest layer, KV cache rows of the widest (n_kv * head_dim) one
    int kv_row = 0;
    m->n_head_kv = 0;
    for (int i = 0; i < m->n_layer; ++i) {
        m->n_head_kv = std::max(m->n_head_kv, g.n_kv[i]);
        kv_row = std::max(kv_row, g.n_kv[i] * (g.swa[i] ? g.hd_swa : g.hd));
    }
    m->head_dim = std::max(g.hd, g.hd_swa);
    m->kv_head_dim = kv_row / m->n_head_kv;
    if (m->kv_head_dim * m->n_head_kv != kv_row) { set_error("gemma4: KV rows of %d values", kv_row); return false; }
    m->attn_gated = false;
    m->n_rot = g.rot;
    m->rope_freq_base = g.base;
    return true;
}

// Gemma-4 block i: attention (sliding or global) with q / k norms, post-attention norm, then the dense GeGLU MLP and
// (if ffn_gate_inp is there) the experts, each with its own norms, their sum normed, and the layer output scale
static bool load_layer_gemma4(inference_model* m, int i, LayerWeights& L, const Gemma4Shape& g) {
    gguf_model_t* f = m->gguf;
    const int E = m->n_embd, HQ = m->n_head, FF = m->n_ff;
    char name[128];
    auto nm = [&](const char* t) { snprintf(name, sizeof(name), "blk.%d.%s", i, t); return name; };
    L.kind = LAYER_FULL_ATTN;
    L.kv_slot = m->n_kv_layers++;
    const bool swa = g.swa[i] != 0;
    L.head_dim = swa ? g.hd_swa : g.hd;
    L.n_head_kv = g.n_kv[i];
    L.n_rot = swa ? g.rot_swa : g.rot;
    L.rot_pairs = swa ? L.n_rot / 2 : g.rot_pairs;
    L.window = swa ? g.window : 0;
    L.rope_base = swa ? g.base_swa : g.base;
    const int HD = L.head_dim, HKV = L.n_head_kv;
    L.v_from_k = find_tensor_index(f, nm("attn_v.weight")) < 0;
    if (!bind_norm(f, nm("attn_norm.weight"), E, L.attn_norm, true) ||
        !bind_weight(f, nm("attn_q.weight"), L.Wq, true) || !bind_weight(f, nm("attn_k.weight"), L.Wk, true) ||
        (!L.v_from_k && !bind_weight(f, nm("attn_v.weight"), L.Wv, true)) ||
        !bind_weight(f, nm("attn_output.weight"), L.Wo, true) ||
        !bind_norm(f, nm("attn_q_norm.weight"), HD, L.q_norm, true) ||
        !bind_norm(f, nm("attn_k_norm.weight"), HD, L.k_norm, true) ||
        !bind_norm(f, nm("post_attention_norm.weight"), E, L.post_attn_norm, true) ||
        !bind_norm(f, nm("ffn_norm.weight"), E, L.ffn_norm, true) ||
        !bind_weight(f, nm("ffn_gate.weight"), L.W_gate, true) || !bind_weight(f, nm("ffn_up.weight"), L.W_up, true) ||
        !bind_weight(f, nm("ffn_down.weight"), L.W_down, true) ||
        !bind_norm(f, nm("post_ffw_norm.weight"), E, L.post_ffw_norm, true) ||
        !bind_norm(f, nm("layer_output_scale.weight"), 1, L.out_scale, false))
        return false;
    if (!check_shape("attn_q", i, L.Wq, HQ * HD, E) || !check_shape("attn_k", i, L.Wk, HKV * HD, E) ||
        (!L.v_from_k && !check_shape("attn_v", i, L.Wv, HKV * HD, E)) ||
        !check_shape("attn_output", i, L.Wo, E, HQ * HD) || !check_shape("ffn_gate", i, L.W_gate, FF, E) ||
        !check_shape("ffn_up", i, L.W_up, FF, E) || !check_shape("ffn_down", i, L.W_down, E, FF))
        return false;
    L.moe = find_tensor_index(f, nm("ffn_gate_inp.weight")) >= 0;
    if (!L.moe) return true;
    const int NE = m->n_expert, FE = m->n_ff_exp;
    if (NE <= 0 || m->n_expert_used <= 0 || m->n_expert_used > NE || FE <= 0) {
        set_error("blk.%d: bad MoE hparams (experts %d used %d ff_exp %d)", i, NE, m->n_expert_used, FE);
        return false;
    }
    if (!bind_weight(f, nm("ffn_gate_inp.weight"), L.W_router, true) ||
        !bind_norm(f, nm("ffn_gate_inp.scale"), E, L.router_scale, true) ||
        !bind_norm(f, nm("pre_ffw_norm_2.weight"), E, L.pre_ffw_norm2, true) ||
        !bind_norm(f, nm("post_ffw_norm_1.weight"), E, L.post_ffw_norm1, true) ||
        !bind_norm(f, nm("post_ffw_norm_2.weight"), E, L.post_ffw_norm2, true) ||
        !bind_weight(f, nm("ffn_gate_up_exps.weight"), L.We_gate_up, true) ||
        !bind_weight(f, nm("ffn_down_exps.weight"), L.We_down, true) ||
        !bind_norm(f, nm("ffn_down_exps.scale"), NE, L.exp_scale, false))
        return false;
    return check_shape("ffn_gate_inp", i, L.W_router, NE, E) &&
           check_shape("ffn_gate_up_exps", i, L.We_gate_up, NE * 2 * FE, E) &&
           check_shape("ffn_down_exps", i, L.We_down, NE * E, FE);
}

// Bind every per-layer tensor, classify each block, and check shapes against the hyperparameters
// Nemotron-H (nemotron_h_moe; llama.cpp nemotron-h.cpp): a block's kind from the per-layer arrays, as llama.cpp
// classifies it: head_count_kv[i] == 0 and feed_forward_length[i] == 0: Mamba-2; feed_forward_length[i] == 0:
// attention; else an FFN block
struct NemotronShape {
    std::vector<int> n_kv, n_ff;
};

static bool load_nemotron(inference_model* m, NemotronShape& g) {
    const std::string a = m->model_name + ".";
    if (!kv_int_array(m->gguf, (a + "attention.head_count_kv").c_str(), g.n_kv) ||
        !kv_int_array(m->gguf, (a + "feed_forward_length").c_str(), g.n_ff) || (int)g.n_kv.size() < m->n_layer ||
        (int)g.n_ff.size() < m->n_layer) {
        set_error("nemotron_h: per-layer head_count_kv / feed_forward_length missing");
        return false;
    }
    m->moe_latent = (int)gguf_arch_int(m->gguf, "moe_latent_size", 0);
    m->nemotron = true;
    m->n_head_kv = *std::max_element(g.n_kv.begin(), g.n_kv.begin() + m->n_layer);
    m->n_ff = 0;
    m->n_rot = 0;  // no RoPE: the Mamba-2 layers carry the positions
    m->attn_gated = false;
    m->expert_weights_scale = gguf_arch_float(m->gguf, "expert_weights_scale", 1.0f);
    m->expert_weights_norm = gguf_arch_int(m->gguf, "expert_weights_norm", 1) != 0;
    const int DI = m->ssm_d_inner, NH = m->ssm_dt_rank, NG = m->ssm_n_group, DS = m->ssm_d_state;
    if (m->ssm_d_conv < 2 || DI <= 0 || NH <= 0 || NG <= 0 || DS != 128 || DI % NH != 0 || NH % NG != 0 ||
        DI % NG != 0 || (DI / NG) % 256 != 0 || m->head_dim != 128) {
        set_error("nemotron_h: Mamba-2 shape (d_inner %d, heads %d, groups %d, d_state %d, head_dim %d) not supported",
                  DI, NH, NG, DS, m->head_dim);
        return false;
    }
    return true;
}

static bool load_layer_nemotron(inference_model* m, int i, LayerWeights& L, const NemotronShape& g) {
    gguf_model_t* f = m->gguf;
    const int E = m->n_embd, HQ = m->n_head, HD = m->head_dim;
    const int DI = m->ssm_d_inner, NH = m->ssm_dt_rank, NG = m->ssm_n_group, DS = m->ssm_d_state;
    const int conv_dim = DI + 2 * NG * DS;
    char name[128];
    auto nm = [&](const char* t) { snprintf(name, sizeof(name), "blk.%d.%s", i, t); return name; };
    if (!bind_norm(f, nm("attn_norm.weight"), E, L.attn_norm, true)) return false;
    if (g.n_kv[i] == 0 && g.n_ff[i] == 0) {
        L.kind = LAYER_MAMBA2;
        if (!bind_weight(f, nm("ssm_in.weight"), L.W_ssm_in, true) ||
            !bind_weight(f, nm("ssm_out.weight"), L.W_ssm_out, true) ||
            !bind_f32(f, nm("ssm_conv1d.weight"), m->ssm_d_conv, conv_dim, L.ssm_conv1d) ||
            !bind_norm(f, nm("ssm_conv1d.bias"), conv_dim, L.ssm_conv1d_b, true) ||
            !bind_norm(f, nm("ssm_dt.bias"), NH, L.ssm_dt, true) ||
            !bind_f32(f, nm("ssm_a"), 1, NH, L.ssm_a) || !bind_f32(f, nm("ssm_d"), 1, NH, L.ssm_d) ||
            !bind_f32(f, nm("ssm_norm.weight"), DI / NG, NG, L.ssm_norm))
            return false;
        return check_shape("ssm_in", i, L.W_ssm_in, 2 * DI + 2 * NG * DS + NH, E) &&
               check_shape("ssm_out", i, L.W_ssm_out, E, DI);
    }
    if (g.n_ff[i] == 0) {  // attention: no RoPE, no q/k norm, no gate
        L.kind = LAYER_FULL_ATTN;
        L.kv_slot = m->n_kv_layers++;
        L.n_head_kv = g.n_kv[i];
        if (!bind_weight(f, nm("attn_q.weight"), L.Wq, true) || !bind_weight(f, nm("attn_k.weight"), L.Wk, true) ||
            !bind_weight(f, nm("attn_v.weight"), L.Wv, true) || !bind_weight(f, nm("attn_output.weight"), L.Wo, true))
            return false;
        return check_shape("attn_q", i, L.Wq, HQ * HD, E) && check_shape("attn_k", i, L.Wk, L.n_head_kv * HD, E) &&
               check_shape("attn_v", i, L.Wv, L.n_head_kv * HD, E) && check_shape("attn_output", i, L.Wo, E, HQ * HD);
    }
    L.kind = LAYER_FFN;
    L.moe = find_tensor_index(f, nm("ffn_gate_inp.weight")) >= 0;
    if (!L.moe) {
        if (!bind_weight(f, nm("ffn_up.weight"), L.W_up, true) || !bind_weight(f, nm("ffn_down.weight"), L.W_down, true))
            return false;
        return check_shape("ffn_up", i, L.W_up, g.n_ff[i], E) && check_shape("ffn_down", i, L.W_down, E, g.n_ff[i]);
    }
    const int NE = m->n_expert, FE = m->n_ff_exp, FS = m->n_ff_shexp, ME = m->moe_latent > 0 ? m->moe_latent : E;
    if (NE <= 0 || m->n_expert_used <= 0 || m->n_expert_used > NE || FE <= 0 || FS <= 0) {
        set_error("blk.%d: bad MoE hparams (experts %d used %d ff_exp %d ff_shexp %d)", i, NE, m->n_expert_used, FE, FS);
        return false;
    }
    if (m->moe_latent > 0 &&  // the latent MoE (Super): the experts between ffn_latent_down and ffn_latent_up
        (!bind_weight(f, nm("ffn_latent_down.weight"), L.W_lat_down, true) ||
         !bind_weight(f, nm("ffn_latent_up.weight"), L.W_lat_up, true) ||
         !check_shape("ffn_latent_down", i, L.W_lat_down, ME, E) || !check_shape("ffn_latent_up", i, L.W_lat_up, E, ME)))
        return false;
    if (!bind_weight(f, nm("ffn_gate_inp.weight"), L.W_router, true) ||
        !bind_norm(f, nm("exp_probs_b.bias"), NE, L.exp_probs_b, true) ||
        !bind_weight(f, nm("ffn_up_exps.weight"), L.We_up, true) ||
        !bind_weight(f, nm("ffn_down_exps.weight"), L.We_down, true) ||
        !bind_weight(f, nm("ffn_up_shexp.weight"), L.W_up, true) ||
        !bind_weight(f, nm("ffn_down_shexp.weight"), L.W_down, true))
        return false;
    return check_shape("ffn_gate_inp", i, L.W_router, NE, E) && check_shape("ffn_up_exps", i, L.We_up, NE * FE, ME) &&
           check_shape("ffn_down_exps", i, L.We_down, NE * ME, FE) && check_shape("ffn_up_shexp", i, L.W_up, FS, E) &&
           check_shape("ffn_down_shexp", i, L.W_down, E, FS);
}

static bool load_model_weights(inference_model* m, const std::vector<int>& ple_layers) {
    m->layers.resize(m->n_layer);
    Gemma4Shape g4;
    NemotronShape gn;
    if (m->model_name == "gemma4" && !load_gemma4(m, g4)) return false;
    if (m->model_name == "nemotron_h_moe" && !load_nemotron(m, gn)) return false;
    for (int i = 0; i < m->n_layer; ++i)
        if (!(m->gemma4     ? load_layer_gemma4(m, i, m->layers[i], g4)
              : m->nemotron ? load_layer_nemotron(m, i, m->layers[i], gn)
                            : load_layer(m, m->gguf, i, m->layers[i], ple_layers)))
            return false;
    return true;
}

// ======================== C API IMPLEMENTATION ========================

inference_model_t* inference_load_model(const char* gguf_path) { return inference_load_model_ex(gguf_path, nullptr); }

const char* inference_stage_info(const inference_model_t* m) { return m ? m->stage_info.c_str() : ""; }

// The tensors staged (inference_load_opts.stage_dir): what the GPU reads. Not the PLE table, nor the token
// embeddings when the LM head is a tensor of its own (the CPU decodes their rows), nor the MTP (nextn) blocks after
// the decoder stack
static void stage_model_tensors(inference_model* m, const char* dir) {
    const bool own_head = find_tensor_index(m->gguf, "output.weight") >= 0;
    const int n_layer = m->n_layer;
    auto keep = [&](const char* name) {
        if (!strcmp(name, "per_layer_token_embd.weight")) return false;
        if (own_head && !strcmp(name, "token_embd.weight")) return false;
        int blk = -1;
        if (sscanf(name, "blk.%d.", &blk) == 1 && blk >= n_layer) return false;
        return true;
    };
    std::string why;
    char b[512];
    if (model_stage(m->gguf, dir, keep, m->stage, why)) {
        snprintf(b, sizeof b, "%s %s (%zu MB)", m->stage.reused ? "reused" : "staged", m->stage.path.c_str(),
                 m->stage.staged >> 20);
    } else {
        snprintf(b, sizeof b, "not staged: %s", why.c_str());
    }
    m->stage_info = b;
    INFER_LOG("[stage] %s\n", b);
}

inference_model_t* inference_load_model_ex(const char* gguf_path, const inference_load_opts* opts) {
    clear_error();
    if (!gguf_path) { set_error("NULL path"); return nullptr; }

    INFER_LOG("[LOAD] Starting load_model for %s\n", gguf_path);
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
    INFER_LOG("[LOAD] arch=%s blocks=%lld (decoder %d + nextn %d) E=%d heads=%d/%d head_dim=%d ff=%d "
            "n_rot=%d rope_base=%g eps=%g\n", arch ? arch : "?", (long long)n_layer, m->n_layer, m->n_nextn,
            m->n_embd, m->n_head, m->n_head_kv, m->head_dim, m->n_ff, m->n_rot, m->rope_freq_base, m->eps);

    m->n_ctx = (int)ctx_len;

    // Staging: before any tensor is bound, so that every weight pointer is the stage copy's
    if (opts && opts->stage_dir && opts->stage_dir[0]) stage_model_tensors(m, opts->stage_dir);

    // Token embeddings: any decodable type; rows are decoded per token in the forward pass
    if (!bind_weight(m->gguf, "token_embd.weight", m->token_embd, true) ||
        m->token_embd.K != m->n_embd || m->token_embd.N != m->vocab_size) {
        if (m->token_embd.data)
            set_error("token_embd: shape %dx%d, expected %dx%d", m->token_embd.N, m->token_embd.K,
                      m->vocab_size, m->n_embd);
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }

    // Per-layer weights and norms (qwen4exp: hyper-connection mixers and the PLE layers instead of the norms)
    std::vector<int> ple_layers;
    if (!load_qwen4exp(m, ple_layers) || !load_model_weights(m, ple_layers)) {
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }

    // LM head: output.weight when present, else tied to token_embd
    if (!bind_weight(m->gguf, "output.weight", m->output, false) ||
        !bind_norm(m->gguf, "output_norm.weight", m->n_embd, m->output_norm, m->n_hc == 0)) {
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }
    if (!m->output.data) m->output = m->token_embd;
    if (m->output.K != m->n_embd || m->output.N != m->vocab_size) {
        set_error("output: shape %dx%d, expected %dx%d", m->output.N, m->output.K, m->vocab_size, m->n_embd);
        gguf_mmap_close(m->gguf); delete m; return nullptr;
    }
    INFER_LOG("[LOAD] token_embd type %d, LM head type %d%s\n", m->token_embd.type, m->output.type,
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

static kv_cache_config_t mtp_kv_config(const inference_model* m, int n_ctx) {
    return kv_cache_config_t{1, n_ctx, m->n_embd, m->head_dim, m->n_head, m->n_head_kv, 0, m->kv_type};
}

static void free_mtp(inference_model* m) {
    if (!m->mtp) return;
    unpin(m->mtp->pinned);
    if (m->mtp->kvc) kv_cache_free(m->mtp->kvc);
    if (m->mtp->gguf) gguf_mmap_close(m->mtp->gguf);
    delete m->mtp;
    m->mtp = nullptr;
}

int inference_set_prefill_batch(inference_model_t* m, int n_tokens) {
    clear_error();
    if (!m || (n_tokens != 0 && (n_tokens < 16 || n_tokens > 8192))) {
        set_error("prefill batch: 0 (default) or 16 .. 8192 tokens");
        return INFERENCE_ERR_INVALID_ARG;
    }
    if (m->qwen) { set_error("prefill batch: call before inference_kv_cache_init"); return -1; }
    m->prefill_batch = n_tokens;
    return 0;
}

int inference_set_draft_tokens(inference_model_t* m, int n) {
    clear_error();
    if (!m || n < 0 || n > SPEC_MAX_DRAFTS) {
        set_error("draft tokens: 0 (adaptive) .. %d", SPEC_MAX_DRAFTS);
        return -1;
    }
    m->spec_drafts = n;
    return 0;
}

int inference_set_draft_model(inference_model_t* m, const char* path) {
    clear_error();
    if (!m) { set_error("NULL model"); return -1; }
    if (m->qwen) qwen_model_sync(m->qwen);
    free_mtp(m);
    for (int k = 0; k < SPEC_MAX_DRAFTS; ++k) m->spec_acc[k] = 0.8f, m->spec_ms[k] = 0.0f;  // the adaptive count anew
    m->spec_passes = 0;
    if (!path) return 0;
    if (m->n_hc <= 0) { set_error("draft heads are supported for qwen4exp models only"); return -1; }
    MtpHead* h = new MtpHead();
    m->mtp = h;
    h->gguf = gguf_mmap_open(path);
    if (!h->gguf) { set_error("cannot open draft model %s", path); free_mtp(m); return -1; }
    const char* arch = gguf_arch(h->gguf);
    const int blocks = (int)gguf_block_count(h->gguf), nextn = (int)gguf_arch_int(h->gguf, "nextn_predict_layers", 0);
    if (!arch || m->model_name != arch || nextn != 1 || gguf_embedding_length(h->gguf) != m->n_embd ||
        gguf_arch_int(h->gguf, "hyper_connection.count", 0) != m->n_hc ||
        gguf_arch_int(h->gguf, "hyper_connection.low_rank", 0) != m->hc_rank ||
        gguf_head_count(h->gguf) != m->n_head || gguf_head_count_kv(h->gguf) != m->n_head_kv ||
        gguf_arch_int(h->gguf, "expert_count", 0) != m->n_expert ||
        gguf_arch_int(h->gguf, "expert_used_count", 0) != m->n_expert_used) {
        set_error("%s is not an MTP head for this model (arch %s, %d nextn layers)", path, arch ? arch : "?", nextn);
        free_mtp(m);
        return -1;
    }
    const int il = blocks - 1, HCD = m->n_hc * m->n_embd;
    char name[128];
    auto nm = [&](const char* t) { snprintf(name, sizeof(name), "blk.%d.%s", il, t); return name; };
    const int n_kv = m->n_kv_layers;
    const bool ok = load_layer(m, h->gguf, il, h->L, {});
    m->n_kv_layers = n_kv;  // the head's attention has its own cache
    h->L.kv_slot = 0;
    if (!ok || h->L.kind != LAYER_FULL_ATTN || !h->L.moe ||
        !bind_weight(h->gguf, "token_embd.weight", h->token_embd, true) ||
        !bind_weight(h->gguf, "output.weight", h->output, true) ||
        !bind_weight(h->gguf, nm("nextn.eh_proj.weight"), h->eh_proj, true) ||
        !bind_norm(h->gguf, nm("nextn.enorm.weight"), m->n_embd, h->enorm, true) ||
        !bind_norm(h->gguf, nm("nextn.hnorm.weight"), HCD, h->hnorm, true) ||
        !bind_hc_mix(m, h->gguf, "output_hc", h->hc_head, false)) {
        free_mtp(m);  // the binding that failed set the error
        return -1;
    }
    if (h->eh_proj.N != m->n_embd || h->eh_proj.K != 2 * m->n_embd || h->token_embd.K != m->n_embd ||
        h->output.K != m->n_embd || h->output.N != m->vocab_size || h->token_embd.N != m->vocab_size) {
        set_error("draft model: eh_proj / token_embd / output shapes do not match the model");
        free_mtp(m);
        return -1;
    }
    size_t pb = 0, ub = 0;
    if (!pin_gguf(h->gguf, {&h->token_embd}, h->pinned, pb, ub)) {
        set_error("draft model: hipHostRegister failed");
        free_mtp(m);
        return -1;
    }
    if (m->kvc) {
        const kv_cache_config_t cfg = mtp_kv_config(m, kv_cache_n_ctx(m->kvc));
        h->kvc = kv_cache_create("/dev/shm", &cfg);
        if (!h->kvc) { set_error("draft model: KV cache"); free_mtp(m); return -1; }
    }
    if (m->qwen) qwen_model_mtp_reset(m->qwen);
    return 0;
}

void inference_free_model(inference_model_t* m) {
    if (!m) return;
    free_mtp(m);
    INFER_LOG("[CLEANUP] Starting model free...\n");
    if (m->tokenizer) { INFER_LOG("[CLEANUP] Freeing tokenizer\n"); tokenizer_free(m->tokenizer); }
    // IMPORTANT: unpin must happen BEFORE qwen_model_free since unpin walks layer weights
    // whose device pointers point into the layer scratch buffers we're about to free.
    INFER_LOG("[CLEANUP] Unpinning model memory\n");
    unpin_model_memory(m);
    model_stage_unmap(m->stage);
    inference_store_close(m);
    if (m->cur_seq != 0) inference_seq_select(m, 0);
    free_seqs(m);
    if (m->kvc) { INFER_LOG("[CLEANUP] Freeing KV cache\n"); kv_cache_free(m->kvc); }
    if (m->kvc_swa) kv_cache_free(m->kvc_swa);
    if (m->qwen) { INFER_LOG("[CLEANUP] Freeing qwen model\n"); qwen_model_free(m->qwen); }
    if (m->gguf) { INFER_LOG("[CLEANUP] Closing GGUF\n"); gguf_mmap_close(m->gguf); }
    INFER_LOG("[CLEANUP] Deleting model struct\n");
    delete m;
    INFER_LOG("[CLEANUP] Model free complete\n");
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

// Gemma-4's sliding-window layers only read their last `window` positions: their K / V go to a ring of R
// positions (a power of two >= window + the prefill batch, since a batch writes all its positions before it
// attends), one block that kernels address with block stride 0, so position p lands in slot p % R. The global
// layers keep the block cache, with rows of their own width. 0 when the model has no sliding layers or
// inference_set_swa_ring(m, 0) asks for one cache of every layer (what the block store needs); a larger ring
// (inference_set_swa_ring(m, N)) keeps more history for prefix reuse.
static int swa_ring_size(const inference_model_t* m, int n_ctx, int* window) {
    *window = 0;
    if (!m->gemma4) return 0;
    for (const LayerWeights& L : m->layers) *window = std::max(*window, L.window);
    if (*window <= 0 || m->swa_ring_opt == 0) return 0;
    const int need = std::max(*window + prefill_batch_tokens(m), m->swa_ring_opt);
    int R = 256;
    while (R < need) R *= 2;
    return R < n_ctx ? R : 0;  // no gain when the context is that short
}

int inference_set_swa_ring(inference_model_t* m, int positions) {
    clear_error();
    if (!m || positions < -1 || positions > (1 << 24)) {
        set_error("sliding-window ring: -1 (automatic), 0 (none) or at least that many positions");
        return INFERENCE_ERR_INVALID_ARG;
    }
    if (m->qwen) { set_error("sliding-window ring: call before inference_kv_cache_init"); return -1; }
    m->swa_ring_opt = positions;
    return 0;
}

int inference_swa_ring(const inference_model_t* m) { return m ? m->swa_ring : 0; }

int inference_set_kv_type(inference_model_t* m, int kv_type) {
    clear_error();
    if (!m || (kv_type != INFERENCE_KV_F16 && kv_type != INFERENCE_KV_Q8_0)) {
        set_error("KV cache type: INFERENCE_KV_F16 or INFERENCE_KV_Q8_0");
        return INFERENCE_ERR_INVALID_ARG;
    }
    if (m->qwen) { set_error("KV cache type: call before inference_kv_cache_init"); return -1; }
    m->kv_type = kv_type == INFERENCE_KV_Q8_0 ? KV_CACHE_Q8_0 : KV_CACHE_F16;
    return 0;
}

int inference_kv_cache_init(inference_model_t* m, int n_ctx) {
    clear_error();
    if (!m) { set_error("NULL model"); return -1; }
    if (m->seqs.size() > 1) { set_error("free the sequences (inference_seq_free) before a new KV cache"); return -1; }
    if (m->kvc_swa) { kv_cache_free(m->kvc_swa); m->kvc_swa = nullptr; }
    int window = 0;
    const int R = swa_ring_size(m, n_ctx, &window);
    m->swa_ring = R;
    m->swa_window = window;
    // One KV layer per full-attention layer (LayerWeights::kv_slot); DeltaNet layers keep recurrent state instead
    kv_cache_config_t cfg{m->n_kv_layers > 0 ? m->n_kv_layers : 1, n_ctx, m->n_embd,
                          m->kv_head_dim > 0 ? m->kv_head_dim : m->head_dim, m->n_head, m->n_head_kv, 0, m->kv_type};
    if (m->gemma4) {  // number the layers' KV slots: in the ring and the block cache, or all in the block cache
        int n_ring = 0, n_glob = 0, ring_hd = 0, ring_kv = 0, glob_hd = 0, glob_kv = 0;
        for (LayerWeights& L : m->layers) {
            L.kv_ring = R > 0 && L.window > 0;
            L.kv_slot = R == 0 ? n_glob++ : L.kv_ring ? n_ring++ : n_glob++;
            (L.kv_ring ? ring_hd : glob_hd) = L.head_dim;
            (L.kv_ring ? ring_kv : glob_kv) = L.n_head_kv;
        }
        if (R > 0) {
            cfg = kv_cache_config_t{std::max(1, n_glob), n_ctx, m->n_embd, glob_hd, m->n_head, glob_kv, 0, m->kv_type};
            const kv_cache_config_t rcfg{n_ring, R, m->n_embd, ring_hd, m->n_head, ring_kv, R, m->kv_type};
            m->kvc_swa = kv_cache_create("/dev/shm", &rcfg);
            if (!m->kvc_swa) { set_error("kv_cache_create (sliding-window ring) failed"); return -1; }
            m->kv_cfg_swa = rcfg;
        }
    }
    m->kvc = kv_cache_create("/dev/shm", &cfg);
    if (!m->kvc) { set_error("kv_cache_create failed"); return -1; }
    m->kv_cfg = cfg;
    if (m->kvc_swa)
        INFER_LOG("[KV_INIT] sliding-window ring: %d layers x %d positions (%.0f MB), %d global layers x %d (%.0f MB)\n",
                  kv_cache_n_layers(m->kvc_swa), R, kv_cache_total_bytes(m->kvc_swa) / 1048576.0,
                  kv_cache_n_layers(m->kvc), kv_cache_n_ctx(m->kvc), kv_cache_total_bytes(m->kvc) / 1048576.0);
    else
        INFER_LOG("[KV_INIT] KV cache: %d layers x %d positions (%.0f MB)\n", kv_cache_n_layers(m->kvc),
                  kv_cache_n_ctx(m->kvc), kv_cache_total_bytes(m->kvc) / 1048576.0);
    if (m->mtp) {
        if (m->mtp->kvc) kv_cache_free(m->mtp->kvc);
        const kv_cache_config_t mcfg = mtp_kv_config(m, kv_cache_n_ctx(m->kvc));
        m->mtp->kvc = kv_cache_create("/dev/shm", &mcfg);
        if (!m->mtp->kvc) { set_error("draft model: KV cache"); return -1; }
    }
    m->n_ctx = n_ctx;
    seq_reset(m);
    // Create Qwen model for forward passes
    INFER_LOG("[KV_INIT] About to create qwen model\n");
    m->qwen = qwen_model_create(m);
    if (!m->qwen) { set_error("model setup failed: %s", qwen_model_last_error()); return -1; }
    INFER_LOG("[KV_INIT] Qwen model created at %p\n", (void*)m->qwen);
    if (m->lm_head_q8 && qwen_model_set_lm_head_q8(m->qwen, 1) < 0) {
        set_error("LM head Q8_0 copy failed: %s", qwen_model_last_error());
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
        set_error("LM head Q8_0 copy failed: %s", qwen_model_last_error());
        m->lm_head_q8 = 0;
        return INFERENCE_ERR_OUT_OF_MEMORY;
    }
    return 0;
}

// Free every sequence but the current one (their caches; QwenModel frees their states with itself)
static void free_seqs(inference_model_t* m) {
    for (inference_model::SeqCtx* c : m->seqs) {
        if (!c) continue;
        if (c->kvc) kv_cache_free(c->kvc);
        if (c->kvc_swa) kv_cache_free(c->kvc_swa);
        delete c;
    }
    m->seqs.clear();
    m->cur_seq = 0;
}

// The model's sequence fields <-> c
static void seq_swap(inference_model_t* m, inference_model::SeqCtx& c) {
    std::swap(m->kvc, c.kvc);
    std::swap(m->kvc_swa, c.kvc_swa);
    std::swap(m->n_past, c.n_past);
    std::swap(m->seq, c.seq);
    std::swap(m->ckpts, c.ckpts);
    std::swap(m->ckpt_clock, c.ckpt_clock);
    std::swap(m->last_reused, c.last_reused);
    std::swap(m->last_prefilled, c.last_prefilled);
    std::swap(m->chain, c.chain);
    std::swap(m->published, c.published);
    std::swap(m->store_snapshot_at, c.store_snapshot_at);
}

int inference_seq_new(inference_model_t* m) {
    clear_error();
    if (!m || !m->qwen || !m->kvc) { set_error("sequences: call inference_kv_cache_init first"); return -1; }
    if (m->mtp) { set_error("sequences: not with a draft model (speculative decoding is single-sequence)"); return -1; }
    if (m->seqs.empty()) m->seqs.push_back(new inference_model::SeqCtx());  // slot 0: the default sequence
    const int j = qwen_model_seq_new(m->qwen);
    if (j < 0) { set_error("%s", qwen_model_last_error()); return -1; }
    if ((int)m->seqs.size() <= j) m->seqs.resize((size_t)j + 1, nullptr);
    if (!m->seqs[j]) m->seqs[j] = new inference_model::SeqCtx();
    inference_model::SeqCtx& c = *m->seqs[j];
    c = inference_model::SeqCtx();
    c.kvc = kv_cache_create("/dev/shm", &m->kv_cfg);
    c.kvc_swa = m->kv_cfg_swa.n_layers > 0 ? kv_cache_create("/dev/shm", &m->kv_cfg_swa) : nullptr;
    if (!c.kvc || (m->kv_cfg_swa.n_layers > 0 && !c.kvc_swa)) {
        if (c.kvc) kv_cache_free(c.kvc);
        if (c.kvc_swa) kv_cache_free(c.kvc_swa);
        c = inference_model::SeqCtx();
        qwen_model_seq_free(m->qwen, j);
        set_error("sequences: no memory for another KV cache");
        return -1;
    }
    c.used = true;
    return j;
}

int inference_seq_select(inference_model_t* m, int seq) {
    clear_error();
    if (!m || !m->qwen) { set_error("Invalid state"); return INFERENCE_ERR_INVALID_STATE; }
    if (seq == m->cur_seq) return 0;
    if (seq < 0 || seq >= (int)m->seqs.size() || !m->seqs[seq] || (seq > 0 && !m->seqs[seq]->used)) {
        set_error("no sequence %d", seq);
        return INFERENCE_ERR_INVALID_ARG;
    }
    if (qwen_model_seq_select(m->qwen, seq) != 0) { set_error("%s", qwen_model_last_error()); return -1; }
    seq_swap(m, *m->seqs[m->cur_seq]);  // the current sequence back into its slot
    seq_swap(m, *m->seqs[seq]);
    m->cur_seq = seq;
    return 0;
}

int inference_seq_current(const inference_model_t* m) { return m ? m->cur_seq : -1; }

int inference_seq_free(inference_model_t* m, int seq) {
    clear_error();
    if (!m || seq <= 0 || seq >= (int)m->seqs.size() || !m->seqs[seq] || !m->seqs[seq]->used) {
        set_error("no sequence %d", seq);
        return INFERENCE_ERR_INVALID_ARG;
    }
    if (seq == m->cur_seq && inference_seq_select(m, 0) != 0) return -1;
    if (qwen_model_seq_free(m->qwen, seq) != 0) { set_error("%s", qwen_model_last_error()); return -1; }
    inference_model::SeqCtx& c = *m->seqs[seq];
    if (c.kvc) kv_cache_free(c.kvc);
    if (c.kvc_swa) kv_cache_free(c.kvc_swa);
    c = inference_model::SeqCtx();
    return 0;
}

void inference_kv_cache_free(inference_model_t* m) {
    if (m && m->kvc) { kv_cache_free(m->kvc); m->kvc = nullptr; }
    if (m && m->kvc_swa) { kv_cache_free(m->kvc_swa); m->kvc_swa = nullptr; }
}

qwen_model_t* inference_get_qwen_model(inference_model_t* m) {
    if (!m) return nullptr;
    return m->qwen;
}

void inference_set_seed(inference_model_t* m, unsigned int seed) {
    if (m) m->seed = seed;
}

// A new, empty sequence: position 0, no history, no checkpoints (their states belong to the old sequence)
static void seq_reset(inference_model_t* m) {
    m->n_past = 0;
    m->seq.clear();
    m->ckpts.clear();
    m->chain.clear();
    m->published = 0;
}

void inference_reset(inference_model_t* m) {
    if (m) seq_reset(m);
}

int inference_n_past(const inference_model_t* m) {
    return m ? m->n_past : 0;
}

// ---- KV block store helpers (Phase 22b; inference_store_open)

static int kv_B(const inference_model_t* m) { return m->kvc ? kv_cache_block_tokens(m->kvc) : 1; }

// Chained block hash: SHA-256 of the previous hash and the block's token ids (32-bit little endian)
static inference_model::Hash block_hash(const inference_model::Hash& prev, const int32_t* tokens, int n) {
    sha256_ctx c;
    sha256_init(&c);
    sha256_update(&c, prev.b, sizeof(prev.b));
    for (int i = 0; i < n; ++i) {
        const uint32_t t = (uint32_t)tokens[i];
        const uint8_t le[4] = {(uint8_t)t, (uint8_t)(t >> 8), (uint8_t)(t >> 16), (uint8_t)(t >> 24)};
        sha256_update(&c, le, 4);
    }
    inference_model::Hash h;
    sha256_final(&c, h.b);
    return h;
}

static std::string store_name(const inference_model::Hash& h, const char* ext) {
    static const char* hx = "0123456789abcdef";
    std::string s;
    for (uint8_t b : h.b) s += hx[b >> 4], s += hx[b & 15];
    return s + "." + ext;
}

// A complete store file: present with its exact size (a file still being copied in counts as missing)
static bool store_has(const inference_model_t* m, const std::string& name, size_t bytes = 0) {
    struct stat st;
    return fstatat(m->store_dirfd, name.c_str(), &st, 0) == 0 && (bytes == 0 || (size_t)st.st_size == bytes);
}

// Mark a store file as used now (the Perl side's LRU eviction looks at mtime)
static void store_touch(const inference_model_t* m, const std::string& name) {
    (void)utimensat(m->store_dirfd, name.c_str(), nullptr, 0);
}

static size_t state_bytes(const inference_model_t* m) {
    size_t b = 0;
    (void)qwen_model_state_memory(m->qwen, &b);
    return b;
}

// Hashes of the sequence's full blocks
static void chain_extend(inference_model_t* m) {
    const size_t B = (size_t)kv_B(m);
    while ((m->chain.size() + 1) * B <= m->seq.size()) {
        const size_t i = m->chain.size();
        m->chain.push_back(block_hash(i ? m->chain[i - 1] : m->store_root, m->seq.data() + i * B, (int)B));
    }
}

// Rewind the sequence to len tokens; its block hashes and published count follow
static void seq_truncate(inference_model_t* m, int len) {
    m->n_past = len;
    m->seq.resize((size_t)len);
    const size_t full = (size_t)len / (size_t)kv_B(m);
    if (m->chain.size() > full) m->chain.resize(full);
    if (m->published > full) m->published = full;
}

// Store blocks are shared and never written: before positions [p, p + n) are written, their slots become private
// (keeping the rows before p in p's own slot)
static int ensure_writable(inference_model_t* m, int p, int n) {
    if (!m->kvc || n <= 0) return 0;
    const int B = kv_B(m);
    for (int s = p / B; s <= (p + n - 1) / B && s < kv_cache_n_slots(m->kvc); ++s)
        if (kv_cache_slot_shared(m->kvc, s) && kv_cache_slot_make_private(m->kvc, s, s == p / B && p % B != 0) != 0) {
            set_error("cannot make KV block %d private", s);
            return -1;
        }
    return 0;
}

// The recurrent state as <hash>.rst: written to an anonymous file, linked under its name once complete
static int store_save_state(inference_model_t* m, const inference_model::Hash& h) {
    size_t bytes = 0;
    void* mem = qwen_model_state_memory(m->qwen, &bytes);
    if (!mem || bytes == 0) return 0;  // no recurrent layers
    const std::string name = store_name(h, "rst");
    if (store_has(m, name)) {
        store_touch(m, name);
        return 0;
    }
    if (qwen_model_sync(m->qwen) != 0) { set_error("sync failed: %s", qwen_model_last_error()); return -1; }
    const int fd = openat(m->store_dirfd, ".", O_TMPFILE | O_RDWR | O_CLOEXEC, 0644);
    if (fd < 0) { set_error("store: cannot create a state file: %s", strerror(errno)); return -1; }
    for (size_t done = 0; done < bytes;) {
        const ssize_t w = write(fd, (const char*)mem + done, bytes - done);
        if (w <= 0) {
            close(fd);
            set_error("store: writing a state snapshot failed: %s", strerror(errno));
            return -1;
        }
        done += (size_t)w;
    }
    char proc[64];
    snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
    const int rc = linkat(AT_FDCWD, proc, m->store_dirfd, name.c_str(), AT_SYMLINK_FOLLOW);
    const int err = errno;
    close(fd);
    if (rc != 0 && err != EEXIST) { set_error("store: linking %s failed: %s", name.c_str(), strerror(err)); return -1; }
    if (rc == 0) ++m->st_snap_saved;
    return 0;
}

static bool snapshot_boundary(const inference_model_t* m, int k) {
    return k > 0 && (k == m->store_snapshot_at || (m->store_snapshot_every > 0 && k % m->store_snapshot_every == 0));
}

// After a forward pass: publish the blocks it completed, and snapshot the state at a chosen block boundary
static int store_after_forward(inference_model_t* m) {
    if (m->store_dirfd < 0) return 0;
    chain_extend(m);
    if (m->published < m->chain.size() && qwen_model_sync(m->qwen) != 0) {
        set_error("sync failed: %s", qwen_model_last_error());
        return -1;
    }
    for (; m->published < m->chain.size(); ++m->published) {
        const int i = (int)m->published;
        if (kv_cache_slot_shared(m->kvc, i)) continue;  // already a store block
        const std::string path = m->store_dir + "/" + store_name(m->chain[i], "kvb");
        const int rc = kv_cache_slot_publish(m->kvc, i, path.c_str());
        if (rc < 0) { set_error("store: publishing KV block %d failed", i); return -1; }
        if (rc == 0) ++m->st_published;  // 1: another session published the same block already
    }
    const int B = kv_B(m), k = m->n_past / B;
    if (m->n_past % B == 0 && snapshot_boundary(m, k) && state_bytes(m) > 0) return store_save_state(m, m->chain[k - 1]);
    return 0;
}

// One forward pass for token at position n_past (the next position of the current sequence)
static int forward_next(inference_model_t* m, int token_id) {
    if (!m || !m->kvc || !m->qwen) { set_error("Invalid state (call inference_kv_cache_init first)"); return -1; }
    if (m->n_past >= m->n_ctx) { set_error("context full (%d tokens)", m->n_ctx); return INFERENCE_ERR_CONTEXT_FULL; }
    if (ensure_writable(m, m->n_past, 1) != 0) return -1;
    if (qwen_model_forward(m->qwen, token_id, m->n_past, nullptr) != 0) { set_error("forward pass failed: %s", qwen_model_last_error()); return -1; }
    ++m->n_past;
    m->seq.push_back(token_id);
    return store_after_forward(m);
}

// A verify pass in flight: drafts d[0 .. k - 1] decoded after the token at position n_past - 1 (not counted yet);
// the next token is sampled from the logits after token i of the pass (i drafts accepted so far)
struct SpecPass {
    int d[SPEC_MAX_DRAFTS];
    int k = 0, i = 0;
};

// Adaptive draft count. A pass with k drafts yields 1 + a1 + a1 a2 + ... tokens, a_j the acceptance of draft j
// given the ones before it accepted, and takes the time of the head's drafts and the k + 1-token verify pass;
// both are moving averages (SPEC_RATE per observed pass). Of SPEC_MIN_DRAFTS .. SPEC_MAX_DRAFTS, the k with the
// most tokens per second is used (one draft was never the best: measured on code, chat, prose and Chinese, it is
// below two). A k not timed yet is estimated from a timed one with a pass of k drafts costing (1 +
// SPEC_COST_PER_DRAFT k) one-token passes (Qwen3.8: 39 / 49 / 58 ms for k = 1 / 2 / 3). Every SPEC_EXPLORE-th pass
// drafts the maximum, so that the last position and its cost keep being measured.
#define SPEC_MIN_DRAFTS 2
#define SPEC_COST_PER_DRAFT 0.33f
#define SPEC_RATE 0.1f
#define SPEC_EXPLORE 16
static int spec_drafts_for_pass(inference_model_t* m) {
    if (m->spec_drafts > 0) return m->spec_drafts;
    if (++m->spec_passes % SPEC_EXPLORE == 0) return SPEC_MAX_DRAFTS;
    int ref = 0;  // a timed draft count to scale the cost model by (0: none yet)
    for (int k = 1; k <= SPEC_MAX_DRAFTS && !ref; ++k)
        if (m->spec_ms[k - 1] > 0.0f) ref = k;
    int best = SPEC_MIN_DRAFTS;
    float best_rate = 0.0f, tokens = 1.0f, p = 1.0f;
    for (int k = 1; k <= SPEC_MAX_DRAFTS; ++k) {
        p *= m->spec_acc[k - 1];
        tokens += p;
        if (k < SPEC_MIN_DRAFTS) continue;
        const float model = 1.0f + SPEC_COST_PER_DRAFT * k;
        const float cost = m->spec_ms[k - 1] > 0.0f ? m->spec_ms[k - 1]
                           : ref         ? m->spec_ms[ref - 1] * model / (1.0f + SPEC_COST_PER_DRAFT * ref)
                                         : model;
        if (tokens / cost > best_rate) best_rate = tokens / cost, best = k;
    }
    return best;
}

// A pass of k drafts kept `accepted` of them: drafts 1 .. accepted were accepted, draft accepted + 1 (if any) not
static void spec_observe(inference_model_t* m, int k, int accepted) {
    for (int j = 0; j < k && j <= accepted; ++j)
        m->spec_acc[j] += SPEC_RATE * ((j < accepted ? 1.0f : 0.0f) - m->spec_acc[j]);
}

// Speculative decoding (an MTP draft head is loaded): decode token_id like forward_next, but verify the head's
// drafts of the tokens after it in the same pass (sp.k of them; 0: the step decoded token_id alone). The drafts
// sit decoded after n_past, not counted yet: the caller accepts them one by one with accept_draft, or keeps a
// prefix with qwen_model_rollback.
static int forward_next_spec(inference_model_t* m, int token_id, SpecPass& sp) {
    sp.k = sp.i = 0;
    if (!m->mtp || !m->kvc || !m->qwen) return forward_next(m, token_id);
    const int B = kv_B(m);
    // the state snapshot of a block boundary inside the pass must not contain drafts after it: only drafts up to
    // the first snapshot boundary (one right after token_id: none)
    int k = std::min(spec_drafts_for_pass(m), m->n_ctx - m->n_past - 1);
    for (int j = 0; j < k; ++j)
        if (m->store_dirfd >= 0 && (m->n_past + 1 + j) % B == 0 && snapshot_boundary(m, (m->n_past + 1 + j) / B)) {
            k = j;
            break;
        }
    if (k <= 0) return forward_next(m, token_id);
    int toks[SPEC_MAX_DRAFTS + 1] = {token_id};
    const auto t0 = std::chrono::steady_clock::now();
    const int n = qwen_model_mtp_drafts(m->qwen, token_id, m->n_past, toks + 1, k);
    if (n <= 0) return forward_next(m, token_id);
    if (ensure_writable(m, m->n_past, n + 1) != 0) return -1;
    if (qwen_model_forward_verify(m->qwen, toks, n + 1, m->n_past) != 0) {
        set_error("forward pass failed: %s", qwen_model_last_error());
        return -1;
    }
    if (m->spec_drafts == 0) {  // adaptive: the time of a pass with n drafts
        const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        float& avg = m->spec_ms[n - 1];
        avg = avg > 0.0f ? avg + SPEC_RATE * (ms - avg) : ms;
    }
    ++m->n_past;
    m->seq.push_back(token_id);
    for (int j = 0; j < n; ++j) sp.d[j] = toks[1 + j];
    sp.k = n;
    return store_after_forward(m);
}

// A verified draft is the token that follows: count it as decoded
static int accept_draft(inference_model_t* m, int draft) {
    ++m->n_past;
    m->seq.push_back(draft);
    return store_after_forward(m);
}

int inference_prefill(inference_model_t* m, const int32_t* tokens, int n) {
    clear_error();
    if (!m || !m->kvc || !m->qwen || !tokens || n <= 0) { set_error("Invalid state or arguments"); return -1; }
    if (m->n_past + n > m->n_ctx) { set_error("context full (%d + %d > %d tokens)", m->n_past, n, m->n_ctx); return INFERENCE_ERR_CONTEXT_FULL; }
    // With a block store, the prefill is cut at the block boundaries that get a state snapshot
    const bool snap = m->store_dirfd >= 0 && state_bytes(m) > 0;
    const int B = kv_B(m);
    for (int done = 0; done < n;) {
        int len = n - done;
        if (snap)
            for (int b = (m->n_past / B + 1) * B; b < m->n_past + len; b += B)
                if (snapshot_boundary(m, b / B)) {
                    len = b - m->n_past;
                    break;
                }
        if (ensure_writable(m, m->n_past, len) != 0) return -1;
        if (qwen_model_prefill(m->qwen, (const int*)tokens + done, len, m->n_past, nullptr) != 0) {
            set_error("prefill failed: %s", qwen_model_last_error());
            return -1;
        }
        m->n_past += len;
        m->seq.insert(m->seq.end(), tokens + done, tokens + done + len);
        done += len;
        if (store_after_forward(m) != 0) return -1;
    }
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
    seq_reset(model);
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
    o.checkpoint_tail = 16;
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

// Save the recurrent state as the checkpoint at the current position (kInferenceCheckpoints slots, the oldest
// one is replaced)
static const int kInferenceCheckpoints = 4;

static int save_checkpoint(inference_model_t* m) {
    int slot = (int)m->ckpts.size();
    if (slot >= kInferenceCheckpoints) {
        size_t oldest = 0;
        for (size_t i = 1; i < m->ckpts.size(); ++i)
            if (m->ckpts[i].age < m->ckpts[oldest].age) oldest = i;
        slot = m->ckpts[oldest].slot;
        m->ckpts.erase(m->ckpts.begin() + (long)oldest);
    }
    if (qwen_model_state_save(m->qwen, slot) != 0) { set_error("recurrent state checkpoint failed"); return -1; }
    m->ckpts.push_back({m->n_past, slot, ++m->ckpt_clock});
    return 0;
}

// Prefill tokens [0, n) of the prompt on top of the current sequence; with a recurrent model and tail > 0, the
// state is saved min(tail, n / 2) tokens before the end (where the next turn's prompt is likely to diverge: a chat template
// re-renders the generation prompt and the reply differently)
static int prefill_with_checkpoint(inference_model_t* m, const int32_t* tokens, int n, int tail) {
    if (n <= 0) return 0;
    tail = std::min(tail, n / 2);  // a short prompt still gets a checkpoint, half way
    if (tail > 0 && qwen_model_is_recurrent(m->qwen)) {
        int rc = inference_prefill(m, tokens, n - tail);
        if (rc == 0) rc = save_checkpoint(m);
        if (rc == 0) rc = inference_prefill(m, tokens + n - tail, tail);
        return rc;
    }
    return inference_prefill(m, tokens, n);
}

// Make the sequence equal to prompt[0, n), reusing the longest common prefix with the current one: append when
// the current sequence is a prefix of the prompt; otherwise rewind to the latest point at or before the
// divergence (any position for attention-only models; a checkpoint for recurrent ones; else a new sequence).
// At least the prompt's last token is always decoded, so the logits are its.
// Deepest store resume point for prompt p: the most blocks 0 .. j - 1 whose .kvb files are all present, ending at a
// block with a state snapshot (recurrent models). Leaves at least the last prompt token to decode. Returns j * B and
// the blocks' hashes.
static int store_lookup(inference_model_t* m, const int32_t* p, int n, std::vector<inference_model::Hash>* hs) {
    hs->clear();
    if (m->store_dirfd < 0) return 0;
    const int B = kv_B(m), K = (n - 1) / B;
    const bool recurrent = state_bytes(m) > 0;
    int best = 0;
    for (int j = 1; j <= K; ++j) {
        const inference_model::Hash h = block_hash(j > 1 ? (*hs)[j - 2] : m->store_root, p + (size_t)(j - 1) * B, B);
        if (!store_has(m, store_name(h, "kvb"), kv_cache_block_bytes(m->kvc))) break;
        hs->push_back(h);
        if (!recurrent || store_has(m, store_name(h, "rst"), state_bytes(m))) best = j;
    }
    hs->resize((size_t)best);
    return best * B;
}

// Make the sequence the store's first j = start / B blocks of prompt p: map the block files (blocks the sequence
// already holds stay as they are) and load the state snapshot
static int store_attach(inference_model_t* m, const int32_t* p, int start, const std::vector<inference_model::Hash>& hs) {
    const int j = start / kv_B(m);
    if (qwen_model_sync(m->qwen) != 0) { set_error("sync failed: %s", qwen_model_last_error()); return -1; }
    for (int i = 0; i < j; ++i) {
        const std::string name = store_name(hs[i], "kvb");
        store_touch(m, name);
        if (i < (int)m->chain.size() && m->chain[i] == hs[i]) continue;  // the same block is already there
        const int fd = openat(m->store_dirfd, name.c_str(), O_RDONLY | O_CLOEXEC);
        const int rc = fd >= 0 ? kv_cache_slot_attach(m->kvc, i, fd) : -1;
        if (fd >= 0) close(fd);
        if (rc != 0) { set_error("store: cannot map KV block %s", name.c_str()); return -1; }
        ++m->st_attached;
    }
    size_t bytes = 0;
    void* mem = qwen_model_state_memory(m->qwen, &bytes);
    if (bytes > 0) {
        const std::string name = store_name(hs[j - 1], "rst");
        const int fd = openat(m->store_dirfd, name.c_str(), O_RDONLY | O_CLOEXEC);
        size_t done = 0;
        while (fd >= 0 && done < bytes) {
            const ssize_t r = pread(fd, (char*)mem + done, bytes - done, (off_t)done);
            if (r <= 0) break;
            done += (size_t)r;
        }
        if (fd >= 0) close(fd);
        if (done != bytes) {
            seq_reset(m);  // the state is now partly overwritten: nothing of the old sequence is valid
            set_error("store: cannot read state snapshot %s", name.c_str());
            return -1;
        }
        store_touch(m, name);
        ++m->st_snap_restored;
    }
    m->n_past = start;
    m->seq.assign(p, p + start);
    m->chain = hs;
    m->published = (size_t)j;
    m->ckpts.clear();  // in-memory checkpoints belong to the old sequence
    m->st_tokens += start;
    return 0;
}

// Make the sequence equal to prompt[0, n), reusing as much as possible: the current sequence when the prompt
// extends it; else the latest point at or before the divergence (any position for attention-only models, an
// in-memory checkpoint for recurrent ones); or the block store, when it reaches further; else a new sequence.
// At least the prompt's last token is always decoded, so the logits are its.
static int sync_prompt(inference_model_t* m, const int32_t* p, int n, int tail) {
    size_t L = 0;
    while (L < m->seq.size() && L < (size_t)n && m->seq[L] == p[L]) ++L;
    int start = 0, ck = -1;
    const bool extend = L == m->seq.size() && (L < (size_t)n || m->n_past > 0);
    if (extend) {
        start = (int)L;  // the prompt extends the sequence (or equals it: its logits are current)
    } else {
        const int target = std::min((int)L, n - 1);
        if (!qwen_model_is_recurrent(m->qwen)) {
            start = target;
            // the sliding-window ring must still hold the window before it; else from the start
            if (!qwen_model_ring_reuse_ok(m->qwen, start)) start = 0;
        } else {
            for (size_t i = 0; i < m->ckpts.size(); ++i)
                if (m->ckpts[i].pos <= target && (ck < 0 || m->ckpts[i].pos > m->ckpts[ck].pos)) ck = (int)i;
            start = ck >= 0 ? m->ckpts[ck].pos : 0;
        }
    }
    std::vector<inference_model::Hash> hs;
    const int from_store = store_lookup(m, p, n, &hs);
    if (from_store > start) {
        if (store_attach(m, p, from_store, hs) != 0) return -1;
        start = from_store;
    } else if (!extend) {
        if (ck >= 0 && qwen_model_state_restore(m->qwen, m->ckpts[ck].slot) != 0) {
            set_error("recurrent state restore failed");
            return -1;
        }
        if (start == 0) {
            seq_reset(m);
        } else {
            seq_truncate(m, start);
            std::vector<inference_model::Checkpoint> keep;  // later checkpoints belong to the discarded tokens
            for (const auto& c : m->ckpts)
                if (c.pos <= start) keep.push_back(c);
            m->ckpts.swap(keep);
        }
    }
    m->last_reused = start;
    m->last_prefilled = n - start;
    return prefill_with_checkpoint(m, p + start, n - start, tail);
}

// ---- Block store API

int inference_store_open(inference_model_t* m, const char* dir, int snapshot_every) {
    clear_error();
    if (!m || !m->kvc || !m->qwen || !dir || !dir[0] || snapshot_every < 0) {
        set_error("Invalid state or arguments (call inference_kv_cache_init first)");
        return INFERENCE_ERR_INVALID_ARG;
    }
    if (m->kvc_swa) {  // a block holds only the global layers; the ring keeps no history to store
        set_error("store: the sliding-window layers' KV cache is a ring (inference_set_swa_ring(m, 0) before "
                  "inference_kv_cache_init keeps every layer's history, as a block store needs)");
        return INFERENCE_ERR_INVALID_ARG;
    }
    inference_store_close(m);
    const int dfd = open(dir, O_DIRECTORY | O_RDONLY | O_CLOEXEC);
    if (dfd < 0) { set_error("store: cannot open directory %s: %s", dir, strerror(errno)); return INFERENCE_ERR_INVALID_ARG; }
    const int probe = openat(dfd, ".", O_TMPFILE | O_RDWR | O_CLOEXEC, 0644);
    if (probe < 0) {
        set_error("store: %s does not support anonymous files (O_TMPFILE): %s", dir, strerror(errno));
        close(dfd);
        return INFERENCE_ERR_INVALID_ARG;
    }
    close(probe);
    // Blocks are published by linking a slot's own file, so the slots must live on the store's filesystem
    if (kv_cache_set_dir(m->kvc, dir) != 0) {
        set_error("store: cannot move the KV slots to %s", dir);
        close(dfd);
        return INFERENCE_ERR_KV_CACHE;
    }
    // Root of every chain: the model (file size and its first 4 MB: header, metadata, the first tensors) and
    // the layout of the blocks and snapshots
    sha256_ctx c;
    sha256_init(&c);
    const char tag[] = "rocm-inference kv-store v1";
    sha256_update(&c, tag, sizeof(tag));
    size_t gsize = 0;
    const void* gbase = gguf_map_base(m->gguf, &gsize);
    const uint64_t layout[] = {(uint64_t)gsize, (uint64_t)kv_cache_block_tokens(m->kvc), (uint64_t)m->n_kv_layers,
                               (uint64_t)kv_cache_d_kv(m->kvc), (uint64_t)kv_cache_block_bytes(m->kvc),
                               (uint64_t)state_bytes(m), (uint64_t)kv_cache_type(m->kvc)};
    sha256_update(&c, layout, sizeof(layout));
    if (gbase) sha256_update(&c, gbase, std::min(gsize, (size_t)4 << 20));
    for (int i = 1; i < gguf_n_maps(m->gguf); ++i) {  // the other files of a split model: their sizes
        size_t sz = 0;
        gguf_map_at(m->gguf, i, &sz);
        const uint64_t v = sz;
        sha256_update(&c, &v, sizeof(v));
    }
    sha256_final(&c, m->store_root.b);
    m->store_dir = dir;
    m->store_dirfd = dfd;
    m->store_snapshot_every = snapshot_every;
    m->chain.clear();
    m->published = 0;  // the current sequence's full blocks are published by the next forward pass
    m->st_published = m->st_attached = m->st_snap_saved = m->st_snap_restored = m->st_tokens = 0;
    return 0;
}

int inference_store_block_names(const inference_model_t* m, const int32_t* tokens, int n, char* out, int max_len) {
    if (!m || m->store_dirfd < 0 || n < 0 || (n > 0 && !tokens) || max_len < 0 || (max_len > 0 && !out)) {
        return INFERENCE_ERR_INVALID_ARG;
    }
    const int B = kv_B(m);
    std::string names;
    inference_model::Hash h = m->store_root;
    for (int j = 0; (j + 1) * B <= n; ++j) {
        h = block_hash(h, tokens + (size_t)j * B, B);
        names += store_name(h, "kvb") + "\n";
    }
    if (max_len > 0) {
        const size_t len = std::min(names.size(), (size_t)max_len - 1);
        memcpy(out, names.data(), len);
        out[len] = '\0';
    }
    return (int)names.size();
}

void inference_store_close(inference_model_t* m) {
    if (!m || m->store_dirfd < 0) return;
    close(m->store_dirfd);
    m->store_dirfd = -1;
    m->store_dir.clear();
    m->chain.clear();
    m->published = 0;
}

int inference_store_get_stats(const inference_model_t* m, inference_store_stats* st) {
    if (!m || !st) return INFERENCE_ERR_INVALID_ARG;
    st->blocks_published = m->st_published;
    st->blocks_attached = m->st_attached;
    st->snapshots_saved = m->st_snap_saved;
    st->snapshots_restored = m->st_snap_restored;
    st->tokens_from_store = m->st_tokens;
    st->block_tokens = m->kvc ? kv_cache_block_tokens(m->kvc) : 0;
    st->block_bytes = m->kvc ? (long long)kv_cache_block_bytes(m->kvc) : 0;
    st->snapshot_bytes = m->qwen ? (long long)state_bytes(m) : 0;
    return 0;
}

const char* inference_chat_template(inference_model_t* m, const char* name) {
    if (!m || !m->gguf) return nullptr;
    const std::string key = name && name[0] ? std::string("tokenizer.chat_template.") + name : "tokenizer.chat_template";
    auto it = m->chat_templates.find(key);
    if (it == m->chat_templates.end()) {
        const int64_t idx = gguf_find_kv(m->gguf, key.c_str());
        const char* str = idx >= 0 ? gguf_kv_str_val(m->gguf, idx) : nullptr;
        if (!str || !str[0]) return nullptr;
        it = m->chat_templates.emplace(key, std::string(str)).first;
    }
    return it->second.c_str();
}

int inference_bos_id(const inference_model_t* m) { return m && m->tokenizer ? tokenizer_bos_id(m->tokenizer) : -1; }
int inference_eos_id(const inference_model_t* m) { return m && m->tokenizer ? tokenizer_eos_id(m->tokenizer) : -1; }

int inference_last_prompt_stats(const inference_model_t* m, int* reused, int* prefilled) {
    if (!m) return INFERENCE_ERR_INVALID_ARG;
    if (reused) *reused = m->last_reused;
    if (prefilled) *prefilled = m->last_prefilled;
    return 0;
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
    // Block store: also snapshot at the last block boundary of this prompt (a later request with the same prefix
    // resumes there)
    m->store_snapshot_at = ((o.reuse_prefix ? 0 : m->n_past) + n_prompt) / kv_B(m);
    struct ResetSnapshotAt {
        inference_model_t* m;
        ~ResetSnapshotAt() { m->store_snapshot_at = -1; }
    } reset_snapshot_at{m};
    if (o.reuse_prefix && n_prompt > 0) {
        const int rc = sync_prompt(m, prompt_ids, n_prompt, o.checkpoint_tail);
        if (rc != 0) return rc;
    } else if (n_prompt > 0) {
        m->last_reused = m->n_past;
        m->last_prefilled = n_prompt;
        const int rc = prefill_with_checkpoint(m, prompt_ids, n_prompt, o.checkpoint_tail);
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
    // Speculative decoding (a draft head loaded): a verify pass decoded the last emitted token and the head's
    // drafts after it. Each next token is sampled as always, from the logits after the tokens before it (row i
    // of the pass); while it is the next draft, that draft is already decoded; at the first one that differs the
    // model rolls back to the tokens kept. Either way the tokens are those of plain decoding, sampled from the
    // same logits in the same order.
    SpecPass sp;
    while (stop == INFERENCE_STOP_NONE) {
        if (emitted >= o.max_new_tokens) { stop = INFERENCE_STOP_MAX_TOKENS; break; }
        const float* lg = sp.k > 0 ? qwen_model_verify_logits(m->qwen, sp.i) : qwen_model_logits(m->qwen);
        const int next = sampler_sample(smp, lg, m->vocab_size);
        const bool eog = !o.ignore_eog && tokenizer_is_eog(m->tokenizer, next);
        bool decoded = false;
        if (sp.k > 0) {
            if (sp.i < sp.k && next == sp.d[sp.i] && !eog) {
                rc = accept_draft(m, next);
                ++sp.i;
                decoded = true;
            } else {
                if (!eog) spec_observe(m, sp.k, sp.i);  // (an end of generation says nothing about the head)
                if (sp.i < sp.k && qwen_model_rollback(m->qwen, sp.i + 1) != 0) {
                    set_error("rollback failed: %s", qwen_model_last_error());
                    rc = -1;
                }
                sp.k = 0;
            }
            if (rc != 0) break;
        }
        if (eog) { stop = INFERENCE_STOP_EOG; break; }
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

        // Every emitted token joins the sequence, so a later call continues right after it (with a draft head,
        // also verifying the head's drafts of the tokens after it, unless generation stops here)
        if (decoded) continue;
        rc = stop == INFERENCE_STOP_NONE && emitted < o.max_new_tokens ? forward_next_spec(m, next, sp)
                                                                      : forward_next(m, next);
        if (rc == INFERENCE_ERR_CONTEXT_FULL) {
            rc = 0;
            if (stop == INFERENCE_STOP_NONE) stop = INFERENCE_STOP_CONTEXT_FULL;
        }
        if (rc != 0) break;
    }
    if (sp.k > 0 && sp.i < sp.k && qwen_model_rollback(m->qwen, sp.i + 1) != 0 && rc == 0) {  // drafts decoded
        set_error("rollback failed: %s", qwen_model_last_error());
        rc = -1;
    }
    if (rc == 0 && o.on_token && !pending.empty()) (void)o.on_token(o.user, -1, pending.data(), (int)pending.size());
    sampler_free(smp);
    if (n_out) *n_out = emitted;
    if (reason) *reason = stop;
    return rc;
}

// ---- Step-wise generation (inference_stream_*): what inference_generate_stream does, one token per call, so that
// a caller can interleave several sequences (no speculative decoding)
struct inference_stream {
    inference_model_t* m = nullptr;
    int seq = 0;
    inference_gen_options o{};
    std::vector<std::string> stops;
    sampler_t* smp = nullptr;
    std::string pending, out;
    std::vector<char> buf = std::vector<char>(256);
    int emitted = 0;
    inference_stop_reason stop = INFERENCE_STOP_NONE;
    bool done = false;
};

inference_stream_t* inference_stream_begin(inference_model_t* m, const int32_t* prompt_ids, int n_prompt,
                                           const inference_gen_options* options) {
    clear_error();
    if (!m || !m->qwen || !m->tokenizer || n_prompt < 0 || (n_prompt > 0 && !prompt_ids)) {
        set_error("Invalid state or arguments");
        return nullptr;
    }
    const inference_gen_options o = options ? *options : inference_gen_default_options();
    m->store_snapshot_at = ((o.reuse_prefix ? 0 : m->n_past) + n_prompt) / kv_B(m);
    int rc = 0;
    if (o.reuse_prefix && n_prompt > 0) {
        rc = sync_prompt(m, prompt_ids, n_prompt, o.checkpoint_tail);
    } else if (n_prompt > 0) {
        m->last_reused = m->n_past;
        m->last_prefilled = n_prompt;
        rc = prefill_with_checkpoint(m, prompt_ids, n_prompt, o.checkpoint_tail);
    } else if (m->n_past == 0) {
        set_error("nothing to continue: empty prompt and an empty sequence");
        rc = INFERENCE_ERR_INVALID_ARG;
    }
    m->store_snapshot_at = -1;
    if (rc != 0) return nullptr;
    inference_stream_t* st = new inference_stream();
    st->m = m;
    st->seq = m->cur_seq;
    st->o = o;
    st->o.stop_strings = nullptr;  // copied below
    for (int i = 0; i < o.n_stop_strings; ++i)
        if (o.stop_strings && o.stop_strings[i] && o.stop_strings[i][0]) st->stops.emplace_back(o.stop_strings[i]);
    st->smp = sampler_create(&o.sampler);
    for (int i = 0; i < n_prompt; ++i) sampler_accept(st->smp, prompt_ids[i]);
    return st;
}

// A step's first half: sample the stream's next token and its text (ready in st->out). Returns the token to decode
// into the sequence, or -1 (none: the stream stops)
static int stream_pick(inference_stream_t* st) {
    inference_model_t* m = st->m;
    st->out.clear();
    if (st->done) return -1;
    int next = -1;
    if (st->emitted >= st->o.max_new_tokens) {
        st->stop = INFERENCE_STOP_MAX_TOKENS;
    } else {
        next = sampler_sample(st->smp, qwen_model_logits(m->qwen), m->vocab_size);
        if (!st->o.ignore_eog && tokenizer_is_eog(m->tokenizer, next)) {
            st->stop = INFERENCE_STOP_EOG;
            next = -1;
        }
    }
    if (next < 0) return -1;
    sampler_accept(st->smp, next);
    ++st->emitted;
    int len = tokenizer_detokenize(m->tokenizer, &next, 1, st->buf.data(), (int)st->buf.size());
    if (len >= (int)st->buf.size()) {
        st->buf.resize((size_t)len + 1);
        len = tokenizer_detokenize(m->tokenizer, &next, 1, st->buf.data(), (int)st->buf.size());
    }
    st->pending.append(st->buf.data(), (size_t)std::max(len, 0));
    // what can no longer become part of a stop string or of a split UTF-8 character
    size_t ready, cut = std::string::npos;
    for (const std::string& x : st->stops) cut = std::min(cut, st->pending.find(x));
    if (cut != std::string::npos) {
        ready = cut;
        st->stop = INFERENCE_STOP_STRING;
    } else {
        ready = st->pending.size() - std::max(utf8_incomplete_tail(st->pending), stop_prefix_tail(st->pending, st->stops));
    }
    st->out.assign(st->pending, 0, ready);
    st->pending.erase(0, st->stop == INFERENCE_STOP_STRING ? st->pending.size() : ready);
    return next;
}

// The decode of the picked token returned rc: a full context stops the stream
static int stream_decoded(inference_stream_t* st, int rc) {
    if (rc == INFERENCE_ERR_CONTEXT_FULL) {
        if (st->stop == INFERENCE_STOP_NONE) st->stop = INFERENCE_STOP_CONTEXT_FULL;
        return 0;
    }
    return rc;
}

// A step's end: a stopped stream's rest of the text goes with its last step
static int stream_finish(inference_stream_t* st, int32_t* token, int next, const char** text, int* text_len,
                         inference_stop_reason* reason) {
    if (!st->done && st->stop != INFERENCE_STOP_NONE) {
        st->out += st->pending;
        st->pending.clear();
        st->done = true;
    }
    if (token) *token = next;
    if (text) *text = st->out.c_str();
    if (text_len) *text_len = (int)st->out.size();
    if (reason) *reason = st->stop;
    return st->done ? 0 : 1;
}

int inference_stream_step(inference_stream_t* st, int32_t* token, const char** text, int* text_len,
                          inference_stop_reason* reason) {
    clear_error();
    if (token) *token = -1;
    if (text) *text = "";
    if (text_len) *text_len = 0;
    if (!st) { set_error("NULL stream"); return INFERENCE_ERR_INVALID_ARG; }
    inference_model_t* m = st->m;
    if (!st->done && m->cur_seq != st->seq && inference_seq_select(m, st->seq) != 0) return -1;
    const int next = stream_pick(st);
    // every emitted token joins the sequence: a later call continues right after it
    if (next >= 0) {
        const int rc = stream_decoded(st, forward_next(m, next));
        if (rc != 0) return rc;
    }
    return stream_finish(st, token, next, text, text_len, reason);
}

int inference_stream_step_batch(inference_stream_t* const* streams, int n, inference_step_result* results) {
    clear_error();
    if (n <= 0 || !streams || !results) { set_error("Invalid arguments"); return INFERENCE_ERR_INVALID_ARG; }
    inference_model_t* m = streams[0] ? streams[0]->m : nullptr;
    for (int i = 0; i < n; ++i) {
        results[i] = inference_step_result{-1, "", 0, INFERENCE_STOP_NONE, 0};
        bool dup = false;
        for (int j = 0; j < i; ++j) dup |= streams[j] && streams[i] && streams[j]->seq == streams[i]->seq;
        if (!streams[i] || streams[i]->m != m || dup) {
            set_error("stream %d: NULL, of another model, or a second one of its sequence", i);
            return INFERENCE_ERR_INVALID_ARG;
        }
    }
    const int cur = m->cur_seq;
    // pick every stream's token (its sequence's logits), and make the sequences' next positions writable
    std::vector<int> next((size_t)n, -1), rc((size_t)n, 0);
    std::vector<int32_t> toks;
    std::vector<qwen_seq_token> rows;
    std::vector<int> row_of;  // the streams in the batch
    int err = 0;
    for (int i = 0; i < n && !err; ++i) {
        inference_stream_t* st = streams[i];
        if (st->done) continue;
        if (inference_seq_select(m, st->seq) != 0) { err = -1; break; }
        next[i] = stream_pick(st);
        if (next[i] < 0) continue;
        if (m->n_past >= m->n_ctx) {
            set_error("context full (%d tokens)", m->n_ctx);
            rc[i] = INFERENCE_ERR_CONTEXT_FULL;
        } else if (ensure_writable(m, m->n_past, 1) != 0) {
            err = -1;
        } else {
            toks.push_back(next[i]);
            rows.push_back(qwen_seq_token{st->seq, m->n_past, m->kvc, m->kvc_swa});
            row_of.push_back(i);
        }
    }
    if (!err && inference_seq_select(m, cur) != 0) err = -1;
    // one forward pass for all (or one by one where batched decoding is not there, and for a single one)
    int frc = rows.size() >= 2 && !err ? qwen_model_forward_seqs(m->qwen, toks.data(), rows.data(), (int)rows.size())
                                      : INFERENCE_ERR_NOT_IMPLEMENTED;
    if (frc != 0 && frc != INFERENCE_ERR_NOT_IMPLEMENTED) {
        set_error("batched forward pass failed: %s", qwen_model_last_error());
        err = -1;
    }
    for (size_t r = 0; r < rows.size() && !err; ++r) {
        const int i = row_of[r];
        if (inference_seq_select(m, streams[i]->seq) != 0) { err = -1; break; }
        if (frc == 0) {
            ++m->n_past;
            m->seq.push_back(toks[r]);
            rc[i] = store_after_forward(m);
        } else {
            rc[i] = forward_next(m, toks[r]);
        }
    }
    if (!err && inference_seq_select(m, cur) != 0) err = -1;
    if (err) return err;
    for (int i = 0; i < n; ++i) {
        inference_step_result& o = results[i];
        o.rc = rc[i] ? stream_decoded(streams[i], rc[i]) : 0;
        if (o.rc == 0) o.rc = stream_finish(streams[i], &o.token, next[i], &o.text, &o.text_len, &o.reason);
        else o.text = "";
    }
    return 0;
}

int inference_stream_tokens(const inference_stream_t* st) { return st ? st->emitted : 0; }
int inference_stream_seq(const inference_stream_t* st) { return st ? st->seq : -1; }

void inference_stream_free(inference_stream_t* st) {
    if (!st) return;
    sampler_free(st->smp);
    delete st;
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
