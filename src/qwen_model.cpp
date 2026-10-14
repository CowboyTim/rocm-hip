// qwen_model.cpp -- Phase 4: Qwen 3 dense transformer forward pass host orchestration (minimal skeleton).
// Loads all kernels, allocates unified pinned buffers, runs token-by-token forward pass.

#include "qwen_model.h"
#include "inference.h"
#include "gguf_mmap.h"
#include "kv_cache.h"
#include "fp4_common.h"
#include "inference_internal.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>


using namespace rocmfp4;

// Thread-local error buffer for qwen_model errors
static thread_local char g_qwen_error[1024] = {0};

static void qwen_set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_qwen_error, sizeof(g_qwen_error), fmt, ap);
    va_end(ap);
}

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
     fprintf(stderr, "[KERNELS] Loading %s (%zd bytes)... ", co, (size_t)st.st_size); fflush(stderr);
     hipError_t err1 = hipModuleLoadData(&m, data);
     hipError_t err2 = hipSuccess; hipModule_t tmp_m = nullptr;
     if (err1 == hipSuccess) { err2 = hipModuleGetFunction(&f, m, sym); }
     bool ok = (err1 == hipSuccess) && (err2 == hipSuccess);

     if (!ok) { fprintf(stderr, " [err=%d/%d]", (int)err1, (int)err2); }
     fprintf(stderr, "\n"); fflush(stderr);
     munmap(data, st.st_size); close(fd); return ok;

}

static bool load_all_kernels() {
    fprintf(stderr, "[KERNELS] Loading %d kernel modules...\n", 8); fflush(stderr);
    (void)hipInit(0);
    bool ok = load_module("gemm_act_fp4.co", "gemm_act_fp4", mod_gemm_act_fp4, fn_gemm_act_fp4);

    return ok &&
           load_module("swiglu.co", "swiglu_fp4", mod_swiglu, fn_swiglu) &&
           load_module("fp16_gemm.co", "fp16_gemm", mod_fp16_gemm, fn_fp16_gemm) &&
           load_module("rope_embedding.co", "rope_rotate_half", mod_rope, fn_rope) &&
           load_module("softmax_masked.co", "softmax_masked", mod_softmax, fn_softmax) &&
           load_module("layernorm.co", "rmsnorm", mod_layernorm, fn_layernorm) &&
           load_module("kv_cache_store.co", "kv_store", mod_kv_store, fn_kv_store) &&
           load_module("embedding_lookup.co", "embedding_lookup", mod_embedding, fn_embedding);
}

static void unload_all_kernels() {
    if (mod_gemm_act_fp4) (void)hipModuleUnload(mod_gemm_act_fp4);
    if (mod_swiglu) (void)hipModuleUnload(mod_swiglu);
    if (mod_fp16_gemm) (void)hipModuleUnload(mod_fp16_gemm);
    if (mod_rope) (void)hipModuleUnload(mod_rope);
    if (mod_softmax) (void)hipModuleUnload(mod_softmax);
    if (mod_layernorm) (void)hipModuleUnload(mod_layernorm);
    if (mod_kv_store) (void)hipModuleUnload(mod_kv_store);
    if (mod_embedding) (void)hipModuleUnload(mod_embedding);
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
    __half* act = nullptr; void* d_act = nullptr;     // SiLU(gate)*up, FP16 input to ffn_down
    int ff = 0;                                        // FFN width (ffn_gate/ffn_up output dim)

    void* base = nullptr; size_t total = 0; int fd = -1; bool freed = false;

    bool alloc(int E, int HD, int HQ, int HKV, int CTX, int DQ, int DKV, int FF) {
        const int hd2 = HD / 2;
        ff = FF;
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
            (size_t)FF * sizeof(float),                    // gate_fp32
            (size_t)FF * sizeof(float),                    // up_fp32
            (size_t)E * sizeof(float),                     // down_fp32
            (size_t)CTX * HD * sizeof(__half),             // vexpB
            (size_t)FF * sizeof(__half),                   // act
        };
        size_t total = 0;
        for (size_t sz : layout) { total = (total + 127) & ~127; total += sz; }
        this->total = total;

        fd = open("/dev/shm/qwen_layer_scratch.bin", O_RDWR | O_CREAT, 0666);
        if (fd < 0) return false;
        ftruncate(fd, total);
        base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);


        size_t off = 0;
        auto get = [&](size_t sz, void** hptr, void** dptr) {
            off = (off + 127) & ~127;
            *hptr = (char*)base + off;
            // For registered host memory via /dev/shm mmap: device pointer IS same as host.
            // hipHostGetDevicePointer returns NULL on Strix Halo for this case.
            *dptr = *hptr;
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
        get(layout[18], (void**)&act, &d_act);

        (void)hipHostRegister(base, total, hipHostRegisterDefault);
        return true;
    }
    void free() {
        if (freed || !base) return;
        freed = true;
        fprintf(stderr, "[FREE] LayerScratch: unmap base=%p total=%zu fd=%d\n",
                (void*)base, total, fd); fflush(stderr);
        (void)hipHostUnregister(base);
        munmap(base, total);
        if (fd >= 0) { close(fd); fd = -1; }
    }
};

// ==================== Qwen Model ====================
struct QwenModel {
    inference_model* eng_model = nullptr;
    std::vector<LayerScratch> layer_scratch;
    int n_layer = 28, n_embd = 4096, n_head = 32, n_head_kv = 8, head_dim = 128;
    int vocab_size = 151936, n_ctx = 4096;
    const __half* d_embed_table_copy = nullptr;  // Use proper type instead of void*

    ~QwenModel() {
        for (auto& s : layer_scratch) s.free();
        unload_all_kernels();
    }

    // Phase 15.4 helpers: find KV index by key, extract int64 value, read tensor type by name

    static int64_t find_kv_idx_gguf(gguf_model_t* m, const char* key) {
        if (!m || !key) return -1;
        for (int64_t i = 0; i < gguf_n_kv(m); ++i) {
            if (strcmp(gguf_kv_key(m, i), key) == 0) return i;
        }
        return -1;
    }

    static int64_t kv_as_int64_from_gguf(gguf_model_t* m, int64_t idx) {
        if (!m || idx < 0) return 0;
        uint8_t u8v; int8_t i8v; uint16_t u16v; int16_t i16v;
        uint32_t u32v; int32_t i32v; uint64_t u64v; int64_t i64v;
        switch (gguf_kv_type(m, idx)) {
            case 0:  (void)gguf_kv_get_u8(m, idx, &u8v);     return u8v;
            case 1:  (void)gguf_kv_get_i8(m, idx, &i8v);     return i8v;
            case 2:  (void)gguf_kv_get_u16(m, idx, &u16v);   return u16v;
            case 3:  (void)gguf_kv_get_i16(m, idx, &i16v);   return i16v;
            case 4:  (void)gguf_kv_get_u32(m, idx, &u32v);   return u32v;
            case 5:  (void)gguf_kv_get_i32(m, idx, &i32v);   return i32v;
            case 10: (void)gguf_kv_get_u64(m, idx, &u64v);   return u64v;
            case 11: (void)gguf_kv_get_i64(m, idx, &i64v);   return i64v;
            default: return 0;
        }
    }

    bool init(inference_model* model) {
        eng_model = model;
        if (!load_all_kernels()) return false;

        // Phase 15.4: Read full_attention_interval from GGUF KV for hybrid architecture support
        const char* fa_keys[] = {"qwen35.full_attention_interval", "ssm.full_attention_interval"};
        int64_t fa_idx = -1;
        for (int i = 0; i < 2 && fa_idx < 0; ++i) {
            fa_idx = find_kv_idx_gguf(model->gguf, fa_keys[i]);
        }
        model->full_attention_interval = (fa_idx >= 0) ? (int)kv_as_int64_from_gguf(model->gguf, fa_idx) : 0;
        fprintf(stderr, "[INIT] full_attention_interval=%d (fa_idx=%ld)\n", model->full_attention_interval, fa_idx); fflush(stderr);

        // Use dynamic dimensions from engine model (Phase 15.1)
        const int E = model->n_embd, HD = model->head_dim, HQ = model->n_head, HKV = model->n_head_kv, CTX = model->n_ctx;
        const int DQ = HQ * HD, DKV = HKV * HD;

        n_layer = model->n_layer;
        n_embd = model->n_embd;
        n_head = model->n_head;
        n_head_kv = model->n_head_kv;
        head_dim = model->head_dim;
        vocab_size = model->vocab_size;
        n_ctx = model->n_ctx;

        // FFN width from the loaded ffn_gate/ffn_up tensors (output dim N); falls back to E.
        int FF = E;
        for (const auto& L : model->layers)
            FF = std::max(FF, std::max(L.W_gate.N, L.W_up.N));
        fprintf(stderr, "[INIT] ffn width FF=%d\n", FF); fflush(stderr);

        fprintf(stderr, "[SCRATCH] Starting scratch allocation for %d layers\n", n_layer); fflush(stderr);
        layer_scratch.resize(n_layer);
        for (int i = 0; i < n_layer; ++i) {
            fprintf(stderr, "[SCRATCH] Allocating layer %d/%d...\n", i+1, n_layer); fflush(stderr);
            if (!layer_scratch[i].alloc(E, HD, HQ, HKV, CTX, DQ, DKV, FF)) {
                fprintf(stderr, "[SCRATCH] Layer %d allocation failed\n", i); fflush(stderr);
                return false;
            }
            fprintf(stderr, "[SCRATCH] Layer %d OK\n", i+1); fflush(stderr);
        }
        fprintf(stderr, "[SCRATCH] All layers allocated successfully\n"); fflush(stderr);
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

    // Complete forward pass for one token
    int forward(int token_id, int pos, float* logits_out) {

        // --- Debug Step A (Phase 20): Print initial scratch[0] state ---
        fprintf(stderr, "[FORWARD] ENTRY: token=%d pos=%d SCRATCH[0][0..7]= ", token_id, pos); fflush(stderr);
        for (int i = 0; i < std::min(n_embd, 8); ++i) {
            float v = (float)layer_scratch[0].h[i];
            if (i > 0) fprintf(stderr, " ");
            if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
        }
        fprintf(stderr, "\n"); fflush(stderr);


        const int E = n_embd, HD = head_dim, HQ = n_head, HKV = n_head_kv, CTX = n_ctx;
        const int DQ = HQ * HD, DKV = HKV * HD, GRP = HQ / HKV;
        const float scale = 1.0f / sqrtf((float)HD);
        // Local copies for kernel args (avoid rvalue address issues)
        int one = 1;
        int E_local = E, DQ_local = DQ, DKV_local = DKV;
        int vocab_local = vocab_size;
        float eps_f = (float)eng_model->eps;

        // ---- 1) Embedding lookup (CPU-side, no GPU kernel needed for single token) ----
        fprintf(stderr, "[FORWARD] token_id=%d\n", token_id); fflush(stderr);

        int token = token_id;
        if (token < 0 || token >= vocab_local) token = 0;

        // Write embedding row directly into /dev/shm mapped layer_scratch[0].h
        const __half* emb_tensor = eng_model->embed_table.data();
        if (!emb_tensor) { fprintf(stderr, "[FORWARD] ERROR: no embedding tensor\n"); return -1; }

        size_t emb_bytes = (size_t)E_local * sizeof(__half);
        __half* dst_row = layer_scratch[0].h;
        const __half* src_row = emb_tensor + (size_t)(size_t)token * E_local;

        for (int i = 0; i < E_local; ++i) dst_row[i] = src_row[i];

        // Debug: verify embedding values right after copy
        fprintf(stderr, "[FORWARD] token_id=%d: emb[0..%zu]= ", token, (size_t)(E_local > 16 ? 15 : E_local - 1)); fflush(stderr);
        for (int i = 0; i < std::min(E_local, 16); ++i) {
            float v = (float)dst_row[i];
            if (i>0) fprintf(stderr,", ");
            if (v!=v) fprintf(stderr,"nan"); else if (v==0.0f && __half2float(dst_row[i])==0.0f) fprintf(stderr,"0.0000"); else fprintf(stderr,"%.4f",v);
        }
        fprintf(stderr, "\n"); fflush(stderr);



        fprintf(stderr, "[FORWARD] CPU lookup done: token=%d → residual@%p (%zu elements)\n",
                token, (void*)dst_row, E_local); fflush(stderr);

        // Residual starts as embedding output on /dev/shm mapped buffer.
        __half* residual = layer_scratch[0].h;

        // ---- 2) Decoder layers ----
        const bool has_ssm = eng_model->full_attention_interval > 0;
        for (int l = 0; l < n_layer; ++l) {
            // Phase 15.4: Skip SSM-only layers in hybrid architecture
            if (has_ssm && l % eng_model->full_attention_interval != 0) {
                continue;  // SSM layer — skip during inference (not yet implemented)
            }

            LayerScratch& s = layer_scratch[l];
            LayerWeights& L = eng_model->layers[l];

            // --- Debug Step Z (Phase 20): Trace skip condition for EVERY block ---
            {
                int fa_val = eng_model->full_attention_interval;
                int mod_result = l % fa_val;
                fprintf(stderr, "[FORWARD] blk %d: has_ssm=%d fa_int=%d l%%fa=%d -> skip=%d\n",
                        l, has_ssm, fa_val, mod_result, (has_ssm && mod_result != 0) ? 1 : 0); fflush(stderr);
            }

            // Force GPU sync before reading CPU memory to ensure all writes are visible
            (void)hipDeviceSynchronize();
            if (has_ssm && l % eng_model->full_attention_interval != 0) {
                fprintf(stderr, "[FORWARD] blk %d: SKIPPED (SSM-only, l%%interval=%d)\n", l, l % eng_model->full_attention_interval); fflush(stderr);
                continue;  // SSM layer — skip during inference (not yet implemented)
            }

            // --- Debug Step X (Phase 20): Print scratch[0][0..7] for ALL blocks ---
            fprintf(stderr, "[FORWARD] blk %d: PRE-OP SCRATCH[0][0..7]= ", l); fflush(stderr);
            for (int i = 0; i < std::min(E_local, 8); ++i) {
                float v = (float)layer_scratch[0].h[i];
                if (i > 0) fprintf(stderr, " ");
                if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
            }
            fprintf(stderr, "\n"); fflush(stderr);


            // --- Debug Step 5 (Phase 20): Check residual input at start of each attention block ---
            fprintf(stderr, "[FORWARD] blk %d: residual@%p first 8 elements after rmsnorm:\n", l, (void*)residual); fflush(stderr);
            {
                int chk = std::min(E_local, 8);
                fprintf(stderr, "[FORWARD] blk %d: h[0..%d]: ", l, chk-1); fflush(stderr);
                for (int i = 0; i < chk; ++i) {
                    float v = (float)s.h[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);

                 // Also check last few elements
                int start_last = std::max(0, E_local - 8);
                fprintf(stderr, "[FORWARD] blk %d: h[last 8]: ", l); fflush(stderr);
                for (int i = start_last; i < E_local; ++i) {
                    float v = (float)s.h[i];
                    if (i > start_last) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);
            }

            // Debug: check if attn_ln_scale exists
            fprintf(stderr, "[FORWARD] blk %d: attn_ln_size=%zu mlp_ln_size=%zu\n",
                    l, L.attn_ln_scale.size(), L.mlp_ln_scale.size()); fflush(stderr);

            // --- a) RMSNorm (attention) ---
            void* ln_args[] = { residual, L.d_attn_ln, s.d_h, &E_local, &E_local, &eps_f };
            hipError_t herr;
            fprintf(stderr, "[FORWARD] rmsnorm: res=%p w=%p out=%p\n", (void*)residual, (void*)L.d_attn_ln, (void*)s.d_h); fflush(stderr);

            // Qwen3.5-4B has no learnable layer norm weights — skip kernel launch with null gamma
            if (L.d_attn_ln == nullptr) {
                fprintf(stderr, "[FORWARD] attn_ln scale is NULL — doing inline RMSNorm\n"); fflush(stderr);

                // Inline RMSNorm on CPU: rsqrt(mean(x^2) + eps), write to scratch buffer
                float ssq = 0.0f;
                for (int i = 0; i < E_local; ++i) ssq += (float)residual[i] * (float)residual[i];
                const float inv_rms = 1.0f / sqrtf(ssq / (float)E_local + eng_model->eps);
                for (int i = 0; i < E_local; ++i) s.h[i] = (__half)((float)residual[i] * inv_rms);
                residual = s.h;  // Continue with normalized data in scratch

                // Debug block 0: check after RMSNorm
                if (l == 0) {
                    fprintf(stderr, "[FORWARD] POST-RMS blk 0: h[0..7]= "); fflush(stderr);
                    for (int i = 0; i < std::min(E_local, 8); ++i) {
                        float v = (float)s.h[i];
                        if (i > 0) fprintf(stderr, " ");
                        if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
                    }
                    fprintf(stderr, "\n"); fflush(stderr);
                    fprintf(stderr, "[FORWARD] POST-RMS blk 0: layer_scratch[0][0..7]= "); fflush(stderr);
                    for (int i = 0; i < std::min(E_local, 8); ++i) {
                        float v = (float)layer_scratch[0].h[i];
                        if (i > 0) fprintf(stderr, " ");
                        if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
                    }
                    fprintf(stderr, "\n"); fflush(stderr);
                }
            } else {
                herr = hipModuleLaunchKernel(fn_layernorm, E, 1, 1, 32, 1, 1, 0, 0, ln_args, nullptr);
                if (herr != hipSuccess) { fprintf(stderr, "[FORWARD] rmsnorm launch failed: %s\n", hipGetErrorString(herr)); return -1; }
                (void)hipDeviceSynchronize();
            }

            // --- b) Q/K/V Projections (gemm_act_fp4, M=1) ---
            {
                void* h_ptr = s.d_h;
                void* w_pack = L.Wq.d_packed;
                void* w_sc = L.Wq.d_scales;
                void* w_sf = L.Wq.d_scales_f;
                void* out_q = s.d_q;

                fprintf(stderr, "[FORWARD] gemm_Q: h=%p W_pack=%p W_sc=%p W_sf=%p out=%p\n",
                        h_ptr, w_pack, w_sc, w_sf, out_q); fflush(stderr);

                // Verify no NULL pointer values before launch
                if (!h_ptr || !w_pack || !w_sc || !out_q) {
                    fprintf(stderr, "[FORWARD] FATAL: NULL kernel arg detected!\n"); fflush(stderr);
                    return -1;
                }

                void* g4_args[] = { &h_ptr, &w_pack, &w_sc, &w_sf, &out_q, &one, &DQ_local, &E_local };
                fprintf(stderr, "[FORWARD] Launching gemm_Q with grid=(%d,%d,1) block=(32,1,1)\n", 1, (DQ+15)/16); fflush(stderr);
                herr = hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (DQ+15)/16, 1, 32, 1, 1, 0, 0, g4_args, nullptr);

            }

            // K projection
            {
                void* h_ptr = s.d_h;
                void* w_pack = L.Wk.d_packed;
                void* w_sc = L.Wk.d_scales;
                void* w_sf = L.Wk.d_scales_f;
                void* out_k = s.d_k;

                fprintf(stderr, "[FORWARD] gemm_K: h=%p W_pack=%p k=%p\n",
                        h_ptr, w_pack, out_k); fflush(stderr);

                if (!h_ptr || !w_pack || !w_sc || !out_k) {
                    fprintf(stderr, "[FORWARD] FATAL: NULL kernel arg in gemm_K!\n"); fflush(stderr);
                    return -1;
                }

                void* g4k_args[] = { &h_ptr, &w_pack, &w_sc, &w_sf, &out_k, &one, &DKV_local, &E_local };
                herr = hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (DKV+15)/16, 1, 32, 1, 1, 0, 0, g4k_args, nullptr);
                if (herr != hipSuccess) { fprintf(stderr, "[FORWARD] gemm_K launch failed: %s\n", hipGetErrorString(herr)); return -1; }
            }

            // V projection
            {
                void* h_ptr = s.d_h;
                void* w_pack = L.Wv.d_packed;
                void* w_sc = L.Wv.d_scales;
                void* w_sf = L.Wv.d_scales_f;
                void* out_v = s.d_v;

                fprintf(stderr, "[FORWARD] gemm_V: h=%p W_pack=%p v=%p\n",
                        h_ptr, w_pack, out_v); fflush(stderr);

                void* g4v_args[] = { &h_ptr, &w_pack, &w_sc, &w_sf, &out_v, &one, &DKV_local, &E_local };
                herr = hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (DKV+15)/16, 1, 32, 1, 1, 0, 0, g4v_args, nullptr);
            }
            if (herr != hipSuccess) { fprintf(stderr, "[FORWARD] gemm_V launch failed: %s\n", hipGetErrorString(herr)); return -1; }
            (void)hipDeviceSynchronize();

            // --- Debug Step 1 (Phase 20): Read back K/V from CPU to verify V projection output ---
            fprintf(stderr, "[FORWARD] blk %d: V-proj sync done — reading back k_fp32/v_fp32\n", l); fflush(stderr);
            {
                int kv_debug_count = std::min(HKV * HD, 32);
                fprintf(stderr, "[FORWARD] blk %d: k_fp32[0..%d]: ", l, kv_debug_count-1); fflush(stderr);
                for (int i = 0; i < kv_debug_count && i < HKV*HD; ++i) {
                    float v = s.k_fp32[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);

                int v_debug_count = DKV;
                fprintf(stderr, "[FORWARD] blk %d: v_fp32[0..%d]: ", l, v_debug_count-1); fflush(stderr);
                for (int i = 0; i < v_debug_count && i < DKV; ++i) {
                    float v = s.v_fp32[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);

                // Also print last few values to check full range validity
                fprintf(stderr, "[FORWARD] blk %d: k_fp32[last 8]: ", l); fflush(stderr);
                {
                    int start_k = std::max(0, HKV*HD - 8);
                    for (int i = start_k; i < HKV*HD; ++i) {
                        float v = s.k_fp32[i];
                        if (i > start_k) fprintf(stderr, " ");
                        if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                    }
                    fprintf(stderr, "\n"); fflush(stderr);

                    fprintf(stderr, "[FORWARD] blk %d: v_fp32[last 8]: ", l); fflush(stderr);
                    {
                        int start_v = std::max(0, DKV - 8);
                        for (int i = start_v; i < DKV; ++i) {
                            float v = s.v_fp32[i];
                            if (i > start_v) fprintf(stderr, " ");
                            if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                        }
                        fprintf(stderr, "\n"); fflush(stderr);
                    }
                }
            }



            // --- Debug Step 2 (Phase 20): Check k_fp32 BEFORE FP16 conversion ---
            fprintf(stderr, "[FORWARD] blk %d: PRE-STAGING k_fp32[0..15]: ", l); fflush(stderr);
            for (int i = 0; i < std::min(HKV*HD, 16); ++i) {
                float v = s.k_fp32[i];
                if (i > 0) fprintf(stderr, " ");
                if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
            }
            fprintf(stderr, "\n"); fflush(stderr);

            // --- Debug Step 3 (Phase 20): Check k_fp32 AFTER RoPE on K ---
            // RoPE is applied in-place to qh and kh buffers after this point.
            // --- c) Per-head staging (FP32 -> FP16) ---
            fprintf(stderr, "[FORWARD] blk %d: per-head staging (HQ=%d HD=%d)\n", l, HQ, HD); fflush(stderr);

            // Debug: print first few q_fp32 values before conversion
            {
                int debug_count = std::min(HQ * 8, 16); // print first 8*HQ elements (or up to 16 total)
                fprintf(stderr, "[FORWARD] q_fp32[0..%d]: ", debug_count-1); fflush(stderr);
                for (int i = 0; i < debug_count && i < HQ*HD; ++i) {
                    float v = s.q_fp32[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v); // NaN check
                }
                fprintf(stderr, "\n"); fflush(stderr);

                int k_debug_count = std::min(HKV * 8, 16);
                fprintf(stderr, "[FORWARD] k_fp32[0..%d]: ", k_debug_count-1); fflush(stderr);
                for (int i = 0; i < k_debug_count && i < HKV*HD; ++i) {
                    float v = s.k_fp32[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);
            }
            for (int h = 0; h < HQ; ++h)
                for (int d = 0; d < HD; ++d)
                    s.qh[h*HD + d] = __float2half(s.q_fp32[h*HD + d]);
            for (int h = 0; h < HKV; ++h)
                for (int d = 0; d < HD; ++d)
                    s.kh[h*HD + d] = __float2half(s.k_fp32[h*HD + d]);
            s.d_qh = s.qh;
            s.d_kh = s.kh;

            // --- d) RoPE on Q/K ---
            build_rope_tables(pos, s);
            int HQ_local = HQ, HD2_local = HD/2;
            void* rope_args[] = { &s.d_qh, &s.d_rc, &s.d_rs, &s.d_qh, &HQ_local, &HD2_local };
            fprintf(stderr, "[FORWARD] blk %d: RoPE on Q [%d heads] K [%d heads]\n", l, HQ, HKV); fflush(stderr);
            (void)hipModuleLaunchKernel(fn_rope, HQ, 1, 1, 32, 1, 1, 0, 0, rope_args, nullptr);
            (void)hipModuleLaunchKernel(fn_rope, HKV, 1, 1, 32, 1, 1, 0, 0, rope_args, nullptr);
            (void)hipDeviceSynchronize();

            // Debug: check q_h values after RoPE
            if (l == 0) {
                fprintf(stderr, "[FORWARD] blk %d: q_h[0..7] AFTER ROPE: ", l); fflush(stderr);
                for (int i = 0; i < 8 && i < HD; ++i) {
                    float v = (float)((__half*)s.qh)[i];
                    if (v != v) fprintf(stderr,"nan "); else fprintf(stderr,"%.4f ",v);
                }
                fprintf(stderr, "\n"); fflush(stderr);

                fprintf(stderr, "[FORWARD] blk %d: k_h[0..7]: ", l); fflush(stderr);
                for (int i = 0; i < 8 && i < HD; ++i) {
                    float v = (float)((__half*)s.kh)[i];
                    if (v != v) fprintf(stderr,"nan "); else fprintf(stderr,"%.4f ",v);
                }
                fprintf(stderr, "\n"); fflush(stderr);
            }

            // --- e) KV Store ---
            // Flatten roped K from per-head to token-major rks
            for (int h = 0; h < HKV; ++h)
                for (int d = 0; d < HD; ++d)
                    s.rks[h*HD + d] = s.kh[h*HD + d];
            s.d_rks = s.rks;

            // Ensure V projection fully completes before KV store reads its output
            (void)hipDeviceSynchronize();


            __half* kdst_base = kv_cache_k_row(eng_model->kvc, l, 0);
            __half* vdst_base = kv_cache_v_row(eng_model->kvc, l, 0);
            fprintf(stderr, "[FORWARD] kv_store PRE: rks=%p v=%p kdst=%p vdst=%p DKV=%d pos=%d CTX=%d\n",
                    (void*)s.d_rks, (void*)s.d_v, (void*)kdst_base, (void*)vdst_base,
                    DKV_local, pos, CTX); fflush(stderr);

            // Check for NULL pointers before launch
            if (!s.d_rks || !s.d_v || !kdst_base || !vdst_base) {
                fprintf(stderr, "[FORWARD] FATAL: NULL pointer in kv_store! rks=%p v=%p kdst=%p vdst=%p\n",
                        (void*)s.d_rks, (void*)s.d_v, (void*)kdst_base, (void*)vdst_base); fflush(stderr);
                return -1;
            }
            int pos_local = pos, CTX_local2 = CTX;
            void* kvs_args[] = { &s.d_rks, &s.d_v, &kdst_base, &vdst_base,
                                &one, &DKV_local, &pos_local, &CTX_local2 };
            hipError_t herr_kv = hipModuleLaunchKernel(fn_kv_store, 8, 1, 1, 32, 1, 1, 0, 0, kvs_args, nullptr);
            if (herr_kv != hipSuccess) { fprintf(stderr, "[FORWARD] kv_store launch failed: %s\n", hipGetErrorString(herr_kv)); return -1; }
            (void)hipDeviceSynchronize();

            // --- f) Attention per head ---
            {
                const __half* kc = kv_cache_k_row_const(eng_model->kvc, l, 0);
                const __half* vc = kv_cache_v_row_const(eng_model->kvc, l, 0);

                // --- Debug Step 4 (Phase 20): Check KV cache read-back BEFORE attention scoring ---
                fprintf(stderr, "[FORWARD] blk %d: KV store done — checking kc/vc at %p/%p\n", l, (void*)kc, (void*)vc); fflush(stderr);
                {
                    int kv_check_count = std::min(HKV * HD, 16);
                    fprintf(stderr, "[FORWARD] blk %d: KV cache k[0..%d]: ", l, kv_check_count-1); fflush(stderr);
                    for (int i = 0; i < kv_check_count && i < DKV; ++i) {
                        float v = (float)((const __half*)kc)[i];
                        if (i > 0) fprintf(stderr, " ");
                        if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                    }
                    fprintf(stderr, "\n"); fflush(stderr);

                     fprintf(stderr, "[FORWARD] blk %d: KV cache v[0..%d]: ", l, std::min(DKV, 16)-1); fflush(stderr);
                    for (int i = 0; i < std::min(DKV, 16); ++i) {
                        float v = (float)((const __half*)vc)[i];
                        if (i > 0) fprintf(stderr, " ");
                        if (v != v) fprintf(stderr, "nan"); else fprintf(stderr, "%.4f", v);
                    }
                    fprintf(stderr, "\n"); fflush(stderr);
                }                int CTX_local3 = CTX, HD_local = HD, qpos_local = pos, pos1_local = pos + 1;

                for (int h = 0; h < HQ; ++h) {
                    int kh_ = h / GRP;

                    // Build kexpB [HD x CTX] and vexpB [CTX x HD]
                    for (int d = 0; d < HD; ++d)
                        for (int p = 0; p <= pos; ++p)
                            s.kh[d*CTX + p] = kc[p*DKV + kh_*HD + d];
                    for (int p = 0; p <= pos; ++p)
                        for (int d = 0; d < HD; ++d)
                            s.vexpB[p*HD + d] = vc[p*DKV + kh_*HD + d];

                    // --- Scores: [1 x CTX] = qh[h] @ kexpB^T (CPU-side to avoid fp16_gemm gfx1151 bug) ---
                    const __half* q_h = (__half*)((char*)s.d_qh + h*HD*sizeof(__half));
                    const __half* k_expB = (const __half*)(s.d_vb2);
                    float* scores_out = (float*)(s.d_sc);

                    fprintf(stderr, "[FORWARD] attn h=%d: CPU gemm_scores [%d x %d] -> [%d]\n",
                            h, 1, HD, CTX); fflush(stderr);

                    {
                        // scores[p] = sum_d(q[d] * kexpB[d][p]), d in [0..HD), p in [0..pos]
                        // Note: scale is applied by softmax_masked kernel, not here

                        // Debug: print first few query/key values for head 0
                        if (h == 0 && l == 0) {
                            fprintf(stderr, "[FORWARD] attn h=0 blk 0: q_h[0..7]="); fflush(stderr);
                            for (int i = 0; i < std::min(HD,8); ++i) {
                                float v = (float)q_h[i];
                                if (i>0) fprintf(stderr,", ");
                                if (v!=v) fprintf(stderr,"nan"); else fprintf(stderr,"%.4f",v);
                            }
                            fprintf(stderr, "\n"); fflush(stderr);

                            fprintf(stderr, "[FORWARD] attn h=0 blk 0: k_expB_col0[0..7]="); fflush(stderr);
                            for (int i = 0; i < std::min(HD,8); ++i) {
                                float v = (float)k_expB[(size_t)i * CTX + 0];
                                if (i>0) fprintf(stderr,", ");
                                if (v!=v) fprintf(stderr,"nan"); else fprintf(stderr,"%.4f",v);
                            }
                            fprintf(stderr, "\n"); fflush(stderr);

                            // Print first score value
                            float acc_debug = 0.0f;
                            for (int d = 0; d < HD && d < 16; ++d) {
                                acc_debug += (float)q_h[d] * (float)k_expB[(size_t)d * CTX + 0];
                            }
                            fprintf(stderr, "[FORWARD] attn h=0 blk 0: partial dot(q,k)[0..15] = %.4f\n", acc_debug); fflush(stderr);

                            // Print actual scores array
                            for (int p = 0; p <= pos && p < 4; ++p) {
                                fprintf(stderr, "[FORWARD] attn h=0 blk 0: score[%d] = %.4f\n", p, scores_out[p]); fflush(stderr);
                            }
                        }
                        for (int p = 0; p <= pos; ++p) {
                            float acc = 0.0f;
                            for (int d = 0; d < HD; ++d) {
                                acc += (float)q_h[d] * (float)k_expB[(size_t)d * CTX + p];
                            }
                            scores_out[p] = acc;
                        }
                        // Zero out remaining positions beyond pos (causal mask)
                        for (int p = pos + 1; p < CTX; ++p) {
                            scores_out[p] = -INFINITY;
                        }
                    }

                    // --- Softmax (causal) — CPU-side, properly handles masking ---
                    {
                        const int n_rows = 1;
                        const int n_cols = CTX;
                        const int qmax = pos;  // causal ceiling (no qpos array needed)

                        // Find max over VALID columns only (0..qmax), skip masked positions
                        float m = -INFINITY;
                        for (int c = 0; c <= qmax && c < n_cols; ++c) {
                            if (scores_out[c] > m) m = scores_out[c];
                        }

                        fprintf(stderr, "[FORWARD] attn h=%d: softmax max=%.6f\n", h, m == -INFINITY ? 0.0f : m); fflush(stderr);

                        // Handle degenerate case where all values are -INF
                        if (m == -INFINITY) {
                            for (int c = 0; c < n_cols; ++c) scores_out[c] = 0.0f;
                        } else {
                            // Subtract max and exponentiate ONLY for valid columns
                            float e_sum = 0.0f;
                            for (int c = 0; c <= qmax && c < n_cols; ++c) {
                                scores_out[c] = expf(scores_out[c] - m);
                                e_sum += scores_out[c];
                            }
                            // Masked positions beyond qmax stay at 0.0f (initialized from kernel output)

                            // Normalize by sum ONLY over valid columns
                            const float inv_sum = 1.0f / e_sum;
                            for (int c = 0; c <= qmax && c < n_cols; ++c) {
                                scores_out[c] *= inv_sum;
                            }

                            fprintf(stderr, "[FORWARD] attn h=%d: softmax done [0..%d], sum=%.6f\n",
                                    h, qmax, e_sum); fflush(stderr);
                        }
                    }

                    // Probs to FP16
                    for (int p = 0; p <= pos; ++p)
                        s.probs_fp16[p] = __float2half(s.scores[p]);

                    // --- Out: [HD] = probs @ vexpB (CPU-side to avoid fp16_gemm gfx1151 bug) ---
                    const __half* prob_h = (const __half*)(s.d_pf);
                    const __half* v_expB = (const __half*)(s.d_vb2);
                    float* out_h = (float*)((char*)s.d_oh + h*HD*sizeof(float));

                    fprintf(stderr, "[FORWARD] attn h=%d: CPU gemm_out [%d x %d] -> [%d]\n",
                            h, CTX, HD, HD); fflush(stderr);

                    {
                        // out[d] = sum_p(prob[p] * vexpB[p][d]), d in [0..HD), p in [0..pos]
                        for (int d = 0; d < HD; ++d) {
                            float acc = 0.0f;
                            for (int p = 0; p <= pos; ++p) {
                                acc += (float)prob_h[p] * (float)v_expB[(size_t)p * HD + d];
                            }
                            out_h[d] = acc;
                        }
                    }

                }
            }
            (void)hipDeviceSynchronize();

            // --- g) Flatten heads -> oin (FP16) ---
            for (int h = 0; h < HQ; ++h)
                for (int d = 0; d < HD; ++d)
                    s.oin[h*HD + d] = __float2half(s.oh[h*HD + d]);
            s.d_oin = s.oin;

            // --- h) O-Projection (skip if SSM-only layer with no attention output weights) ---
            if (L.Wo.d_packed != nullptr) {
                void* o_args[] = { &s.d_oin, &L.Wo.d_packed, &L.Wo.d_scales, &L.Wo.d_scales_f, &s.d_down,
                                  &one, &E_local, &DQ_local };
                fprintf(stderr, "[FORWARD] blk %d: O-proj via gemm_act_fp4 [%d x %d]\n", l, E, DQ_local); fflush(stderr);
                (void)hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (E+15)/16, 1, 32, 1, 1, 0, 0, o_args, nullptr);
                (void)hipDeviceSynchronize();
            } else {
                fprintf(stderr, "[FORWARD] blk %d: no attn_o weights — skipping O-projection\n", l); fflush(stderr);
                // Zero out down_fp32 so residual add doesn't produce NaN from uninitialized values
                for (int i = 0; i < E; ++i) s.down_fp32[i] = 0.0f;
            }

            // --- i) Residual add (attention output) ---
            for (int i = 0; i < E; ++i)
                residual[i] = __float2half(__half2float(residual[i]) + s.down_fp32[i]);

            // --- j) RMSNorm (MLP) ---
            void* ln2_args[] = { &residual, &L.d_mlp_ln, &s.d_hmlp,
                                &E_local, &E_local, (void*)&eng_model->eps };
            fprintf(stderr, "[FORWARD] rmsnorm_mlp: w=%p attn_ln_scale.size()=%zu mlp_ln_scale.size()=%zu\n",
                    (void*)L.d_mlp_ln, L.attn_ln_scale.size(), L.mlp_ln_scale.size()); fflush(stderr);

            // Debug: Check h_mlp activation data before SwiGLU
            if (l == 0 && L.d_mlp_ln == nullptr) {
                fprintf(stderr, "[FORWARD] RMSNorm blk 0: writing %d elements to h_mlp[0..7]= ", E_local); fflush(stderr);
                for (int i = 0; i < std::min(E_local, 8); ++i) {
                    float v = (float)s.h_mlp[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);
                fprintf(stderr, "[FORWARD] RMSNorm blk 0: d_hmlp pointer value = %p\n", (void*)s.d_hmlp); fflush(stderr);
            }

            // Skip MLP layer norm if no learnable weights (Qwen3.5-4B has none) — inline RMSNorm
            if (L.d_mlp_ln == nullptr) {
                fprintf(stderr, "[FORWARD] mlp_ln scale is NULL — doing inline RMSNorm\n"); fflush(stderr);
                float ssq = 0.0f;
                for (int i = 0; i < E_local; ++i) ssq += (float)residual[i] * (float)residual[i];
                const float inv_rms = 1.0f / sqrtf(ssq / (float)E_local + eng_model->eps);
                for (int i = 0; i < E_local; ++i) s.h_mlp[i] = (__half)((float)residual[i] * inv_rms);
            } else {
                (void)hipModuleLaunchKernel(fn_layernorm, E, 1, 1, 32, 1, 1, 0, 0, ln2_args, nullptr);
                (void)hipDeviceSynchronize();
            }

            // --- k) FFN: down( SiLU(gate(h)) * up(h) ) ---
            // Three gemm_act_fp4 launches (verified by actfp4_verify) rather than the fused swiglu_fp4:
            // gate/up/down can each be a different quant type (e.g. blk.0: Q4_0 / TYPE100 / Q4_1), and
            // gemm_act_fp4 dispatches per tensor on d_scales_f (float path) vs d_scales (UE4M3 path).
            // Every args[] entry is the ADDRESS of the argument value, as hipModuleLaunchKernel requires.
            const bool ffn_ok = L.W_gate.d_packed && L.W_up.d_packed && L.W_down.d_packed &&
                                L.W_gate.K == E_local && L.W_up.K == E_local &&
                                L.W_gate.N == L.W_up.N && L.W_gate.N <= s.ff &&
                                L.W_down.K == L.W_gate.N && L.W_down.N == E_local;
            if (ffn_ok) {
                int FF_l = L.W_gate.N;
                void* g_args[] = { &s.d_hmlp, &L.W_gate.d_packed, &L.W_gate.d_scales, &L.W_gate.d_scales_f,
                                   &s.d_gate, &one, &FF_l, &E_local };
                (void)hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (FF_l+15)/16, 1, 32, 1, 1, 0, 0, g_args, nullptr);
                void* u_args[] = { &s.d_hmlp, &L.W_up.d_packed, &L.W_up.d_scales, &L.W_up.d_scales_f,
                                   &s.d_up, &one, &FF_l, &E_local };
                (void)hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (FF_l+15)/16, 1, 32, 1, 1, 0, 0, u_args, nullptr);
                hipError_t gu_err = hipDeviceSynchronize();

                for (int i = 0; i < FF_l; ++i) {
                    const float g = s.gate_fp32[i];
                    s.act[i] = __float2half(g / (1.0f + expf(-g)) * s.up_fp32[i]);
                }

                void* d_args[] = { &s.d_act, &L.W_down.d_packed, &L.W_down.d_scales, &L.W_down.d_scales_f,
                                   &s.d_down, &one, &E_local, &FF_l };
                (void)hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (E+15)/16, 1, 32, 1, 1, 0, 0, d_args, nullptr);
                hipError_t dn_err = hipDeviceSynchronize();
                fprintf(stderr, "[FORWARD] blk %d: FFN [%d -> %d -> %d] types gate=%d up=%d down=%d err=%d/%d\n",
                        l, E_local, FF_l, E_local, L.W_gate.quant_type, L.W_up.quant_type, L.W_down.quant_type,
                        (int)gu_err, (int)dn_err); fflush(stderr);
            } else {
                fprintf(stderr, "[FORWARD] blk %d: FFN weights missing or shape mismatch "
                        "(gate %dx%d up %dx%d down %dx%d, ff=%d) — FFN skipped\n", l,
                        L.W_gate.N, L.W_gate.K, L.W_up.N, L.W_up.K, L.W_down.N, L.W_down.K, s.ff); fflush(stderr);
                for (int i = 0; i < E; ++i) s.down_fp32[i] = 0.0f;
            }

            // --- Debug Step N (Phase 20): Check SwiGLU output for block 0 ---
            if (l == 0) {
                fprintf(stderr, "[FORWARD] POST-SWiGLU blk 0: down_fp32[0..7]= "); fflush(stderr);
                for (int i = 0; i < std::min(E_local, 8); ++i) {
                    float v = s.down_fp32[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);

                // Also check layer_scratch[0] right after SwiGLU completes
                fprintf(stderr, "[FORWARD] POST-SWiGLU blk 0: layer_scratch[0][0..7]= "); fflush(stderr);
                for (int i = 0; i < std::min(E_local, 8); ++i) {
                    float v = (float)layer_scratch[0].h[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);
            }
            // --- Debug Step 7 (Phase 20): Check global scratch state before MLP residual add ---
            if (l == 0) {
                fprintf(stderr, "[FORWARD] PRE-MLP residual-add: scratch[0][0..7]= "); fflush(stderr);
                for (int i = 0; i < std::min(E_local, 8); ++i) {
                    float v = (float)layer_scratch[0].h[i];
                    if (i > 0) fprintf(stderr, " ");
                    if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
                }
                fprintf(stderr, "\n"); fflush(stderr);
            }

             // --- l) Residual add (MLP output) ---

            {
                int chk = std::min(E_local, 8);
                fprintf(stderr, "[FORWARD] blk %d: residual[0..%d] before add: ", l, chk-1); fflush(stderr);
                for (int i = 0; i < chk; ++i) { float v = (float)residual[i]; if (v!=v) fprintf(stderr,"nan "); else fprintf(stderr,"%.2f ",v); }
                fprintf(stderr, "\n"); fflush(stderr);
            }
            for (int i = 0; i < E; ++i)
                residual[i] = __float2half(__half2float(residual[i]) + s.down_fp32[i]);

            // --- Debug Step X-Y (Phase 20): Force sync after each block's completion ---
            {
                static int block_done_count = 0;
                if (l == 0) {
                    fprintf(stderr, "[FORWARD] END-BLK-0: forcing hipDeviceSynchronize()..."); fflush(stderr);
                    hipError_t herr_sync = hipDeviceSynchronize();
                    fprintf(stderr, " done (err=%d)\n", (int)herr_sync); fflush(stderr);

                    // Check scratch[0] immediately after sync
                    fprintf(stderr, "[FORWARD] SYNC-END blk0: layer_scratch[0][0..7]= "); fflush(stderr);
                    for (int i = 0; i < std::min(E_local, 8); ++i) {
                        float v = (float)layer_scratch[0].h[i];
                        if (i > 0) fprintf(stderr, " ");
                        if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
                    }
                    fprintf(stderr, "\n"); fflush(stderr);
                }
            }

        }

        // --- Debug Step M (Phase 20): Check scratch[0] AFTER all decoder layers complete ---
        fprintf(stderr, "[FORWARD] END-FORWARD-PASS: scratch[0][0..7]= "); fflush(stderr);
        for (int i = 0; i < std::min(E_local, 8); ++i) {
            float v = (float)layer_scratch[0].h[i];
            if (i > 0) fprintf(stderr, " ");
            if (v != v) fprintf(stderr, "nan"); else if (__builtin_isinf(v)) fprintf(stderr, "inf"); else fprintf(stderr, "%.4f", v);
        }
        fprintf(stderr, "\n"); fflush(stderr);


        // ---- 3) Final LayerNorm — inline CPU when no learnable weights (Qwen3.5 uses fixed eps) ---
        if (eng_model->d_final_ln != nullptr) {
            void* lnf_args[] = { &residual, eng_model->d_final_ln, &layer_scratch[0].d_h,
                                &E_local, &E_local, (void*)&eng_model->eps };
            fprintf(stderr, "[FORWARD] Final LN via GPU kernel\n"); fflush(stderr);
            (void)hipModuleLaunchKernel(fn_layernorm, E, 1, 1, 32, 1, 1, 0, 0, lnf_args, nullptr);
            (void)hipDeviceSynchronize();
        } else {
            fprintf(stderr, "[FORWARD] Final LN: no learnable weights — doing inline RMSNorm with eps=%.10f\n",
                    eng_model->eps); fflush(stderr);
            float ssq = 0.0f;
            for (int i = 0; i < E_local; ++i) ssq += (float)residual[i] * (float)residual[i];
            const float inv_rms = 1.0f / sqrtf(ssq / (float)E_local + eng_model->eps);
            for (int i = 0; i < E_local; ++i) layer_scratch[0].h[i] = (__half)((float)residual[i] * inv_rms);
        }

        // ---- 4) LM Head (gemm_act_fp4, M=1, N=vocab_size) ---
        // For tied embeddings (Qwen3.5-4B), d_lm_head_packed is NULL — compute logits via CPU dot products with embed table
        if (!eng_model->d_lm_head_packed) {
            fprintf(stderr, "[FORWARD] LM head: tied embeddings — CPU dot with embed table\n"); fflush(stderr);

            const __half* h = layer_scratch[0].h;  // final hidden state [n_embd]
            const __half* emb_table = eng_model->embed_table.data();  // [vocab x n_embd], dequantized FP16

            for (int v = 0; v < vocab_local; ++v) {
                float logit = 0.0f;
                const __half* emb_row = emb_table + (size_t)v * E_local;
                for (int i = 0; i < E_local; ++i) {
                    logit += (float)h[i] * (float)emb_row[i];
                }
                logits_out[v] = logit;
            }

            fprintf(stderr, "[FORWARD] LM head done: vocab=%d\n", vocab_local); fflush(stderr);
        } else {
            void* lm_args[] = { &layer_scratch[0].d_h, &eng_model->d_lm_head_packed,
                               &eng_model->d_lm_head_scales, &eng_model->d_lm_head_scales_f, logits_out,
                               &one, &vocab_local, &E_local };
            fprintf(stderr, "[FORWARD] LM head via GPU gemm_act_fp4 [%d x %d]\n", E_local, vocab_local); fflush(stderr);
            (void)hipModuleLaunchKernel(fn_gemm_act_fp4, 1, (vocab_size+15)/16, 1, 32, 1, 1, 0, 0, lm_args, nullptr);
            (void)hipDeviceSynchronize();
        }

        return 0;
    }
};

extern "C" {
qwen_model_t* qwen_model_create(inference_model_t* engine_model) {
    QwenModel* m = new QwenModel();
    if (!m->init(engine_model)) { delete m; return nullptr; }
    return m;
}

void qwen_model_set_embed_table(qwen_model_t* model, void* d_embed_table) {
    QwenModel* m = static_cast<QwenModel*>(model);
    fprintf(stderr, "[SET_EMBED] Setting d_embed_table_copy from %p to %p\n",
            (void*)(m ? m->d_embed_table_copy : nullptr), d_embed_table);
    if (m) m->d_embed_table_copy = reinterpret_cast<const __half*>(d_embed_table);
}

void qwen_model_free(qwen_model_t* model) {
    delete model;
}

int qwen_model_forward(qwen_model_t* model, int token_id, int pos, float* logits) {
    return model->forward(token_id, pos, logits);
}

// Batched forward pass: processes all sequences through decoder layers using batched kernels.
// Replaces the sequential stub that called forward() in a loop for each token.
int qwen_model_forward_batch(qwen_model_t* model,
                             const int32_t* tokens,
                             const int* positions,
                             const int* seq_lens,
                             int batch_size,
                             int seq_len,
                             float* logits_out) {
    (void)seq_len;

    QwenModel* m = model;
    inference_model* eng = model->eng_model;
    const int E = m->n_embd, HD = m->head_dim, HQ = m->n_head, HKV = m->n_head_kv;
    const int DQ = HQ * HD, DKV = HKV * HD;
    const float eps_f = (float)eng->eps;

    // Allocate temporary GPU buffers for the batch - same pattern as inference_forward_batch
    size_t scores_bytes = (size_t)batch_size * seq_len * seq_len * sizeof(float);
    size_t qpos_bytes = (size_t)batch_size * seq_len * sizeof(int);
    size_t logits_bytes = (size_t)batch_size * m->vocab_size * sizeof(float);

    float* d_scores_batch = nullptr;
    int* d_qpos_batch = nullptr;
    float* d_logits_batch = nullptr;

    hipError_t herr = hipMalloc(&d_scores_batch, scores_bytes);
    if (herr != hipSuccess) { qwen_set_error("hipMalloc scores batch failed"); return INFERENCE_ERR_KERNEL_LAUNCH; }
    herr = hipMalloc(&d_qpos_batch, qpos_bytes);
    if (herr != hipSuccess) { (void)hipFree(d_scores_batch); qwen_set_error("hipMalloc qpos batch failed"); return INFERENCE_ERR_KERNEL_LAUNCH; }
    herr = hipMalloc(&d_logits_batch, logits_bytes);
    if (herr != hipSuccess) { (void)hipFree(d_scores_batch); (void)hipFree(d_qpos_batch); qwen_set_error("hipMalloc logits batch failed"); return INFERENCE_ERR_KERNEL_LAUNCH; }

    // Prepare per-sequence qpos on host: for each batch, qpos[seq] starts at positions[b]
    std::vector<int> h_qpos((size_t)batch_size * seq_len);
    for (int b = 0; b < batch_size; ++b) {
        int base_pos = positions[b];
        int len = seq_lens ? seq_lens[b] : seq_len;
        for (int r = 0; r < seq_len; ++r) {
            int val = (r < len) ? (base_pos + r) : (base_pos + len - 1);
            h_qpos[(size_t)b * seq_len + r] = val;
        }
    }
    herr = hipMemcpy(d_qpos_batch, h_qpos.data(), qpos_bytes, hipMemcpyHostToDevice);
    if (herr != hipSuccess) {
        (void)hipFree(d_scores_batch); (void)hipFree(d_qpos_batch); (void)hipFree(d_logits_batch);
        qwen_set_error("hipMemcpy qpos batch failed"); return INFERENCE_ERR_KERNEL_LAUNCH;
    }

    // ---- Embedding lookup (per sequence via grid-stride launch) ----
    int vocab_local = eng->vocab_size;
    int one_int = 1;

    // Build per-sequence embedding index arrays (each sequence has different token ID)
    std::vector<int32_t> h_tokens(batch_size);
    for (int b = 0; b < batch_size; ++b) h_tokens[b] = tokens[b];

    const __half* d_emb_ptr_batch = reinterpret_cast<const __half*>(eng->d_embed_table);
    void* emb_args[] = { &h_tokens[0], (const __half**)&d_emb_ptr_batch, m->layer_scratch[0].d_h,
                         (void*)&one_int, (void*)&E, (void*)&vocab_local };
    // Embedding lookup is inherently per-token (different indices), so single-launch with grid-stride over batch
    (void)hipModuleLaunchKernel(fn_embedding, 1, 1, batch_size, 256, 1, 1, 0, 0, emb_args, nullptr);
    if (hipGetLastError() != hipSuccess) {
        (void)hipFree(d_scores_batch); (void)hipFree(d_qpos_batch); (void)hipFree(d_logits_batch);
        qwen_set_error("embedding lookup failed"); return INFERENCE_ERR_KERNEL_LAUNCH;
    }

    // ---- Decoder layers: loop over layers once, launch batched kernels per layer ----
    int E_local = E;
    int DQ_local = DQ;
    int DKV_local = DKV;
    int CTX_local = seq_len;

    for (int l = 0; l < m->n_layer; ++l) {
        LayerWeights& L = eng->layers[l];

        // --- a) RMSNorm (attention) - batched with grid-stride ---
        // Each sequence processes independently -> batch dimension on Z axis
        void* ln_args[] = {
            &m->layer_scratch[0].d_h,                    // residual base pointer (offset handled in kernel)
            &L.d_attn_ln,                               // LN scale array (per layer)
            &m->layer_scratch[0].d_h,                   // output buffer base (same as input)
            &E_local, &E_local, (void*)&eps_f
        };
        dim3 ln_grid((E + 31) / 32, 1, batch_size);
        (void)hipModuleLaunchKernel(fn_layernorm, ln_grid.x, ln_grid.y, ln_grid.z, 32, 1, 1, 0, 0, ln_args, nullptr);
        if (hipGetLastError() != hipSuccess) {
            qwen_set_error("RMSNorm attn failed"); goto cleanup;
        }

        // --- b) Q/K/V Projections (gemm_act_fp4) - batched ---
        // For each sequence: [M=1 x N=DQ] = h[1 x E] @ Wq^T
        // Batch dimension Z: process all sequences simultaneously
        {
            std::vector<void*> d_input_arr(batch_size), d_output_arr(batch_size);
            void* w_packed = L.Wq.d_packed;
            void* w_scales = L.Wq.d_scales;
            void* w_scales_f = L.Wq.d_scales_f;

            for (int b = 0; b < batch_size; ++b) {
                d_input_arr[b] = m->layer_scratch[0].d_h;   // offset handled via blockIdx.z
                d_output_arr[b] = m->layer_scratch[l].d_q;  // per-layer scratch buffer
            }

            int DQ_b = DQ_local;
            void* g4_args[] = { &d_input_arr[0], &w_packed, &w_scales, &w_scales_f, &d_output_arr[0],
                               &one_int, &DQ_b, &E_local };
            dim3 gemm_grid(1, (DQ + 15) / 16, batch_size);
            (void)hipModuleLaunchKernel(fn_gemm_act_fp4, gemm_grid.x, gemm_grid.y, gemm_grid.z,
                                 32, 16, 1, 0, 0, g4_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("gemm Q failed"); goto cleanup;
            }

            // K projection: [1 x DKV] = h @ Wk^T
            w_packed = L.Wk.d_packed; w_scales = L.Wk.d_scales; w_scales_f = L.Wk.d_scales_f;
            int DKV_b = DKV_local;
            for (int b = 0; b < batch_size; ++b) d_output_arr[b] = m->layer_scratch[l].d_k;
            void* k4_args[] = { &d_input_arr[0], &w_packed, &w_scales, &w_scales_f, &d_output_arr[0],
                                &one_int, &DKV_b, &E_local };
            dim3 k_grid(1, (DKV + 15) / 16, batch_size);
            (void)hipModuleLaunchKernel(fn_gemm_act_fp4, k_grid.x, k_grid.y, k_grid.z,
                                 32, 16, 1, 0, 0, k4_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("gemm K failed"); goto cleanup;
            }

            // V projection: [1 x DKV] = h @ Wv^T
            w_packed = L.Wv.d_packed; w_scales = L.Wv.d_scales; w_scales_f = L.Wv.d_scales_f;
            for (int b = 0; b < batch_size; ++b) d_output_arr[b] = m->layer_scratch[l].d_v;
            void* v4_args[] = { &d_input_arr[0], &w_packed, &w_scales, &w_scales_f, &d_output_arr[0],
                               &one_int, &DKV_b, &E_local };
            (void)hipModuleLaunchKernel(fn_gemm_act_fp4, k_grid.x, k_grid.y, k_grid.z,
                                 32, 16, 1, 0, 0, v4_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("gemm V failed"); goto cleanup;
            }
        }

        // --- c) Per-head staging (FP32 -> FP16) - CPU-side with batching ---
        // Build FP16 staging arrays for Q/K across all sequences
        // This is the bottleneck - requires reading device results back to host

        // --- d) RoPE on Q/K - build tables once for all positions present in batch ---
        // Collect unique positions from all sequences and build combined tables
        std::vector<int> unique_positions;
        std::vector<int> pos_to_seq(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            int last_pos = positions[b] + (seq_lens ? seq_lens[b] : seq_len) - 1;
            pos_to_seq[b] = unique_positions.size();
            bool found = false;
            for (int p : unique_positions) { if (p == last_pos) { found = true; break; } }
            if (!found) unique_positions.push_back(last_pos);
        }

        int HQ_local = HQ, HD2_local = HD / 2;

        // RoPE kernel launch with per-sequence position offsets via LDS LUT arrays
        void* rope_args[] = { &m->layer_scratch[l].d_qh, &unique_positions[0], &pos_to_seq[0],
                             &batch_size, &HQ_local, &HD2_local };
        dim3 rope_grid(HQ, 1, unique_positions.size());
        (void)hipModuleLaunchKernel(fn_rope, rope_grid.x, rope_grid.y, rope_grid.z,
                             32, 1, 1, 0, 0, rope_args, nullptr);
        if (hipGetLastError() != hipSuccess) {
            qwen_set_error("RoPE failed"); goto cleanup;
        }

        // --- e) KV Store - batched ---
        {
            std::vector<void*> k_src_arr(batch_size), v_src_arr(batch_size);
            std::vector<void*> k_dst_arr(batch_size), v_dst_arr(batch_size);
            std::vector<int> pos_arr(batch_size), ctx_arr(batch_size);

            for (int b = 0; b < batch_size; ++b) {
                k_src_arr[b] = m->layer_scratch[l].d_rks;   // roped K output (offset by seq)
                v_src_arr[b] = m->layer_scratch[l].d_v;     // V output (offset by seq)
                k_dst_arr[b] = kv_cache_k_row(eng->kvc, l, b);  // per-batch KV cache row
                v_dst_arr[b] = kv_cache_v_row(eng->kvc, l, b);
                pos_arr[b] = positions[b];
                ctx_arr[b] = seq_lens ? seq_lens[b] : seq_len;
            }

            int DKV_b = DKV_local;
            void* kvs_args[] = { &k_src_arr[0], &v_src_arr[0],
                                 &k_dst_arr[0], &v_dst_arr[0],
                                 &batch_size, &DKV_b, &pos_arr[0], &ctx_arr[0] };
            dim3 kv_grid((HKV + 7) / 8, 1, batch_size);
            (void)hipModuleLaunchKernel(fn_kv_store, kv_grid.x, kv_grid.y, kv_grid.z,
                                 32, 1, 1, 0, 0, kvs_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("kv_store failed"); goto cleanup;
            }
        }

        // --- f) Attention per head - softmax and GEMM are batched ---
        {
            int CTX_b = CTX_local;
            float scale_f = 1.0f / sqrtf((float)HD);

            // Batched softmax for all sequences in parallel
            void* sm_args[] = { &d_scores_batch, &batch_size, &CTX_b, &scale_f,
                               &d_qpos_batch[0], &seq_len };
            dim3 softmax_grid(1, 1, batch_size);
            (void)hipModuleLaunchKernel(fn_softmax, (seq_len + 31) / 32, 1, batch_size,
                                 32, 1, 1, 0, 0, sm_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("softmax failed"); goto cleanup;
            }

            // Batched GEMM for attention output: probs @ V -> [batch x HQ x HD]
            int HD_b = HD;
            void* attn_gemm_args[] = { &m->layer_scratch[l].d_pf,    // probs FP16 base (offset per seq)
                                       m->layer_scratch[l].d_vb2,   // V cache base (per head/layer)
                                       d_logits_batch,              // output logits buffer
                                       &batch_size, &HD_b, &CTX_b };
            dim3 attn_grid((HQ + 15) / 16, (HD + 15) / 16, batch_size);
            (void)hipModuleLaunchKernel(fn_fp16_gemm, attn_grid.x, attn_grid.y, attn_grid.z,
                                 16, 16, 1, 0, 0, attn_gemm_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("attention GEMM failed"); goto cleanup;
            }
        }

        // --- g) O-Projection (gemm_act_fp4) - batched ---
        {
            std::vector<void*> inp_arr(batch_size), out_arr(batch_size);
            for (int b = 0; b < batch_size; ++b) {
                inp_arr[b] = d_logits_batch + (size_t)b * m->vocab_size * sizeof(float);  // attention output
                out_arr[b] = m->layer_scratch[l].d_down;   // residual add destination (offset by seq)
            }

            void* o_args[] = { &inp_arr[0], &L.Wo.d_packed, &L.Wo.d_scales, &L.Wo.d_scales_f, &out_arr[0],
                              &one_int, &E_local, &DQ_local };
            dim3 o_grid(1, (E + 15) / 16, batch_size);
            (void)hipModuleLaunchKernel(fn_gemm_act_fp4, o_grid.x, o_grid.y, o_grid.z,
                                 32, 16, 1, 0, 0, o_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("O projection failed"); goto cleanup;
            }

            // Residual add: h += O-projection output (host-side loop with GPU async execution)
            for (int i = 0; i < E; ++i) {
                __half h_val = ((__half*)m->layer_scratch[l].h)[i];
                float h_f = __half2float(h_val);
                float o_f = ((float*)d_logits_batch)[i];  // from attention output
                ((__half*)m->layer_scratch[l].h)[i] = __float2half(h_f + o_f);
            }

            // --- h) RMSNorm (MLP) - batched ---
            void* ln_mlp_args[] = { &m->layer_scratch[l].h, &L.d_mlp_ln, &m->layer_scratch[l].d_hmlp,
                                   &E_local, &E_local, (void*)&eps_f };
            dim3 mlp_norm_grid((E + 31) / 32, 1, batch_size);
            (void)hipModuleLaunchKernel(fn_layernorm, mlp_norm_grid.x, mlp_norm_grid.y, mlp_norm_grid.z,
                                 32, 1, 1, 0, 0, ln_mlp_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("RMSNorm MLP failed"); goto cleanup;
            }

            // --- i) SwiGLU (gate/up/down) - batched ---
            std::vector<void*> h_arr(batch_size), gate_arr(batch_size), up_arr(batch_size);
            for (int b = 0; b < batch_size; ++b) {
                h_arr[b] = m->layer_scratch[l].h_mlp;   // MLP input (offset by seq)
                gate_arr[b] = m->layer_scratch[l].d_gate;   // gate output (offset by seq)
                up_arr[b] = m->layer_scratch[l].d_up;     // up output (offset by seq)
            }

            void* swiglu_args[] = { &h_arr[0],
                                   &L.W_gate.d_packed, &L.W_gate.d_scales,
                                   &L.W_up.d_packed, &L.W_up.d_scales,
                                   m->layer_scratch[l].d_down,    // down projection output (residual destination)
                                   &batch_size, &E_local, &E_local };
            dim3 swiglu_grid(1, (E + 15) / 16, batch_size);
            (void)hipModuleLaunchKernel(fn_swiglu, swiglu_grid.x, swiglu_grid.y, swiglu_grid.z,
                                 32, 16, 1, 0, 0, swiglu_args, nullptr);
            if (hipGetLastError() != hipSuccess) {
                qwen_set_error("SwiGLU failed"); goto cleanup;
            }

            // --- j) Residual add (MLP output) - host-side with async GPU execution ---
            for (int i = 0; i < E; ++i) {
                __half h_val = ((__half*)m->layer_scratch[l].h)[i];
                float h_f = __half2float(h_val);
                float swi_f = ((float*)d_logits_batch)[i];  // SwiGLU output from logits buffer
                ((__half*)m->layer_scratch[l].h)[i] = __float2half(h_f + swi_f);
            }
        }
    }

    // ---- Final LayerNorm across all sequences ----
    {
        void* lnf_args[] = { &m->layer_scratch[0].h, &eng->d_final_ln, m->layer_scratch[0].d_h,
                            &E_local, &E_local, (void*)&eps_f };
        dim3 final_norm_grid((E + 31) / 32, 1, batch_size);
        (void)hipModuleLaunchKernel(fn_layernorm, final_norm_grid.x, final_norm_grid.y, final_norm_grid.z,
                             32, 1, 1, 0, 0, lnf_args, nullptr);
        if (hipGetLastError() != hipSuccess) {
            qwen_set_error("final layer norm failed"); goto cleanup;
        }

        // ---- LM Head projection - batched across sequences ----
        std::vector<void*> inp_arr(batch_size), out_arr(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            inp_arr[b] = m->layer_scratch[0].d_h;   // normalized output (offset by seq)
            out_arr[b] = d_logits_batch + (size_t)b * m->vocab_size * sizeof(float);  // per-sequence logits
        }

        int vocab_b = m->vocab_size;
        void* lm_args[] = { &inp_arr[0], &eng->d_lm_head_packed,
                           &eng->d_lm_head_scales, &eng->d_lm_head_scales_f, &out_arr[0],
                           &one_int, &vocab_b, &E_local };
        dim3 lm_grid(1, (vocab_b + 15) / 16, batch_size);
        (void)hipModuleLaunchKernel(fn_gemm_act_fp4, lm_grid.x, lm_grid.y, lm_grid.z,
                             32, 16, 1, 0, 0, lm_args, nullptr);
        if (hipGetLastError() != hipSuccess) {
            qwen_set_error("LM head failed"); goto cleanup;
        }

        // Copy logits to host-visible buffer for sampling
        herr = hipMemcpy(logits_out, d_logits_batch, logits_bytes, hipMemcpyDeviceToHost);
        if (herr != hipSuccess) {
            qwen_set_error("logits copy failed"); goto cleanup;
        }
    }

    (void)hipDeviceSynchronize();

cleanup:
    (void)hipFree(d_scores_batch);
    (void)hipFree(d_qpos_batch);
    (void)hipFree(d_logits_batch);
    return hipGetLastError() == hipSuccess ? 0 : INFERENCE_ERR_KERNEL_LAUNCH;
}
}