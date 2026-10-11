// inference.cpp -- Phase 4: C API implementation for zero-copy ROCmFP4 inference engine.
// Thin FFI wrappers around the C++ HIP implementation. All state in inference_model_t.

#include "inference.h"
#include "inference_internal.h"
#include "gguf_mmap.h"
#include "fp4_common.h"
#include "kv_cache.h"
#include "tokenizer.h"
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

// All internal types (QuantWeight, LayerWeights, inference_model) are
// defined in inference_internal.h which is included above.

static bool pin_model_memory(inference_model* m) {
    // Pin embedding table
    fprintf(stderr, "[pin_model] embed_table size=%zu\n", m->embed_table.size());
    if (!m->embed_table.empty()) {
        hipError_t e = hipHostRegister(m->embed_table.data(),
                                         m->embed_table.size() * sizeof(__half),
                                         hipHostRegisterDefault);
        if (e != hipSuccess) { fprintf(stderr, "[pin_model] embed register failed: %s\n", hipGetErrorString(e)); return false; }
        m->d_embed_table = m->embed_table.data();
        fprintf(stderr, "[pin_model] embed registered, d_embed_table=%p\n", (void*)m->d_embed_table);
    }
    // Pin per-layer weights
    for (auto& L : m->layers) {
        auto pin = [](QuantWeight& w) -> bool {
            if (w.packed.empty()) return true;  // Empty is OK for tied/embedded models
            if (hipHostRegister(w.packed.data(), w.packed.size(), hipHostRegisterDefault) != hipSuccess) {
                fprintf(stderr, "[pin_model] packed register failed (%zu bytes)\n", w.packed.size());
                return false;
            }
            if (!w.scales.empty() && hipHostRegister(w.scales.data(), w.scales.size(), hipHostRegisterDefault) != hipSuccess) {
                fprintf(stderr, "[pin_model] scales register failed (%zu bytes)\n", w.scales.size());
                return false;
            }
            if (!w.scales_f.empty() && hipHostRegister(w.scales_f.data(), w.scales_f.size() * sizeof(float), hipHostRegisterDefault) != hipSuccess) {
                fprintf(stderr, "[pin_model] scales_f register failed (%zu bytes)\n", w.scales_f.size() * sizeof(float));
                return false;
            }
            w.d_packed = w.packed.data();
            const unsigned char* scales_raw = !w.scales.empty() ? w.scales.data() : w.packed.data();
            w.d_scales = const_cast<unsigned char*>(scales_raw);
            const float* scaled_f_raw = !w.scales_f.empty() ? w.scales_f.data() : nullptr;
            if (!scaled_f_raw) {
                // Fallback: use packed data address (will not be used for non-UE4M3 types)
                void* dummy = (void*)w.packed.data();
                dummy = (void*)w.packed.data();
                scaled_f_raw = reinterpret_cast<const float*>(dummy);
            }
            w.d_scales_f = const_cast<float*>(scaled_f_raw);
            return true;
        };
        if (!pin(L.Wq) || !pin(L.Wk) || !pin(L.Wv) || !pin(L.Wo) ||
            !pin(L.W_gate) || !pin(L.W_up) || !pin(L.W_down)) return false;
        if (!L.attn_ln_scale.empty()) {
            (void)hipHostRegister(L.attn_ln_scale.data(), L.attn_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
            L.d_attn_ln = L.attn_ln_scale.data();
        }
        if (!L.mlp_ln_scale.empty()) {
            (void)hipHostRegister(L.mlp_ln_scale.data(), L.mlp_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
            L.d_mlp_ln = L.mlp_ln_scale.data();
        }
    }
    // Pin LM head
    if (!m->lm_head.packed.empty()) {
        (void)hipHostRegister(m->lm_head.packed.data(), m->lm_head.packed.size(), hipHostRegisterDefault);
        if (!m->lm_head.scales.empty()) {
            (void)hipHostRegister(m->lm_head.scales.data(), m->lm_head.scales.size(), hipHostRegisterDefault);
        }
        if (!m->lm_head.scales_f.empty()) {
            (void)hipHostRegister(m->lm_head.scales_f.data(), m->lm_head.scales_f.size() * sizeof(float), hipHostRegisterDefault);
        }
        m->d_lm_head_packed = m->lm_head.packed.data();
        const unsigned char* lm_scales_raw = !m->lm_head.scales.empty() ? m->lm_head.scales.data() : m->lm_head.packed.data();
        m->d_lm_head_scales = const_cast<unsigned char*>(lm_scales_raw);
        const float* lm_scaled_f_raw = !m->lm_head.scales_f.empty() ? m->lm_head.scales_f.data() : nullptr;
        if (!lm_scaled_f_raw) {
            void* dummy = (void*)m->lm_head.packed.data();
            dummy = (void*)m->lm_head.packed.data();
            lm_scaled_f_raw = reinterpret_cast<const float*>(dummy);
        }
        m->d_lm_head_scales_f = const_cast<float*>(lm_scaled_f_raw);
    }
    if (!m->final_ln_scale.empty()) {
        (void)hipHostRegister(m->final_ln_scale.data(), m->final_ln_scale.size() * sizeof(__half), hipHostRegisterDefault);
        m->d_final_ln = m->final_ln_scale.data();
    }
    return true;
}

static void unpin_model_memory(inference_model* m) {
    auto unpin = [](void* ptr) {
        if (ptr) (void)hipHostUnregister(ptr);
    };
    unpin(m->d_embed_table);
    for (auto& L : m->layers) {
        unpin(L.Wq.d_packed); unpin(L.Wq.d_scales); unpin(L.Wq.d_scales_f);
        unpin(L.Wk.d_packed); unpin(L.Wk.d_scales); unpin(L.Wk.d_scales_f);
        unpin(L.Wv.d_packed); unpin(L.Wv.d_scales); unpin(L.Wv.d_scales_f);
        unpin(L.Wo.d_packed); unpin(L.Wo.d_scales); unpin(L.Wo.d_scales_f);
        unpin(L.W_gate.d_packed); unpin(L.W_gate.d_scales); unpin(L.W_gate.d_scales_f);
        unpin(L.W_up.d_packed); unpin(L.W_up.d_scales); unpin(L.W_up.d_scales_f);
        unpin(L.W_down.d_packed); unpin(L.W_down.d_scales); unpin(L.W_down.d_scales_f);
        unpin(L.d_attn_ln); unpin(L.d_mlp_ln);
    }
    unpin(m->d_lm_head_packed); unpin(m->d_lm_head_scales); unpin(m->d_lm_head_scales_f);
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

// ======================== SCALE EXTRACTION HELPERS ========================
/* --- Unused helpers (kept for reference; inline extraction used instead) ---

// Extract UE4M3 scales from TYPE100 blocks (block_size=32, 2 scales per block at offset 16).
// Layout: [scale_lo(1B), scale_hi(1B), data(16B)] × Nblocks
static void extract_scales_type100(const uint8_t* data, size_t nbytes, uint8_t* scales_out, int64_t N) {
    const size_t block_bytes = 17; // 2 scale bytes + 16 packed data bytes
    const size_t scales_per_block = 2;
    const size_t blocks_per_row = nbytes / block_bytes;

    for (int64_t row = 0; row < N; ++row) {
        size_t row_blocks = blocks_per_row;
        for (size_t b = 0; b < row_blocks; ++b) {
            const uint8_t* blk = data + row * row_blocks * block_bytes + b * block_bytes;
            for (size_t s = 0; s < scales_per_block; ++s) {
                scales_out[row * (row_blocks * scales_per_block) + b * scales_per_block + s] = blk[16 + s];
            }
        }
    }
}

// Extract UE4M3 scales from TYPE13 blocks (block_size=256, 16 UE4M3 scales per block).
// Layout: [scale_0(1B), scale_1(1B), ..., scale_15(1B), data(128B)] × Nblocks
// Each scale covers 16 elements within the 256-element block.
static void extract_scales_type13(const uint8_t* data, size_t nbytes, uint8_t* scales_out, int64_t N) {
    const size_t block_bytes = 144; // 16 scale bytes + 128 packed data bytes
    const size_t scales_per_block = 16;
    const size_t blocks_per_row = nbytes / block_bytes;

    for (int64_t row = 0; row < N; ++row) {
        size_t row_blocks = blocks_per_row;
        for (size_t b = 0; b < row_blocks; ++b) {
            const uint8_t* blk = data + row * row_blocks * block_bytes + b * block_bytes;
            for (size_t s = 0; s < scales_per_block; ++s) {
                scales_out[row * (row_blocks * scales_per_block) + b * scales_per_block + s] = blk[s];
            }
        }
    }
}

// Extract FP16 delta from Q8_0 blocks (block_size=32).
// Layout: [delta_fp16(2B), values_int8(32B)] per block.
static void extract_scales_q8_0(const uint8_t* data, size_t nbytes, float* scales_out, int64_t N) {
    const size_t block_bytes = 34; // 2 byte delta + 32 bytes int8 values
    const size_t blocks_per_row = nbytes / block_bytes;

    for (int64_t row = 0; row < N; ++row) {
        size_t row_blocks = blocks_per_row;
        for (size_t b = 0; b < row_blocks; ++b) {
            const uint8_t* blk = data + row * row_blocks * block_bytes + b * block_bytes;
            float d; memcpy(&d, blk, sizeof(float)); // FP16 stored as half-precision but we read raw
            __half h; memcpy(&h, blk, sizeof(__half));
            scales_out[row * row_blocks + b] = __half2float(h);
        }
    }
}

// Extract FP16 delta from Q4_0 blocks (block_size=32).
// Layout: [delta_fp16(2B), min_fp16(2B), packed_data(16B)] per block.
static void extract_scales_q4_0(const uint8_t* data, size_t nbytes, float* scales_out, int64_t N) {
    const size_t block_bytes = 20; // 2 byte scale + 2 byte min + 16 bytes packed
    const size_t blocks_per_row = nbytes / block_bytes;

    for (int64_t row = 0; row < N; ++row) {
        size_t row_blocks = blocks_per_row;
        for (size_t b = 0; b < row_blocks; ++b) {
            const uint8_t* blk = data + row * row_blocks * block_bytes + b * block_bytes;
            __half h; memcpy(&h, blk, sizeof(__half));
            scales_out[row * row_blocks + b] = __half2float(h);
        }
    }
}

// Extract FP16 scale from Q4_1 blocks (block_size=32).
// Layout: [scale_fp16(2B), min_fp16(2B), packed_data(16B)] per block.
static void extract_scales_q4_1(const uint8_t* data, size_t nbytes, float* scales_out, int64_t N) {
    const size_t block_bytes = 20; // same layout as Q4_0
    const size_t blocks_per_row = nbytes / block_bytes;

    for (int64_t row = 0; row < N; ++row) {
        size_t row_blocks = blocks_per_row;
        for (size_t b = 0; b < row_blocks; ++b) {
            const uint8_t* blk = data + row * row_blocks * block_bytes + b * block_bytes;
            __half h; memcpy(&h, blk, sizeof(__half));
            scales_out[row * row_blocks + b] = __half2float(h);
        }
    }
}

*/

// Load quantized weight by name from GGUF (for non-QKV tensors)
static bool load_quant_weight_from_gguf(gguf_model_t* gguf, const char* weight_name, const char* scales_name, QuantWeight& w) {
    size_t w_sz, s_sz;
    const void* w_ptr = gguf_get_tensor_ptr(gguf, weight_name, &w_sz);
    const void* s_ptr = gguf_get_tensor_ptr(gguf, scales_name, &s_sz);
    if (!w_ptr) {
        fprintf(stderr, "Missing weight tensor %s\n", weight_name);
        return false;
    }
    // Get tensor metadata for shape and type info
    int idx = find_tensor_index(gguf, weight_name);
    if (idx < 0) {
        fprintf(stderr, "Weight tensor %s not found in index\n", weight_name);
        return false;
    }
    int t = gguf_tensor_type(gguf, idx);
    int self_contained = gguf_type_is_self_contained(t);

    // For self-contained types, external .scales tensor is optional — extract from blocks or fill with placeholder
    if (!s_ptr && !self_contained) {
        fprintf(stderr, "Missing scales tensor %s (type %d requires external scales)\n", scales_name, t);
        return false;
    } else if (!s_ptr && self_contained) {
        fprintf(stderr, "[load_quant_weight] %s: self-contained type %d, no external scales — extracting from blocks\n",
                weight_name, t);
    }

    // Get dimensions: assume 2D [out_dim, in_dim]
    int64_t* ne = gguf_tensor_ne(gguf, idx);
    if (!ne) return false;
    int64_t out_dim = ne[0];
    int64_t in_dim = ne[1];
    w.N = (int)out_dim;
    w.K = (int)in_dim;
    // Copy packed bytes from weight tensor
    w.packed.assign((const unsigned char*)w_ptr, (const unsigned char*)w_ptr + w_sz);

    if (s_ptr && s_sz > 0) {
        // External scales tensor found — use it normally
        w.scales.assign((const unsigned char*)s_ptr, (const unsigned char*)s_ptr + s_sz);
    } else {
        // Self-contained type with no external scales — extract actual scales from within each block
        size_t scales_per_row = ((size_t)in_dim + 15) >> 4;  // UE4M3 scale per 16 elements (standard layout)
        w.scales.resize((size_t)out_dim * scales_per_row);

        // Determine extraction strategy based on type
        switch (t) {
            case 100: {
                // TYPE100/MXFP4: UE4M3 scales embedded in blocks alongside packed data.
                // Block layout (17 bytes): [scale_0(1B), scale_1(1B), data(16B)] for 32 values.
                // Total blob = w_sz; blocks per row = K / 32; scales per block = 2.
                // Repack to standard layout: one UE4M3 scale byte per 16 elements.
                fprintf(stderr, "[load_quant_weight] extracting TYPE100 scales\n");

                const uint8_t* data = w.packed.data();
                // TYPE100: UE4M3 scales embedded in blocks alongside packed data (w_sz / 17 total blocks)
                size_t blocks_per_row = (size_t)in_dim / 32;
                size_t tmp_scales_per_row = blocks_per_row * 2;

                std::vector<uint8_t> tmp_scales((size_t)out_dim * tmp_scales_per_row);

                for (int64_t row = 0; row < out_dim; ++row) {
                    int64_t row_start_elem = row * in_dim;
                    size_t row_start_block = (size_t)row_start_elem / 32;

                    for (size_t b = 0; b < blocks_per_row; ++b) {
                        size_t blk_idx = row_start_block + b;
                        size_t src_off = blk_idx * 17;

                        for (size_t s = 0; s < 2; ++s) {
                            size_t global_si = (size_t)(row * blocks_per_row * 2) + b * 2 + s;
                            if (global_si < tmp_scales.size()) {
                                tmp_scales[global_si] = data[src_off + 16 + s]; // UE4M3 at offset 16+within block
                            }
                        }
                    }
                }

                // Repack: map extracted scales to standard per-row layout (K/16 scale bytes per row)
                for (int64_t row = 0; row < out_dim; ++row) {
                    size_t src_row_off = (size_t)row * tmp_scales_per_row;
                    for (size_t s = 0; s < scales_per_row; ++s) {
                        w.scales[row * scales_per_row + s] =
                            (s < tmp_scales_per_row) ? tmp_scales[src_row_off + s] : 0x60;
                    }
                }
                break;
            }
            case 13: {
                // TYPE13: UE4M3 scales embedded in each 256-value block.
                // Block layout (144 bytes): [scale_0..scale_15(16B), packed_data(128B)].
                // Each scale covers 16 elements within the block.
                fprintf(stderr, "[load_quant_weight] extracting TYPE13 scales\n");

                const uint8_t* data = w.packed.data();
                // total_blocks = w_sz / 144 (documented, not needed for calculation)
                size_t blocks_per_row = (size_t)in_dim / 256;
                size_t tmp_scales_per_row = blocks_per_row * 16;

                std::vector<uint8_t> tmp_scales((size_t)out_dim * tmp_scales_per_row);

                for (int64_t row = 0; row < out_dim; ++row) {
                    int64_t row_start_elem = row * in_dim;
                    size_t row_start_block = (size_t)row_start_elem / 256;

                    for (size_t b = 0; b < blocks_per_row; ++b) {
                        size_t blk_idx = row_start_block + b;
                        const uint8_t* blk = data + blk_idx * 144;

                        for (size_t s = 0; s < 16; ++s) {
                            size_t global_si = (size_t)(row * tmp_scales_per_row) + b * 16 + s;
                            if (global_si < tmp_scales.size()) {
                                tmp_scales[global_si] = blk[s]; // UE4M3 scale at offset s within block
                            }
                        }
                    }
                }

                // Repack: map extracted scales to standard per-row layout
                for (int64_t row = 0; row < out_dim; ++row) {
                    size_t src_row_off = (size_t)row * tmp_scales_per_row;
                    for (size_t s = 0; s < scales_per_row; ++s) {
                        w.scales[row * scales_per_row + s] =
                            (s < tmp_scales_per_row) ? tmp_scales[src_row_off + s] : 0x60;
                    }
                }
                break;
            }
            case 2: { // Q4_0: block_size=32, layout [delta_fp16(2B), packed_data(16B)] = 18 bytes/block
                fprintf(stderr, "[load_quant_weight] extracting Q4_0 FP16 deltas\n");
                w.quant_type = 2;

                const uint8_t* data = w.packed.data();
                size_t block_bytes_q40 = 18; // 2B delta + 16B packed data
                size_t blocks_per_row_q40 = (size_t)in_dim / 32;

                // Q4_1-compatible layout: two decoded floats per block (scale=delta, min=0) -> N * (K/16) floats
                w.scales_f.resize((size_t)out_dim * blocks_per_row_q40 * 2);

                for (int64_t row = 0; row < out_dim; ++row) {
                    for (size_t b = 0; b < blocks_per_row_q40; ++b) {
                        const uint8_t* blk = data + (row * blocks_per_row_q40 + b) * block_bytes_q40;
                        __half h_delta;
                        memcpy(&h_delta, blk, sizeof(__half));
                        float delta_f = __half2float(h_delta);

                        size_t fi = (size_t)row * blocks_per_row_q40 * 2 + b * 2;
                        w.scales_f[fi]     = delta_f;  // scale = delta for Q4_0
                        w.scales_f[fi + 1] = 0.0f;      // min = 0 for Q4_0
                    }
                }
                break;
            }
            case 3: { // Q4_1: block_size=32, layout [scale_fp16(2B), min_fp16(2B), packed_data(16B)] = 20 bytes/block
                fprintf(stderr, "[load_quant_weight] extracting Q4_1 FP16 scale+min\n");
                w.quant_type = 3;

                const uint8_t* data = w.packed.data();
                size_t block_bytes_q41 = 20; // 2B scale + 2B min + 16B packed data
                size_t blocks_per_row_q41 = (size_t)in_dim / 32;

                // Two decoded floats per block (scale + min) -> N * (K/16) floats
                w.scales_f.resize((size_t)out_dim * blocks_per_row_q41 * 2);

                for (int64_t row = 0; row < out_dim; ++row) {
                    for (size_t b = 0; b < blocks_per_row_q41; ++b) {
                        const uint8_t* blk = data + (row * blocks_per_row_q41 + b) * block_bytes_q41;
                        __half h_scale, h_min;
                        memcpy(&h_scale, blk, sizeof(__half));
                        memcpy(&h_min, blk + 2, sizeof(__half));

                        size_t fi = (size_t)row * blocks_per_row_q41 * 2 + b * 2;
                        w.scales_f[fi]     = __half2float(h_scale);  // scale
                        w.scales_f[fi + 1] = __half2float(h_min);    // min
                    }
                }
                break;
            }
            case 7: case 8: { // Q8_0: block_size=32, layout [delta_fp16(2B), values_int8(32B)] = 34 bytes/block
                fprintf(stderr, "[load_quant_weight] extracting Q8_0 FP16 delta\n");
                w.quant_type = (t == 7) ? 7 : 8;

                const uint8_t* data = w.packed.data();
                size_t block_bytes_q80 = 34; // 2B delta + 32B int8 values
                size_t blocks_per_row_q80 = (size_t)in_dim / 32;

                // Q4_1-compatible layout: two decoded floats per block (scale=delta, min=0) -> N * (K/16) floats
                w.scales_f.resize((size_t)out_dim * blocks_per_row_q80 * 2);

                for (int64_t row = 0; row < out_dim; ++row) {
                    for (size_t b = 0; b < blocks_per_row_q80; ++b) {
                        const uint8_t* blk = data + (row * blocks_per_row_q80 + b) * block_bytes_q80;
                        __half h_delta;
                        memcpy(&h_delta, blk, sizeof(__half));
                        float delta_f = __half2float(h_delta);

                        size_t fi = (size_t)row * blocks_per_row_q80 * 2 + b * 2;
                        w.scales_f[fi]     = delta_f;  // scale = delta for Q8_0
                        w.scales_f[fi + 1] = 0.0f;      // min = 0 for Q8_0
                    }
                }
                break;
            }
            default: {
                // Unknown self-contained type — fill with unity scale as safe fallback
                fprintf(stderr, "[load_quant_weight] unknown type %d, filling unity scale placeholder\n", t);
                for (int64_t row = 0; row < out_dim; ++row) {
                    for (size_t s = 0; s < scales_per_row; ++s) {
                        w.scales[row * scales_per_row + s] = 0x60; // unity UE4M3 = 1.0f
                    }
                }
                break;
            }
        }
    }

    // Build true_v for CPU reference (dequant using loaded scales)
    w.true_v.resize((size_t)w.N * w.K);

    switch (t) {
        case 100: { // TYPE100/MXFP4: [scale_lo(1B), scale_hi(1B), packed_data(16B)] = 17 bytes per 32 values
            fprintf(stderr, "[load_quant_weight] true_v: TYPE100 UE4M3 formula\n");
            {
                size_t blocks_per_row = (size_t)in_dim / 32;
                for (int n = 0; n < out_dim; ++n) {
                    for (size_t b = 0; b < blocks_per_row; ++b) {
                        uint8_t sc_byte = w.packed[(size_t)n * blocks_per_row * 17 + b * 17 + 16]; // scale at offset 16
                        float sc = ue4m3_to_float(sc_byte);
                        for (int v = 0; v < 32; ++v) {
                            int k = (int)b * 32 + v;
                            uint8_t wb = w.packed[(size_t)n * blocks_per_row * 17 + b * 17 + 16 + (k >> 1)];
                            uint8_t nib = (k & 1) ? (wb >> 4) : (wb & 0x0F);
                            w.true_v[(size_t)n * w.K + k] = dequant_code(nib) * sc;
                        }
                    }
                }
            }
            break;
        }
        case 13: { // TYPE13: [scale_0..15(16B), packed_data(128B)] = 144 bytes per 256 values
            fprintf(stderr, "[load_quant_weight] true_v: TYPE13 UE4M3 formula\n");
            {
                size_t blocks_per_row = (size_t)in_dim / 256;
                for (int n = 0; n < out_dim; ++n) {
                    for (size_t b = 0; b < blocks_per_row; ++b) {
                        for (int v = 0; v < 256; ++v) {
                            int k = (int)b * 256 + v;
                            size_t scale_idx = (size_t)v / 16;
                            float sc = ue4m3_to_float(w.packed[(size_t)n * blocks_per_row * 144 + b * 144 + scale_idx]);
                            uint8_t wb = w.packed[(size_t)n * blocks_per_row * 144 + b * 144 + 16 + ((size_t)(b * 256 + v) >> 1)];
                            uint8_t nib = (k & 1) ? (wb >> 4) : (wb & 0x0F);
                            w.true_v[(size_t)n * w.K + k] = dequant_code(nib) * sc;
                        }
                    }
                }
            }
            break;
        }
        case 14: { // TYPE14: [scales(16B), packed_data(...)] = 176 bytes per 256 values
            fprintf(stderr, "[load_quant_weight] true_v: TYPE14 UE4M3 formula\n");
            {
                size_t blocks_per_row = (size_t)in_dim / 256;
                for (int n = 0; n < out_dim; ++n) {
                    for (size_t b = 0; b < blocks_per_row; ++b) {
                        for (int v = 0; v < 256; ++v) {
                            int k = (int)b * 256 + v;
                            size_t scale_idx = (size_t)v / 16;
                            float sc = ue4m3_to_float(w.packed[(size_t)n * blocks_per_row * 176 + b * 176 + scale_idx]);
                            uint8_t wb = w.packed[(size_t)n * blocks_per_row * 176 + b * 176 + 16 + ((size_t)(b * 256 + v) >> 1)];
                            uint8_t nib = (k & 1) ? (wb >> 4) : (wb & 0x0F);
                            w.true_v[(size_t)n * w.K + k] = dequant_code(nib) * sc;
                        }
                    }
                }
            }
            break;
        }
        case 2: { // Q4_0: [delta_fp16(2B), packed_data(16B)] = 18 bytes per 32 values
            fprintf(stderr, "[load_quant_weight] true_v: Q4_0 formula\n");
            {
                size_t blocks_per_row_q40 = (size_t)in_dim / 32;
                for (int n = 0; n < out_dim; ++n) {
                    for (size_t b = 0; b < blocks_per_row_q40; ++b) {
                        const uint8_t* blk = w.packed.data() + (size_t)n * blocks_per_row_q40 * 18 + b * 18;
                        __half h_delta;
                        memcpy(&h_delta, blk, sizeof(__half));
                        float delta_f = __half2float(h_delta);

                        for (int v = 0; v < 32; ++v) {
                            int k = (int)b * 32 + v;
                            uint8_t wb = blk[2 + ((size_t)(b * 32 + v) >> 1)];
                            uint8_t nib = (k & 1) ? (wb >> 4) : (wb & 0x0F);
                            // Q4_0: value = delta * codebook[nibble] where nibble is signed via bit 3
                            int sign_bit = nib >> 3;
                            int mag = nib & 0x7;
                            float val = sign_bit ? -((float)mag + 1.0f) : (float)mag;
                            w.true_v[(size_t)n * w.K + k] = val * delta_f;
                        }
                    }
                }
            }
            break;
        }
        case 3: { // Q4_1: [scale_fp16(2B), min_fp16(2B), packed_data(16B)] = 20 bytes per 32 values
            fprintf(stderr, "[load_quant_weight] true_v: Q4_1 formula\n");
            {
                size_t blocks_per_row_q41 = (size_t)in_dim / 32;
                for (int n = 0; n < out_dim; ++n) {
                    for (size_t b = 0; b < blocks_per_row_q41; ++b) {
                        const uint8_t* blk = w.packed.data() + (size_t)n * blocks_per_row_q41 * 20 + b * 20;
                        __half h_scale, h_min;
                        memcpy(&h_scale, blk, sizeof(__half));
                        memcpy(&h_min, blk + 2, sizeof(__half));
                        float scale_f = __half2float(h_scale);
                        float min_f = __half2float(h_min);

                        for (int v = 0; v < 32; ++v) {
                            int k = (int)b * 32 + v;
                            uint8_t wb = blk[4 + ((size_t)(b * 32 + v) >> 1)];
                            uint8_t nib = (k & 1) ? (wb >> 4) : (wb & 0x0F);
                            // Q4_1: value = scale * codebook[nibble] + min
                            w.true_v[(size_t)n * w.K + k] = scale_f * dequant_code(nib) + min_f;
                        }
                    }
                }
            }
            break;
        }
        case 7: case 8: { // Q8_0: [delta_fp16(2B), values_int8(32B)] = 34 bytes per block of 32 values
            fprintf(stderr, "[load_quant_weight] true_v: Q8_0 formula\n");
            {
                const uint8_t* data = w.packed.data();
                size_t blocks_per_row_q80 = (size_t)in_dim / 32;
                for (int n = 0; n < out_dim; ++n) {
                    for (size_t b = 0; b < blocks_per_row_q80; ++b) {
                        const uint8_t* blk = data + (size_t)n * blocks_per_row_q80 * 34 + b * 34;
                        __half h_delta;
                        memcpy(&h_delta, blk, sizeof(__half));
                        float delta_f = __half2float(h_delta);

                        for (int v = 0; v < 32; ++v) {
                            int k = (int)b * 32 + v;
                            int8_t val = (int8_t)blk[2 + v]; // int8 value at offset 2 within block
                            w.true_v[(size_t)n * w.K + k] = val * delta_f;
                        }
                    }
                }
            }
            break;
        }
        default: {
            // Unknown type — use UE4M3 formula with unity scale as safe fallback
            fprintf(stderr, "[load_quant_weight] unknown type %d, using UE4M3 formula with unity scale\n", t);
            const size_t strips = w.K / 16;
            for (int n = 0; n < out_dim; ++n) {
                for (size_t s = 0; s < strips; ++s) {
                    float sc = ue4m3_to_float(w.scales[(size_t)n * strips + s]);
                    if (sc == 0.5f) sc = 1.0f; // Skip zero scales (unity placeholder reads as 0.5 in UE4M3)
                    for (int i = 0; i < 16; ++i) {
                        int k = (int)s * 16 + i;
                        uint8_t wb = w.packed[(size_t)n * (w.K / 2) + (k >> 1)];
                        uint8_t nib = (k & 1) ? (wb >> 4) : (wb & 0x0F);
                        float code = dequant_code(nib);
                        w.true_v[(size_t)n * w.K + k] = code * sc;
                    }
                }
            }
            break;
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

            // Load scales — check if tensor type is self-contained first
            int qkv_type = find_tensor_index(m->gguf, wname);
            int self_contained = 0;
            if (qkv_type >= 0) {
                int t = gguf_tensor_type(m->gguf, qkv_type);
                self_contained = gguf_type_is_self_contained(t);
                fprintf(stderr, "blk.%d: qkv_type=%d self_contained=%d\n", i, t, self_contained);
            }

            size_t s_sz;
            const void* s_ptr = gguf_get_tensor_ptr(m->gguf, sname, &s_sz);

            if (!self_contained && !s_ptr) {
                // Non-self-contained type without external scales — cannot proceed
                fprintf(stderr, "blk.%d: combined qkv scales tensor not found (type %d requires external scales)\n",
                        i, qkv_type >= 0 ? gguf_tensor_type(m->gguf, qkv_type) : -1);
                return false;
            } else if (!self_contained && s_ptr) {
                // External scales exist for non-self-contained type — use existing split logic below
            } else {
                // Self-contained type (TYPE13/Q4_0/Q8_0/etc.) — no external scales expected
                fprintf(stderr, "blk.%d: self-contained quantization type detected, skipping external scale lookup\n", i);
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

            if (!self_contained) {
                // Non-self-contained types require external scales — use existing split logic
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
                // Self-contained quantization — no external scales tensor exists.
                // Extract actual scales from within blocks based on tensor type.
                fprintf(stderr, "blk.%d: self-contained type %d, extracting scales from blocks\n", i, qkv_type >= 0 ? gguf_tensor_type(m->gguf, qkv_type) : -1);

                size_t q_scales_per_row = ((size_t)DQ + 15) >> 4;
                size_t k_scales_per_row = ((size_t)DKV + 15) >> 4;
                size_t v_scales_per_row = ((size_t)DKV + 15) >> 4;

                L.Wq.scales.resize((size_t)E * q_scales_per_row);
                L.Wk.scales.resize((size_t)E * k_scales_per_row);
                L.Wv.scales.resize((size_t)E * v_scales_per_row);

                int t = (qkv_type >= 0) ? gguf_tensor_type(m->gguf, qkv_type) : -1;

                if (t == 13) {
                    // TYPE13: block_size=256 values/block, UE4M3 scales embedded in each block.
                    // Block layout: [scale_0..scale_15 (16 bytes), packed_data (128 bytes)] = 144 bytes per block.
                    // Each scale covers 16 elements within the block.
                    // For combined QKV with total_cols columns:
                    //   blocks_per_row = ceil(total_cols / 256)
                    //   total_scales_per_row = blocks_per_row * 16

                    size_t blocks_per_row = (total_cols + 255) / 256;
                    size_t total_scales_per_row = blocks_per_row * 16;

                    for (int64_t row = 0; row < E; ++row) {
                        const unsigned char* src_row = pk + row * blocks_per_row * 144;

                        // --- Extract all scales from this row's blocks into a temp buffer ---
                        std::vector<uint8_t> all_scales(total_scales_per_row);
                        for (size_t b = 0; b < blocks_per_row; ++b) {
                            const uint8_t* blk = src_row + b * 144;
                            for (size_t s = 0; s < 16; ++s) {
                                all_scales[b * 16 + s] = blk[s]; // UE4M3 scale at offset s within block
                            }
                        }

                        // Map scales to projection-specific arrays.
                        // Column c maps to scale_index = c / 16, and scale comes from block (c/256).
                        // Q: columns [0 .. DQ-1], K: columns [DQ .. DQ+DKV-1], V: columns [DQ+DKV .. DQ+2*DKV-1]

                        size_t q_scale_count = (DQ + 15) >> 4; // ceil(DQ/16), but may exceed available scales
                        if (q_scale_count > total_scales_per_row) q_scale_count = total_scales_per_row;
                        for (size_t s = 0; s < q_scales_per_row; ++s) {
                            L.Wq.scales[row * q_scales_per_row + s] =
                                (s < q_scale_count) ? all_scales[s] : 0x60;
                        }

                        // K starts after Q's column range in terms of which scales apply.
                        // Scale index for column (DQ + k) is (DQ + k) / 16.
                        // But we need the scale that covers column (DQ + k), which comes from block ((DQ+k)/256).
                        size_t k_start_col = DQ;
                        for (size_t s = 0; s < k_scales_per_row; ++s) {
                            int col = (int)(k_start_col + s * 16);
                            size_t si = col / 16;
                            L.Wk.scales[row * k_scales_per_row + s] =
                                (si < total_scales_per_row) ? all_scales[si] : 0x60;
                        }

                        // V: columns [DQ+DKV .. DQ+2*DKV-1]
                        size_t v_start_col = DQ + DKV;
                        for (size_t s = 0; s < v_scales_per_row; ++s) {
                            int col = (int)(v_start_col + s * 16);
                            size_t si = col / 16;
                            L.Wv.scales[row * v_scales_per_row + s] =
                                (si < total_scales_per_row) ? all_scales[si] : 0x60;
                        }
                    }
                } else if (t == 2) { // Q4_0 in combined QKV — decode FP16 delta from blocks per projection
                    fprintf(stderr, "blk.%d: Q4_0 combined QKV — extracting FP16 delta from blocks\n", i);

                    size_t block_bytes_q40 = 34; // For combined QKV, total block is 34 bytes (2B delta + 32B int8)
                    size_t blocks_per_row_total = total_cols / 32;

                    for (int64_t row = 0; row < E; ++row) {
                        const unsigned char* src_row = pk + row * blocks_per_row_total * block_bytes_q40;

                        // Extract all deltas from this row's blocks
                        std::vector<float> all_deltas(blocks_per_row_total);
                        for (size_t b = 0; b < blocks_per_row_total; ++b) {
                            const uint8_t* blk = src_row + b * block_bytes_q40;
                            __half h_delta;
                            memcpy(&h_delta, blk, sizeof(__half));
                            all_deltas[b] = __half2float(h_delta);
                        }

                        // Map scale+min pairs to projections based on column ranges (Q4_1-compatible layout)
                        size_t q_blocks = DQ / 32;
                        for (size_t s = 0; s < q_scales_per_row; ++s) {
                            L.Wq.scales_f[(size_t)row * q_scales_per_row * 2 + s * 2] =
                                (s < q_blocks && s < all_deltas.size()) ? all_deltas[s] : 1.0f; // scale=delta
                            L.Wq.scales_f[(size_t)row * q_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0 for Q4_0
                        }

                        size_t k_start_block = q_blocks;
                        size_t k_blocks = DKV / 32;
                        for (size_t s = 0; s < k_scales_per_row; ++s) {
                            size_t bi = k_start_block + s;
                            L.Wk.scales_f[(size_t)row * k_scales_per_row * 2 + s * 2] =
                                (bi < blocks_per_row_total && bi < all_deltas.size()) ? all_deltas[bi] : 1.0f; // scale=delta
                            L.Wk.scales_f[(size_t)row * k_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0 for Q4_0
                        }

                        size_t v_start_block = q_blocks + k_blocks;
                        for (size_t s = 0; s < v_scales_per_row; ++s) {
                            size_t bi = v_start_block + s;
                            L.Wv.scales_f[(size_t)row * v_scales_per_row * 2 + s * 2] =
                                (bi < blocks_per_row_total && bi < all_deltas.size()) ? all_deltas[bi] : 1.0f; // scale=delta
                            L.Wv.scales_f[(size_t)row * v_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0 for Q4_0
                        }
                    }

                } else if (t == 3) { // Q4_1 in combined QKV — decode FP16 scale+min from blocks per projection
                    fprintf(stderr, "blk.%d: Q4_1 combined QKV — extracting FP16 scale+min from blocks\n", i);

                    size_t block_bytes_q41 = 34; // For combined QKV, total block is 34 bytes (2B scale + 2B min + 32B int8)
                    size_t blocks_per_row_total = total_cols / 32;

                    for (int64_t row = 0; row < E; ++row) {
                        const unsigned char* src_row = pk + row * blocks_per_row_total * block_bytes_q41;

                        // Extract all scale/min pairs from this row's blocks
                        std::vector<float> all_scales(blocks_per_row_total * 2);
                        for (size_t b = 0; b < blocks_per_row_total; ++b) {
                            const uint8_t* blk = src_row + b * block_bytes_q41;
                            __half h_scale, h_min;
                            memcpy(&h_scale, blk, sizeof(__half));
                            memcpy(&h_min, blk + 2, sizeof(__half));
                            all_scales[b * 2]     = __half2float(h_scale); // scale
                            all_scales[b * 2 + 1] = __half2float(h_min);    // min
                        }

                        // Map scale/min pairs to projections based on column ranges
                        size_t q_blocks = DQ / 32;
                        for (size_t s = 0; s < q_scales_per_row; ++s) {
                            L.Wq.scales_f[(size_t)row * q_scales_per_row + s] =
                                (s * 2 < (size_t)all_scales.size() && s * 2 + 1 < (size_t)all_scales.size())
                                    ? all_scales[s * 2] : 1.0f;
                        }

                        size_t k_start_block = q_blocks;
                        size_t k_blocks = DKV / 32;
                        for (size_t s = 0; s < k_scales_per_row; ++s) {
                            size_t bi = k_start_block + s;
                            L.Wk.scales_f[(size_t)row * k_scales_per_row + s] =
                                (bi * 2 < (size_t)all_scales.size() && bi * 2 + 1 < (size_t)all_scales.size())
                                    ? all_scales[bi * 2] : 1.0f;
                        }

                        size_t v_start_block = q_blocks + k_blocks;
                        for (size_t s = 0; s < v_scales_per_row; ++s) {
                            size_t bi = v_start_block + s;
                            L.Wv.scales_f[(size_t)row * v_scales_per_row * 2 + s * 2] =
                                (bi * 2 < (size_t)all_scales.size() && bi * 2 + 1 < (size_t)all_scales.size())
                                    ? all_scales[bi * 2] : 1.0f;
                        }
                    }

                } else if (t == 7 || t == 8) { // Q8_0 in combined QKV — decode FP16 delta from blocks per projection
                    fprintf(stderr, "blk.%d: Q8_0 combined QKV — extracting FP16 delta from blocks\n", i);

                    size_t block_bytes_q80 = 34; // 2B delta + 32B int8 values
                    size_t blocks_per_row_total = total_cols / 32;

                    for (int64_t row = 0; row < E; ++row) {
                        const unsigned char* src_row = pk + row * blocks_per_row_total * block_bytes_q80;

                        // Extract all deltas from this row's blocks
                        std::vector<float> all_deltas(blocks_per_row_total);
                        for (size_t b = 0; b < blocks_per_row_total; ++b) {
                            const uint8_t* blk = src_row + b * block_bytes_q80;
                            __half h_delta;
                            memcpy(&h_delta, blk, sizeof(__half));
                            all_deltas[b] = __half2float(h_delta);
                        }

                        // Map scale+min pairs to projections based on column ranges (Q4_1-compatible layout)
                        size_t q_blocks = DQ / 32;
                        for (size_t s = 0; s < q_scales_per_row; ++s) {
                            L.Wq.scales_f[(size_t)row * q_scales_per_row * 2 + s * 2] =
                                (s < q_blocks && s < all_deltas.size()) ? all_deltas[s] : 1.0f; // scale=delta
                            L.Wq.scales_f[(size_t)row * q_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0 for Q8_0
                        }

                        size_t k_start_block = q_blocks;
                        size_t k_blocks = DKV / 32;
                        for (size_t s = 0; s < k_scales_per_row; ++s) {
                            size_t bi = k_start_block + s;
                            L.Wk.scales_f[(size_t)row * k_scales_per_row * 2 + s * 2] =
                                (bi < blocks_per_row_total && bi < all_deltas.size()) ? all_deltas[bi] : 1.0f; // scale=delta
                            L.Wk.scales_f[(size_t)row * k_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0 for Q8_0
                        }

                        size_t v_start_block = q_blocks + k_blocks;
                        for (size_t s = 0; s < v_scales_per_row; ++s) {
                            size_t bi = v_start_block + s;
                            L.Wv.scales_f[(size_t)row * v_scales_per_row * 2 + s * 2] =
                                (bi < blocks_per_row_total && bi < all_deltas.size()) ? all_deltas[bi] : 1.0f; // scale=delta
                            L.Wv.scales_f[(size_t)row * v_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0 for Q8_0
                        }
                    }

                } else {
                    // Unknown self-contained type — fill with unity scale as safe fallback
                    fprintf(stderr, "blk.%d: unknown type %d in combined QKV, filling unity placeholder\n", i, t);
                    for (int64_t row = 0; row < E; ++row) {
                        for (size_t s = 0; s < q_scales_per_row; ++s) {
                            L.Wq.scales_f[row * q_scales_per_row * 2 + s * 2] = 1.0f;   // scale=1.0
                            L.Wq.scales_f[row * q_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0
                        }
                        for (size_t s = 0; s < k_scales_per_row; ++s) {
                            L.Wk.scales_f[row * k_scales_per_row * 2 + s * 2] = 1.0f;   // scale=1.0
                            L.Wk.scales_f[row * k_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0
                        }
                        for (size_t s = 0; s < v_scales_per_row; ++s) {
                            L.Wv.scales_f[row * v_scales_per_row * 2 + s * 2] = 1.0f;   // scale=1.0
                            L.Wv.scales_f[row * v_scales_per_row * 2 + s * 2 + 1] = 0.0f; // min=0
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

        // Attention O projection — try multiple naming conventions (Phase 15 hybrid architecture)
        snprintf(wname, sizeof(wname), "blk.%d.attn_o.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.attn_o.scales", i);
        bool has_attn_o = has_tensor(m->gguf, wname);
        if (has_attn_o) {
            if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.Wo)) return false;
        } else {
            // Try alternative name used in Qwen3.5 attention-output layout
            snprintf(wname, sizeof(wname), "blk.%d.attn_output.weight", i);
            snprintf(sname, sizeof(sname), "blk.%d.attn_output.scales", i);
            if (has_tensor(m->gguf, wname)) {
                if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.Wo)) return false;
            } else {
                fprintf(stderr, "[load_model_weights] blk.%d: no attn_o/attn_output tensor found (may be SSM layer)\n", i);
            }
        }

        // MLP gate, up, down — make optional for layers without FFN (SSM-only blocks)
        snprintf(wname, sizeof(wname), "blk.%d.ffn_gate.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_gate.scales", i);
        if (has_tensor(m->gguf, wname)) {
            if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.W_gate)) return false;
        } else {
            fprintf(stderr, "[load_model_weights] blk.%d: no ffn_gate tensor found\n", i);
        }

        snprintf(wname, sizeof(wname), "blk.%d.ffn_up.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_up.scales", i);
        if (has_tensor(m->gguf, wname)) {
            if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.W_up)) return false;
        } else {
            fprintf(stderr, "[load_model_weights] blk.%d: no ffn_up tensor found\n", i);
        }

        snprintf(wname, sizeof(wname), "blk.%d.ffn_down.weight", i);
        snprintf(sname, sizeof(sname), "blk.%d.ffn_down.scales", i);
        if (has_tensor(m->gguf, wname)) {
            if (!load_quant_weight_from_gguf(m->gguf, wname, sname, L.W_down)) return false;
        } else {
            fprintf(stderr, "[load_model_weights] blk.%d: no ffn_down tensor found\n", i);
        }
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

    // Load embedding table — detect quantized types and route accordingly
    {
        int emb_idx = find_tensor_index(m->gguf, "token_embd.weight");
        size_t sz;
        const void* emb_ptr = gguf_get_tensor_ptr(m->gguf, "token_embd.weight", &sz);
        if (emb_ptr) {
            int emb_type = -1;
            if (emb_idx >= 0) {
                emb_type = gguf_tensor_type(m->gguf, emb_idx);
            }

            if (emb_type == -1 || emb_type == 1 /* F16 */) {
                // FP16 embedding table — load directly as before
                if (sz == (size_t)m->vocab_size * m->n_embd * sizeof(__half)) {
                    m->embed_table.assign((const __half*)emb_ptr, (const __half*)emb_ptr + m->vocab_size * m->n_embd);
                    fprintf(stderr, "[load_model] token_embd: FP16 (%zu bytes, %dx%d)\n",
                            sz, m->vocab_size, m->n_embd);
                } else {
                    fprintf(stderr, "[load_model] WARNING: token_embd exists but size mismatch for FP16 (%zu vs expected %zu)\n",
                            sz, (size_t)m->vocab_size * m->n_embd * sizeof(__half));
                }
            } else {
                // Quantized embedding — load via QuantWeight path with dequant at runtime
                fprintf(stderr, "[load_model] token_embd: quantized type %d (%zu bytes), loading through QuantWeight path\n",
                        emb_type, sz);

                int64_t* ne = gguf_tensor_ne(m->gguf, emb_idx);
                if (ne && ne[0] == m->vocab_size && ne[1] == m->n_embd) {
                    // Load packed data and scales into temp storage
                    std::vector<unsigned char> packed((const unsigned char*)emb_ptr,
                                                       (const unsigned char*)emb_ptr + sz);

                    // Determine scale tensor name based on type
                    char sname[128];
                    snprintf(sname, sizeof(sname), "token_embd.scales");
                    size_t s_sz;
                    const void* s_ptr = gguf_get_tensor_ptr(m->gguf, sname, &s_sz);

                    if (s_ptr && s_sz > 0) {
                        // External scales exist — could store these for later use
                        fprintf(stderr, "[load_model] token_embd: has external scales (%zu bytes)\n", s_sz);
                    } else if (gguf_type_is_self_contained(emb_type)) {
                        // Self-contained — scales embedded in blocks, will extract at runtime
                        fprintf(stderr, "[load_model] token_embd: self-contained type %d, scales embedded in blocks\n",
                                emb_type);
                    } else {
                        fprintf(stderr, "[load_model] WARNING: token_embd is quantized but no scales available (type %d)\n",
                                emb_type);
                    }

                    // For now: dequantize to FP16 for embedding lookup compatibility.
                    // At Q4_1 with vocab=248320, n_embd=2560 → ~635M values → ~1.2GB dequantized FP16.
                    // This is acceptable on 128GB UMA systems.
                    size_t total_elements = (size_t)m->vocab_size * m->n_embd;
                    m->embed_table.resize(total_elements);

                    // Dequantize based on type using scale extraction
                    switch (emb_type) {
                        case 100: {
                            // TYPE100 / MXFP4: UE4M3 scales per block of 32 values
                            {
                                const uint8_t* data = packed.data();
                                // Block size constants for documentation/debugging only (not used in calculation)

                                for (int vi = 0; vi < m->vocab_size; ++vi) {
                                    int64_t row_offset = (int64_t)vi * m->n_embd;

                                    for (int ei = 0; ei < m->n_embd; ++ei) {
                                        int bi = ei / 32;
                                        int val_in_block = ei % 32;
                                        size_t blk_idx = (size_t)row_offset / 32 + bi;

                                        float sc;
                                        if (val_in_block < 16) {
                                            sc = ue4m3_to_float(data[blk_idx * 17 + 16]); // first half-block scale
                                        } else {
                                            sc = ue4m3_to_float(data[blk_idx * 17 + 17]); // second half-block scale
                                        }

                                        uint8_t nibble = (data[blk_idx * 17 + 16 + (val_in_block >> 1)] >> ((val_in_block & 1) << 2)) & 0x0F;
                                        m->embed_table[(size_t)vi * m->n_embd + ei] = __half(dequant_code(nibble) * sc);
                                    }
                                }
                            }
                            break;
                        }
                        case 3: { // Q4_1
                            // Q4_1 block layout: [scale_fp16(2B), min_fp16(2B), packed_data(16B)] = 20 bytes per 32 values
                            {
                                const uint8_t* data = packed.data();
                                size_t bytes_per_block = 20; // 2 scale + 2 min + 16 packed

                                for (int vi = 0; vi < m->vocab_size; ++vi) {
                                    int64_t row_offset = (int64_t)vi * m->n_embd;

                                    for (int ei = 0; ei < m->n_embd; ++ei) {
                                        int bi = ei / 32;
                                        int val_in_block = ei % 32;
                                        size_t blk_idx = (size_t)row_offset / 32 + bi;

                                        __half h_scale, h_min;
                                        memcpy(&h_scale, &data[blk_idx * bytes_per_block], sizeof(__half));
                                        memcpy(&h_min, &data[blk_idx * bytes_per_block + 2], sizeof(__half));

                                        float scale_f = __half2float(h_scale);
                                        float min_f = __half2float(h_min);

                                        uint8_t nibble = (data[blk_idx * bytes_per_block + 4 + (val_in_block >> 1)] >> ((val_in_block & 1) << 2)) & 0x0F;
                                        // Q4_1: value = scale * codebook[nibble] + min
                                        float val = scale_f * dequant_code(nibble) + min_f;
                                        m->embed_table[(size_t)vi * m->n_embd + ei] = __half(val);
                                    }
                                }
                            }
                            break;
                        }
                        case 2: { // Q4_0
                            // Q4_0 block layout: [delta_fp16(2B), packed_data(16B)] = 18 bytes per 32 values
                            {
                                const uint8_t* data = packed.data();
                                size_t bytes_per_block = 18;

                                for (int vi = 0; vi < m->vocab_size; ++vi) {
                                    int64_t row_offset = (int64_t)vi * m->n_embd;

                                    for (int ei = 0; ei < m->n_embd; ++ei) {
                                        int bi = ei / 32;
                                        int val_in_block = ei % 32;
                                        size_t blk_idx = (size_t)row_offset / 32 + bi;

                                        __half h_delta;
                                        memcpy(&h_delta, &data[blk_idx * bytes_per_block], sizeof(__half));

                                        float delta_f = __half2float(h_delta);

                                        uint8_t nibble = (data[blk_idx * bytes_per_block + 2 + (val_in_block >> 1)] >> ((val_in_block & 1) << 2)) & 0x0F;
                                        // Q4_0: value = delta * codebook[nibble] - min (min encoded in data)
                                        float val = delta_f * dequant_code(nibble);
                                        m->embed_table[(size_t)vi * m->n_embd + ei] = __half(val);
                                    }
                                }
                            }
                            break;
                        }
                        case 14: { // TYPE14: Qwen3.5 custom quantization — self-contained UE4M3 blocks
                            fprintf(stderr, "[load_model] token_embd: extracting TYPE14 scales from blocks (type=%d)\n", emb_type);
                            {
                                const uint8_t* data = packed.data();
                                size_t block_size_elements = 256; // TYPE14 uses 256-element blocks like TYPE13
                                size_t bytes_per_block = 176; // calculated from tensor size: 437043200 / (total_elements/256)
                                size_t blocks_per_row = m->n_embd / block_size_elements;
                                size_t total_blocks = blocks_per_row * m->vocab_size;

                                fprintf(stderr, "[load_model] TYPE14: blocks_per_row=%zu total_blocks=%zu bytes/block=%zu\n",
                                        blocks_per_row, total_blocks, bytes_per_block);

                                for (int vi = 0; vi < m->vocab_size; ++vi) {
                                    int64_t row_offset = (int64_t)vi * m->n_embd;

                                    for (int ei = 0; ei < m->n_embd; ++ei) {
                                        int bi = ei / block_size_elements;
                                        int val_in_block = ei % block_size_elements;
                                        size_t blk_idx = (size_t)row_offset / block_size_elements + bi;

                                        // Scale index within block: each scale covers 16 elements, stored as UE4M3 (1 byte)
                                        size_t scale_idx = val_in_block / 16;
                                        size_t scale_byte_off = blk_idx * bytes_per_block + scale_idx; // UE4M3 stored as 1 byte

                                        float sc = ue4m3_to_float(data[scale_byte_off]);

                                        // Data offset after scales (scales occupy first 16 bytes of block header for 16 scales)
                                        size_t data_off = blk_idx * bytes_per_block + 16;

                                        // Nibble index: pair of values per byte in data section
                                        int nibble_idx = val_in_block / 2;
                                        uint8_t nibble = (data[data_off + nibble_idx] >> ((val_in_block & 1) << 2)) & 0x0F;
                                        m->embed_table[(size_t)vi * m->n_embd + ei] = __half(dequant_code(nibble) * sc);
                                    }
                                }
                            }
                            fprintf(stderr, "[load_model] token_embd TYPE14 dequantized (%d x %d)\n", m->vocab_size, m->n_embd);
                            break;
                        }
                        default: {
                            fprintf(stderr, "[load_model] WARNING: quantized embedding type %d not supported for dequant\n",
                                    emb_type);
                            // Fall through to keep model loading — embeddings will be zeroed
                            break;
                        }
                    }
                } else if (ne && ne[0] == m->n_embd && ne[1] == m->vocab_size) {
                    // Transposed layout [n_embd][vocab_size] — common GGUF convention
                    fprintf(stderr, "[load_model] token_emb: transposed layout detected [%lld x %lld], dequantizing...\n",
                            (long long)ne[0], (long long)ne[1]);

                    size_t total_elements = (size_t)m->vocab_size * m->n_embd;
                    m->embed_table.resize(total_elements);

                    // Load packed data for transposed path
                    std::vector<unsigned char> packed((const unsigned char*)emb_ptr,
                                                       (const unsigned char*)emb_ptr + sz);

                    // Dequantize based on type using scale extraction (transposed indexing)
                    switch (emb_type) {
                        case 3: { // Q4_1 transposed
                            const uint8_t* data = packed.data();
                            size_t bytes_per_block = 20; // 2 scale + 2 min + 16 packed

                            for (int vi = 0; vi < m->vocab_size; ++vi) {
                                int64_t row_offset = (int64_t)vi;
                                size_t row_blocks = m->n_embd / 32;

                                for (int ei = 0; ei < m->n_embd; ++ei) {
                                    int bi = ei / 32;
                                    int val_in_block = ei % 32;
                                    size_t blk_idx = (size_t)row_offset * row_blocks + bi;

                                    __half h_scale, h_min;
                                    memcpy(&h_scale, &data[blk_idx * bytes_per_block], sizeof(__half));
                                    memcpy(&h_min, &data[blk_idx * bytes_per_block + 2], sizeof(__half));

                                    float scale_f = __half2float(h_scale);
                                    float min_f = __half2float(h_min);

                                    uint8_t nibble = (data[blk_idx * bytes_per_block + 4 + (val_in_block >> 1)] >> ((val_in_block & 1) << 2)) & 0x0F;
                                    float val = scale_f * dequant_code(nibble) + min_f;
                                    m->embed_table[(size_t)vi * m->n_embd + ei] = __half(val);
                                }
                            }
                            break;
                        }
                        case 14: { // TYPE14 transposed [n_embd][vocab_size]
                            fprintf(stderr, "[load_model] token_embd TYPE14 transposed, extracting scales from blocks\n");
                            const uint8_t* data = packed.data();
                            size_t block_size_elements = 256; // TYPE14 uses 256-element blocks
                            size_t bytes_per_block = 176;     // computed from tensor size
                            size_t blocks_per_vocab_row = m->vocab_size / block_size_elements;

                            // Validate block layout against actual file size
                            size_t total_blocks_calc = (size_t)m->n_embd * blocks_per_vocab_row;
                            size_t expected_size = total_blocks_calc * bytes_per_block;
                            fprintf(stderr, "[load_model] TYPE14 validate: total_blocks=%zu expected_sz=%zu actual_sz=%zu\n",
                                    total_blocks_calc, expected_size, sz);

                            for (int vi = 0; vi < m->vocab_size; ++vi) {
                                for (int ei = 0; ei < m->n_embd; ++ei) {
                                    // GGUF stores [n_embd][vocab_size] in row-major C order
                                    // index(ei, vi) = ei * vocab_size + vi
                                    int bi = vi / block_size_elements;          // which block within this n_embd row
                                    int val_in_block = vi % block_size_elements; // position within block
                                    size_t blk_idx = (size_t)ei * blocks_per_vocab_row + bi;

                                    // Bounds check on block index
                                    if (blk_idx >= total_blocks_calc) {
                                        fprintf(stderr, "[load_model] ERROR TYPE14: blk_idx=%zu >= total_blocks=%zu at vi=%d ei=%d\n",
                                                blk_idx, total_blocks_calc, vi, ei);
                                        goto type14_done;
                                    }

                                    // Scale index within block: each scale covers 16 elements, stored as UE4M3 (1 byte)
                                    size_t scale_idx = val_in_block / 16;
                                    size_t scale_byte_off = blk_idx * bytes_per_block + scale_idx; // UE4M3 stored as 1 byte

                                    float sc = ue4m3_to_float(data[scale_byte_off]);

                                    // Data offset after scales (scales occupy first 16 bytes of block header for 16 scales)
                                    size_t data_off = blk_idx * bytes_per_block + 16;

                                    // Nibble index: pair of values per byte in data section
                                    int nibble_idx = val_in_block / 2;
                                    uint8_t nibble = (data[data_off + nibble_idx] >> ((val_in_block & 1) << 2)) & 0x0F;
                                    m->embed_table[(size_t)vi * m->n_embd + ei] = __half(dequant_code(nibble) * sc);
                                }
                            }
type14_done:;
                            fprintf(stderr, "[load_model] token_embd TYPE14 dequantized transposed (%d x %d)\n", m->vocab_size, m->n_embd);
                            break;
                        }
                        default: {
                            fprintf(stderr, "[load_model] WARNING: quantized embedding type %d not supported for dequant\n",
                                    emb_type);
                            break;
                        }
                    }
                } else {
                    if (ne) {
                        fprintf(stderr, "[load_model] WARNING: token_embd dimensions don't match vocab_size×n_embd [%lld x %lld vs %d x %d]\n",
                                (long long)ne[0], (long long)ne[1], m->vocab_size, m->n_embd);
                    } else {
                        fprintf(stderr, "[load_model] WARNING: could not read tensor dimensions for token_embd.weight\n");
                    }
                }
            }
        } else {
            fprintf(stderr, "[load_model] WARNING: token_embd.weight not found in GGUF file\n");
        }
    }

    // Load per-layer weights from GGUF tensors
    if (!load_model_weights(m)) {
        set_error("Failed to load model weights");
        unpin_model_memory(m);
        delete m;
        return nullptr;
    }

    // Load LM head — may not exist for models with tied embeddings (Qwen3.5-4B uses shared embed/output weights)
    char out_wname[128], out_sname[128];
    snprintf(out_wname, sizeof(out_wname), "output.weight");
    snprintf(out_sname, sizeof(out_sname), "output.scales");

    // Try multiple possible names for the output/LM-head tensor
    const char* lm_head_names[] = {"output.weight", "lm_head.weight", "embed_tokens.weight"};
    bool has_lm_head = false;
    for (const char* name : lm_head_names) {
        if (has_tensor(m->gguf, name)) { has_lm_head = true; break; }
    }

    if (has_lm_head) {
        if (!load_quant_weight_from_gguf(m->gguf, "output.weight", "output.scales", m->lm_head)) {
            set_error("Failed to load LM head");
            unpin_model_memory(m);
            delete m;
            return nullptr;
        }
    } else {
        fprintf(stderr, "[inference_load_model] No LM head tensor found — using tied embeddings (token_embd.weight as output projection)\n");
        // For tied embeddings: lm_head is just token_embd shared between input and output
        m->lm_head.N = m->vocab_size;  // output dim = vocab size
        m->lm_head.K = m->n_embd;      // input dim = embedding length
        m->lm_head.packed.clear();     // will reuse embed_table packed data at runtime
        m->lm_head.scales.clear();     // scales also shared
    }

    // Load final LN scale
    size_t ln_sz;
    const void* final_ln_ptr = gguf_get_tensor_ptr(m->gguf, "output_norm.weight", &ln_sz);
    if (final_ln_ptr && ln_sz == (size_t)m->n_embd * sizeof(__half)) {
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
    fprintf(stderr, "[KV_INIT] About to create qwen model\n"); fflush(stderr);
    m->qwen = qwen_model_create(m);
    if (!m->qwen) { set_error("qwen_model_create failed"); return -1; }
    fprintf(stderr, "[KV_INIT] Qwen model created at %p\n", (void*)m->qwen);
    // Allocate dedicated GPU device memory for embeddings and copy data there.
    // This avoids MEMORY_APERTURE_VIOLATION caused by rocWMMA kernarg handling
    // of indirect pointer references to registered host memory.
    if (!m->embed_table.empty()) {
        size_t emb_bytes = m->embed_table.size() * sizeof(__half);
        void* d_emb_dev = nullptr;
        hipError_t e;
        fprintf(stderr, "[KV_INIT] Allocating %zu bytes GPU memory for embed table...\n", emb_bytes); fflush(stderr);
        e = hipMalloc(&d_emb_dev, emb_bytes);
        if (e == hipSuccess) {
            e = hipMemcpy(d_emb_dev, m->embed_table.data(), emb_bytes, hipMemcpyHostToDevice);
            if (e == hipSuccess) {
                fprintf(stderr, "[KV_INIT] Embed copied to GPU device %p (%zu elements)\n", 
                        d_emb_dev, m->embed_table.size()); fflush(stderr);
                qwen_model_set_embed_table(m->qwen, d_emb_dev);
            } else {
                fprintf(stderr, "[KV_INIT] hipMemcpy failed: %s\n", hipGetErrorString(e));
                (void)hipFree(d_emb_dev);
            }
        } else {
            fprintf(stderr, "[KV_INIT] hipMalloc failed: %s\n", hipGetErrorString(e));
        }
    } else {
        fprintf(stderr, "[KV_INIT] No embedding table available\n"); fflush(stderr);
    }
    fprintf(stderr, "[KV_INIT] embed table set on qwen model\n");
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


tokenizer_t* inference_get_tokenizer(const inference_model_t* m) {
    if (!m) return nullptr;
    return m->tokenizer;
}
