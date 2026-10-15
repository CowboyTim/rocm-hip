// quant_types.h — per-type GGUF weight decoding (Phase 20 Step 3).
//
// One table entry per ggml_type id: block size (values), block bytes, and a CPU to_float.
// Every format the engine reads goes through this table; no per-call-site block arithmetic.
// Decoders are standalone ports (no ggml/llama.cpp link) and are checked bit-exact against
// the ROCmFPX reference with tests/dequant_verify (see tools/ref_oracle).

#ifndef QUANT_TYPES_H
#define QUANT_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ggml_type ids (values from ggml.h; 100+ are ROCmFPX-only)
enum qt_type {
    QT_F32  = 0,
    QT_F16  = 1,
    QT_Q4_0 = 2,
    QT_Q4_1 = 3,
    QT_Q5_0 = 6,
    QT_Q5_1 = 7,
    QT_Q8_0 = 8,
    QT_Q4_K = 12,
    QT_Q5_K = 13,
    QT_Q6_K = 14,
    QT_BF16 = 30,
    QT_Q4_0_ROCMFP4      = 100,
    QT_Q4_0_ROCMFP4_FAST = 101,
};

typedef void (*qt_to_float_fn)(const void *src, float *dst, int64_t n);

typedef struct {
    int            type;        // ggml_type id
    const char    *name;        // ggml type name
    int            blck_size;   // values per block
    int            type_size;   // bytes per block
    qt_to_float_fn to_float;    // n must be a multiple of blck_size
    int            verified;    // 1 = bit-exact vs reference on a real tensor
} qt_traits;

// Traits for a ggml_type id, or NULL when the engine cannot decode it yet.
const qt_traits *qt_get_traits(int type);

// Bytes for one row of ne0 values; 0 if unsupported or ne0 is not a multiple of the block size.
size_t qt_row_size(int type, int64_t ne0);

// Decode n values; returns 0 on success, -1 if the type is unsupported or n is misaligned.
int qt_to_float(int type, const void *src, float *dst, int64_t n);

// Scalar helpers shared with other host code
float qt_fp16_to_fp32(uint16_t h);
float qt_bf16_to_fp32(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif // QUANT_TYPES_H
