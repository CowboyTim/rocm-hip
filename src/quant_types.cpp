// quant_types.cpp — CPU decoders for GGUF weight formats. See quant_types.h.
//
// Block layouts and arithmetic follow ggml's reference dequantizers (MIT) and ROCmFPX's
// ggml/rocmfp4/rocmfp4.c, re-implemented here so the engine never links ggml.
// The float expression order matches the reference so results are bit-exact.

#include "quant_types.h"

#include <string.h>

// ---------------------------------------------------------------- scalars

float qt_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // subnormal: renormalize
            int e = -1;
            do {
                mant <<= 1;
                ++e;
            } while (!(mant & 0x400));
            bits = sign | (uint32_t)(127 - 15 - e) << 23 | (mant & 0x3ff) << 13;
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | mant << 13;
    } else {
        bits = sign | (exp + 127 - 15) << 23 | mant << 13;
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

float qt_bf16_to_fp32(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline float rd_fp16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return qt_fp16_to_fp32(h);
}

// ---------------------------------------------------------------- plain

static void deq_f32(const void *src, float *y, int64_t n) {
    memcpy(y, src, (size_t)n * sizeof(float));
}

static void deq_f16(const void *src, float *y, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int64_t i = 0; i < n; ++i) y[i] = rd_fp16(p + 2 * i);
}

static void deq_bf16(const void *src, float *y, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    for (int64_t i = 0; i < n; ++i) {
        uint16_t h;
        memcpy(&h, p + 2 * i, 2);
        y[i] = qt_bf16_to_fp32(h);
    }
}

// ---------------------------------------------------------------- 32-value legacy blocks

// Q4_0: d(f16) qs[16]; value j from the low nibble, j+16 from the high nibble
static void deq_q4_0(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 18, y += 32) {
        const float d = rd_fp16(b);
        const uint8_t *qs = b + 2;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (qs[j] & 0x0F) - 8;
            const int x1 = (qs[j] >> 4) - 8;
            y[j]      = x0 * d;
            y[j + 16] = x1 * d;
        }
    }
}

// Q4_1: d(f16) m(f16) qs[16]
static void deq_q4_1(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 20, y += 32) {
        const float d = rd_fp16(b);
        const float m = rd_fp16(b + 2);
        const uint8_t *qs = b + 4;
        for (int j = 0; j < 16; ++j) {
            const int x0 = qs[j] & 0x0F;
            const int x1 = qs[j] >> 4;
            y[j]      = x0 * d + m;
            y[j + 16] = x1 * d + m;
        }
    }
}

// Q5_0: d(f16) qh[4] qs[16]
static void deq_q5_0(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 22, y += 32) {
        const float d = rd_fp16(b);
        uint32_t qh;
        memcpy(&qh, b + 2, 4);
        const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const int32_t x0 = ((qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((qs[j] >> 4) | xh_1) - 16;
            y[j]      = x0 * d;
            y[j + 16] = x1 * d;
        }
    }
}

// Q5_1: d(f16) m(f16) qh[4] qs[16]
static void deq_q5_1(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 24, y += 32) {
        const float d = rd_fp16(b);
        const float m = rd_fp16(b + 2);
        uint32_t qh;
        memcpy(&qh, b + 4, 4);
        const uint8_t *qs = b + 8;
        for (int j = 0; j < 16; ++j) {
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const int x0 = (qs[j] & 0x0F) | xh_0;
            const int x1 = (qs[j] >> 4) | xh_1;
            y[j]      = x0 * d + m;
            y[j + 16] = x1 * d + m;
        }
    }
}

// Q8_0: d(f16) qs[32] int8
static void deq_q8_0(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 34, y += 32) {
        const float d = rd_fp16(b);
        const int8_t *qs = (const int8_t *)(b + 2);
        for (int j = 0; j < 32; ++j) y[j] = qs[j] * d;
    }
}

// ---------------------------------------------------------------- 256-value K-quants

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Q2_K: scales[16] qs[64] d(f16) dmin(f16) = 84 B
static void deq_q2_K(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 84) {
        const uint8_t *scales = b;
        const uint8_t *q = b + 16;
        const float d   = rd_fp16(b + 80);
        const float min = rd_fp16(b + 82);
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF);
                ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

// Q3_K: hmask[32] qs[64] scales[12] d(f16) = 110 B
static void deq_q3_K(const void *src, float *y, int64_t n) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    const uint8_t *b = (const uint8_t *)src;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;
    for (int64_t i = 0; i < n / 256; ++i, b += 110) {
        const uint8_t *hm = b;
        const uint8_t *q = b + 32;
        const float d_all = rd_fp16(b + 108);
        memcpy(aux, b + 96, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        uint8_t m = 1;
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// Q4_K: d(f16) dmin(f16) scales[12] qs[128] = 144 B
static void deq_q4_K(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 144) {
        const float d   = rd_fp16(b);
        const float min = rd_fp16(b + 2);
        const uint8_t *scales = b + 4;
        const uint8_t *q = b + 16;
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

// Q5_K: d(f16) dmin(f16) scales[12] qh[32] qs[128] = 176 B
static void deq_q5_K(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 176) {
        const float d   = rd_fp16(b);
        const float min = rd_fp16(b + 2);
        const uint8_t *scales = b + 4;
        const uint8_t *qh = b + 16;
        const uint8_t *ql = b + 48;
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// Q6_K: ql[128] qh[64] scales[16](int8) d(f16) = 210 B
static void deq_q6_K(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 210) {
        const float d = rd_fp16(b + 208);
        const uint8_t *ql = b;
        const uint8_t *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        for (int nn = 0; nn < 256; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ---------------------------------------------------------------- ROCmFP4

// Signed FP4 codebook at half scale (E2M1-derived, max magnitude 10)
static const int8_t rocmfp4_codebook[16] = {0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};

// Unsigned E4M3 scale byte -> half-scale float. 0x7f and above decode to 0.
static inline float ue4m3_half(uint8_t e) {
    if (e > 0x7e) return 0.0f;
    const int ex = e >> 3, m = e & 7;
    // subnormal m * 2^-10; normal (8 + m) * 2^(ex - 11). Both are exact in float.
    return ex == 0 ? (float)m * 0x1p-10f : (float)(8 + m) * __builtin_ldexpf(1.0f, ex - 11);
}

// Q4_0_ROCMFP4: qs[16] e[2] = 18 B; values 0..15 = low nibbles * e[0], 16..31 = high nibbles * e[1]
static void deq_rocmfp4(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 18, y += 32) {
        const float d0 = ue4m3_half(b[16]);
        const float d1 = ue4m3_half(b[17]);
        for (int j = 0; j < 16; ++j) {
            y[j]      = (float)rocmfp4_codebook[b[j] & 0x0f] * d0;
            y[j + 16] = (float)rocmfp4_codebook[b[j] >> 4] * d1;
        }
    }
}

// Q4_0_ROCMFP4_FAST: qs[16] e = 17 B; one scale for the block
static void deq_rocmfp4_fast(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 17, y += 32) {
        const float d = ue4m3_half(b[16]);
        for (int j = 0; j < 16; ++j) {
            y[j]      = (float)rocmfp4_codebook[b[j] & 0x0f] * d;
            y[j + 16] = (float)rocmfp4_codebook[b[j] >> 4] * d;
        }
    }
}

// ---------------------------------------------------------------- table

// verified: 1 = bit-exact vs `ref_dump dequant` on a real GGUF tensor (tests/dequant_verify)
static const qt_traits k_traits[] = {
    {QT_F32,               "f32",               1,   4,   deq_f32,          1},
    {QT_F16,               "f16",               1,   2,   deq_f16,          0},
    {QT_Q4_0,              "q4_0",              32,  18,  deq_q4_0,         1},
    {QT_Q4_1,              "q4_1",              32,  20,  deq_q4_1,         1},
    {QT_Q5_0,              "q5_0",              32,  22,  deq_q5_0,         0},
    {QT_Q5_1,              "q5_1",              32,  24,  deq_q5_1,         0},
    {QT_Q8_0,              "q8_0",              32,  34,  deq_q8_0,         1},
    {QT_Q2_K,              "q2_K",              256, 84,  deq_q2_K,         1},
    {QT_Q3_K,              "q3_K",              256, 110, deq_q3_K,         1},
    {QT_Q4_K,              "q4_K",              256, 144, deq_q4_K,         1},
    {QT_Q5_K,              "q5_K",              256, 176, deq_q5_K,         1},
    {QT_Q6_K,              "q6_K",              256, 210, deq_q6_K,         1},
    {QT_BF16,              "bf16",              1,   2,   deq_bf16,         1},
    {QT_Q4_0_ROCMFP4,      "q4_0_rocmfp4",      32,  18,  deq_rocmfp4,      1},
    {QT_Q4_0_ROCMFP4_FAST, "q4_0_rocmfp4_fast", 32,  17,  deq_rocmfp4_fast, 1},
};

const qt_traits *qt_get_traits(int type) {
    for (size_t i = 0; i < sizeof(k_traits) / sizeof(k_traits[0]); ++i) {
        if (k_traits[i].type == type) return &k_traits[i];
    }
    return NULL;
}

size_t qt_row_size(int type, int64_t ne0) {
    const qt_traits *t = qt_get_traits(type);
    if (!t || ne0 % t->blck_size != 0) return 0;
    return (size_t)(ne0 / t->blck_size) * (size_t)t->type_size;
}

int qt_to_float(int type, const void *src, float *dst, int64_t n) {
    const qt_traits *t = qt_get_traits(type);
    if (!t || n % t->blck_size != 0) return -1;
    t->to_float(src, dst, n);
    return 0;
}
