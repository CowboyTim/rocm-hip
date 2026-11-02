// quant_types.cpp — CPU decoders for GGUF weight formats. See quant_types.h.
//
// Block layouts and arithmetic follow ggml's reference dequantizers (MIT) and ROCmFPX's
// ggml/rocmfp4/rocmfp4.c, re-implemented here so the engine never links ggml.
// The float expression order matches the reference so results are bit-exact.

#include "quant_types.h"

#include <string.h>

#define IQ_TABLE(type, name, size) static const type name[size]
#include "iq_grids.h"

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

// IQ4_NL: d(f16) qs[16], Q4_0's layout; a nibble indexes ggml's kvalues_iq4nl
static const int8_t iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static void deq_iq4_nl(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 18, y += 32) {
        const float d = rd_fp16(b);
        const uint8_t *qs = b + 2;
        for (int j = 0; j < 16; ++j) {
            y[j]      = d * iq4nl_values[qs[j] & 0x0F];
            y[j + 16] = d * iq4nl_values[qs[j] >> 4];
        }
    }
}

// IQ4_XS: d(f16) scales_h(u16) scales_l[4] qs[128] = 136 B per 256 values; sub-block ib (32 values) has the 6-bit
// scale ls = scales_l nibble ib | 2 bits ib of scales_h << 4, d * (ls - 32), and IQ4_NL's 16 bytes at qs + 16 ib
static void deq_iq4_xs(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 136) {
        const float d = rd_fp16(b);
        const int sh = b[2] | (b[3] << 8);
        for (int ib = 0; ib < 8; ++ib, y += 32) {
            const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            const float dl = d * (ls - 32);
            const uint8_t *qs = b + 8 + 16 * ib;
            for (int j = 0; j < 16; ++j) {
                y[j]      = dl * iq4nl_values[qs[j] & 0x0F];
                y[j + 16] = dl * iq4nl_values[qs[j] >> 4];
            }
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

// ---------------------------------------------------------------- MXFP4 / NVFP4

// E2M1 at half scale (ggml's kvalues_mxfp4)
static const int8_t mxfp4_codebook[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// E8M0 scale byte -> half-scale float 2^(e - 128), as ggml's GGML_E8M0_TO_FP32_HALF
static inline float e8m0_half(uint8_t e) {
    const uint32_t bits = e < 2 ? 0x00200000u << e : (uint32_t)(e - 1) << 23;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// MXFP4: e, qs[16] = 17 B; values 0..15 = low nibbles, 16..31 = high nibbles
static void deq_mxfp4(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 32; ++i, b += 17, y += 32) {
        const float d = e8m0_half(b[0]);
        for (int j = 0; j < 16; ++j) {
            y[j]      = (float)mxfp4_codebook[b[1 + j] & 0x0f] * d;
            y[j + 16] = (float)mxfp4_codebook[b[1 + j] >> 4] * d;
        }
    }
}

// NVFP4: d[4] (UE4M3, the sign bit ignored as ggml's ggml_ue4m3_to_fp32), qs[32] = 36 B per 64 values; sub-block s
// (16 values) has scale d[s] and bytes 8s .. 8s + 7: value j in the low nibble of byte j, j + 8 in the high one
static void deq_nvfp4(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 64; ++i, b += 36) {
        for (int s = 0; s < 4; ++s, y += 16) {
            const float d = ue4m3_half(b[s] & 0x7f);
            const uint8_t *qs = b + 4 + 8 * s;
            for (int j = 0; j < 8; ++j) {
                y[j]     = (float)mxfp4_codebook[qs[j] & 0x0f] * d;
                y[j + 8] = (float)mxfp4_codebook[qs[j] >> 4] * d;
            }
        }
    }
}

// ---------------------------------------------------------------- IQ1 / IQ2 / IQ3 and ternary (as ggml's dequantize_row_*)

static inline uint16_t rd_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline uint32_t rd_u32(const uint8_t *p) { return (uint32_t)p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }

// 8 grid bytes (a uint64 entry) times signs bit j of sgn
static void iq_grid8(float *y, float s, uint64_t g, unsigned sgn) {
    for (int j = 0; j < 8; ++j) y[j] = s * (float)(uint8_t)(g >> (8 * j)) * ((sgn >> j) & 1 ? -1.0f : 1.0f);
}

// IQ2_XXS: d, qs[32] u16; per 32 values two u32: 4 grid bytes, then 4 x 7 sign bits and a 4-bit scale
static void deq_iq2_xxs(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 66) {
        const float d = rd_fp16(b);
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t *q = b + 2 + 8 * ib;
            const uint32_t a1 = rd_u32(q + 4);
            const float db = d * (0.5f + (a1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l, y += 8) iq_grid8(y, db, iq2xxs_grid[q[l]], ksigns_iq2xs[(a1 >> (7 * l)) & 127]);
        }
    }
}

// IQ2_XS: d, qs[32] u16 (9-bit grid index, 7 sign bits), scales[8] (4 bits per 16 values)
static void deq_iq2_xs(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 74) {
        const float d = rd_fp16(b);
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t sc = b[66 + ib];
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint16_t q = rd_u16(b + 2 + 2 * (4 * ib + l));
                const float db = d * (0.5f + ((l < 2 ? sc : sc >> 4) & 0xF)) * 0.25f;
                iq_grid8(y, db, iq2xs_grid[q & 511], ksigns_iq2xs[q >> 9]);
            }
        }
    }
}

// IQ2_S: d, qs[32] (grid index low 8 bits), signs[32], qh[8] (2 high bits per 8 values), scales[8]
static void deq_iq2_s(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 82) {
        const float d = rd_fp16(b);
        const uint8_t *qs = b + 2, *signs = b + 34, *qh = b + 66, *sc = b + 74;
        for (int ib = 0; ib < 8; ++ib)
            for (int l = 0; l < 4; ++l, y += 8) {
                const float db = d * (0.5f + ((l < 2 ? sc[ib] : sc[ib] >> 4) & 0xF)) * 0.25f;
                iq_grid8(y, db, iq2s_grid[qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)], signs[4 * ib + l]);
            }
    }
}

// IQ3_XXS: d, qs[64] (4-value grid indices), then per 32 values a u32: 4 x 7 sign bits, a 4-bit scale
static void deq_iq3_xxs(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 98) {
        const float d = rd_fp16(b);
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t *q = b + 2 + 8 * ib;
            const uint32_t a = rd_u32(b + 66 + 4 * ib);
            const float db = d * (0.5f + (a >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint64_t g = iq3xxs_grid[q[2 * l]] | (uint64_t)iq3xxs_grid[q[2 * l + 1]] << 32;
                iq_grid8(y, db, g, ksigns_iq2xs[(a >> (7 * l)) & 127]);
            }
        }
    }
}

// IQ3_S: d, qs[64] (grid index low 8 bits), qh[8] (the 9th bits), signs[32], scales[4] (4 bits per 32 values)
static void deq_iq3_s(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 110) {
        const float d = rd_fp16(b);
        const uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *sc = b + 106;
        for (int ib = 0; ib < 8; ++ib) {
            const float db = d * (1 + 2 * ((sc[ib / 2] >> (4 * (ib % 2))) & 0xF));
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint8_t *q = qs + 8 * ib + 2 * l;
                const uint64_t g = iq3s_grid[q[0] | ((qh[ib] << (8 - 2 * l)) & 256)] |
                                   (uint64_t)iq3s_grid[q[1] | ((qh[ib] << (7 - 2 * l)) & 256)] << 32;
                iq_grid8(y, db, g, signs[4 * ib + l]);
            }
        }
    }
}

// IQ1_S: d, qs[32] (grid index low 8 bits), qh[8] u16 (3 high bits per 8 values, a 3-bit scale, the delta sign)
static void deq_iq1_s(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 50) {
        const float d = rd_fp16(b);
        for (int ib = 0; ib < 8; ++ib) {
            const uint16_t qh = rd_u16(b + 34 + 2 * ib);
            const float dl = d * (2 * ((qh >> 12) & 7) + 1), delta = qh & 0x8000 ? -0.125f : 0.125f;
            for (int l = 0; l < 4; ++l, y += 8) {
                const uint64_t g = iq1s_grid[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; ++j) y[j] = dl * ((float)(int8_t)(g >> (8 * j)) + delta);
            }
        }
    }
}

// IQ1_M: qs[32], qh[16] (3 high bits and a delta sign per 8 values), scales[8] (3 bits per 16 values; the FP16 d in
// the top nibbles of the four u16)
static void deq_iq1_m(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 56) {
        uint16_t sc[4];
        for (int k = 0; k < 4; ++k) sc[k] = rd_u16(b + 48 + 2 * k);
        const uint16_t dh = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const uint8_t dd[2] = {(uint8_t)dh, (uint8_t)(dh >> 8)};
        const float d = rd_fp16(dd);
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t *qs = b + 4 * ib, *qh = b + 32 + 2 * ib;
            for (int l = 0; l < 4; ++l, y += 8) {
                const float dl = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1);
                const uint8_t h = qh[l / 2] >> (4 * (l % 2));
                const float delta = h & 0x08 ? -0.125f : 0.125f;
                const uint64_t g = iq1s_grid[qs[l] | ((h & 7) << 8)];
                for (int j = 0; j < 8; ++j) y[j] = dl * ((float)(int8_t)(g >> (8 * j)) + delta);
            }
        }
    }
}

// TQ1_0: qs[48] (5 trits per byte: base-3 digits as fixed point), qh[4] (4 trits per byte), d. Value order: bytes
// 0..31 digit l, then bytes 32..47 digit l, then qh digit l
static int tq_trit(uint8_t q, int l) {
    static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
    return (((uint16_t)(uint8_t)(q * pow3[l]) * 3) >> 8) - 1;
}
static void deq_tq1_0(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 54) {
        const float d = rd_fp16(b + 52);
        for (int l = 0; l < 5; ++l)
            for (int m = 0; m < 32; ++m) *y++ = tq_trit(b[m], l) * d;
        for (int l = 0; l < 5; ++l)
            for (int m = 0; m < 16; ++m) *y++ = tq_trit(b[32 + m], l) * d;
        for (int l = 0; l < 4; ++l)
            for (int j = 0; j < 4; ++j) *y++ = tq_trit(b[48 + j], l) * d;
    }
}

// TQ2_0: qs[64] (2 bits per value, q - 1), d. Per 32 bytes: bits 2l of byte m are value 32 l + m
static void deq_tq2_0(const void *src, float *y, int64_t n) {
    const uint8_t *b = (const uint8_t *)src;
    for (int64_t i = 0; i < n / 256; ++i, b += 66) {
        const float d = rd_fp16(b + 64);
        for (int j = 0; j < 64; j += 32)
            for (int l = 0; l < 4; ++l)
                for (int m = 0; m < 32; ++m) *y++ = (float)(((b[j + m] >> (2 * l)) & 3) - 1) * d;
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
    {QT_IQ4_NL,            "iq4_nl",            32,  18,  deq_iq4_nl,       1},
    {QT_IQ4_XS,            "iq4_xs",            256, 136, deq_iq4_xs,       1},
    {QT_IQ2_XXS,           "iq2_xxs",           256, 66,  deq_iq2_xxs,      1},
    {QT_IQ2_XS,            "iq2_xs",            256, 74,  deq_iq2_xs,       1},
    {QT_IQ2_S,             "iq2_s",             256, 82,  deq_iq2_s,        1},
    {QT_IQ3_XXS,           "iq3_xxs",           256, 98,  deq_iq3_xxs,      1},
    {QT_IQ3_S,             "iq3_s",             256, 110, deq_iq3_s,        1},
    {QT_IQ1_S,             "iq1_s",             256, 50,  deq_iq1_s,        1},
    {QT_IQ1_M,             "iq1_m",             256, 56,  deq_iq1_m,        1},
    {QT_TQ1_0,             "tq1_0",             256, 54,  deq_tq1_0,        1},
    {QT_TQ2_0,             "tq2_0",             256, 66,  deq_tq2_0,        1},
    {QT_BF16,              "bf16",              1,   2,   deq_bf16,         1},
    {QT_MXFP4,             "mxfp4",             32,  17,  deq_mxfp4,        1},
    {QT_NVFP4,             "nvfp4",             64,  36,  deq_nvfp4,        1},
    {QT_Q4_0_ROCMFP4,      "q4_0_rocmfp4",      32,  18,  deq_rocmfp4,      1},
    {QT_Q4_0_ROCMFP4_FAST, "q4_0_rocmfp4_fast", 32,  17,  deq_rocmfp4_fast, 1},
};

int qt_vec_dot_q8k(int type) {
    switch (type) {
    case QT_Q2_K: case QT_Q3_K: case QT_Q4_K: case QT_Q5_K: case QT_Q6_K: case QT_IQ4_XS:
    case QT_IQ2_XXS: case QT_IQ2_XS: case QT_IQ2_S: case QT_IQ3_XXS: case QT_IQ3_S: case QT_IQ1_S: case QT_IQ1_M:
    case QT_TQ1_0: case QT_TQ2_0:
        return 1;
    default:
        return 0;
    }
}

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
