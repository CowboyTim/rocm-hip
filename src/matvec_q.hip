// matvec_q.hip — Phase 20 Step 3b: y = W x for raw GGUF weight blocks, zero-copy.
//
// W is a GGUF tensor read in place from the pinned read-only file mapping: N rows of K values,
// row n starting at W + n * row_bytes, stored in the tensor's native block format (no repack).
// x is FP32 [K], y is FP32 [N]. One wave32 per output row; lanes stride over the row's blocks,
// decode each block with the same arithmetic as src/quant_types.cpp (bit-exact on CPU) and
// accumulate in FP32, then reduce across the wave.
//
// One extern "C" kernel per type: matvec_<type name>(W, row_bytes, x, y, N, K).
// Launch: grid (ceil(N / ROWS_PER_BLOCK)), block (32 * ROWS_PER_BLOCK).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS_PER_BLOCK 4

__device__ __forceinline__ float fp16(const uint8_t *p) {
    const uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    return __half2float(__ushort_as_half(h));
}

__device__ __forceinline__ float bf16(const uint8_t *p) {
    const uint32_t bits = ((uint32_t)p[0] | ((uint32_t)p[1] << 8)) << 16;
    return __uint_as_float(bits);
}

// ---------------------------------------------------------------- per-block dots
// Each returns sum_i deq(block)[i] * x[i] over one block.

struct BlkF32 {
    static constexpr int QK = 1, BYTES = 4;
    __device__ static float dot(const uint8_t *b, const float *x) {
        float w;
        __builtin_memcpy(&w, b, 4);
        return w * x[0];
    }
};

struct BlkF16 {
    static constexpr int QK = 1, BYTES = 2;
    __device__ static float dot(const uint8_t *b, const float *x) { return fp16(b) * x[0]; }
};

struct BlkBF16 {
    static constexpr int QK = 1, BYTES = 2;
    __device__ static float dot(const uint8_t *b, const float *x) { return bf16(b) * x[0]; }
};

struct BlkQ4_0 {
    static constexpr int QK = 32, BYTES = 18;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b);
        const uint8_t *qs = b + 2;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            s += ((qs[j] & 0x0F) - 8) * d * x[j];
            s += ((qs[j] >> 4) - 8) * d * x[j + 16];
        }
        return s;
    }
};

struct BlkQ4_1 {
    static constexpr int QK = 32, BYTES = 20;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b), m = fp16(b + 2);
        const uint8_t *qs = b + 4;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            s += ((qs[j] & 0x0F) * d + m) * x[j];
            s += ((qs[j] >> 4) * d + m) * x[j + 16];
        }
        return s;
    }
};

struct BlkQ5_0 {
    static constexpr int QK = 32, BYTES = 22;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b);
        const uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8) | ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
        const uint8_t *qs = b + 6;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const int x0 = ((qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16;
            const int x1 = ((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
            s += x0 * d * x[j];
            s += x1 * d * x[j + 16];
        }
        return s;
    }
};

struct BlkQ5_1 {
    static constexpr int QK = 32, BYTES = 24;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b), m = fp16(b + 2);
        const uint32_t qh = (uint32_t)b[4] | ((uint32_t)b[5] << 8) | ((uint32_t)b[6] << 16) | ((uint32_t)b[7] << 24);
        const uint8_t *qs = b + 8;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const int x0 = (qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10);
            const int x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10);
            s += (x0 * d + m) * x[j];
            s += (x1 * d + m) * x[j + 16];
        }
        return s;
    }
};

struct BlkQ8_0 {
    static constexpr int QK = 32, BYTES = 34;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b);
        const int8_t *qs = (const int8_t *)(b + 2);
        float s = 0.0f;
        for (int j = 0; j < 32; ++j) s += qs[j] * d * x[j];
        return s;
    }
};

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct BlkQ2_K {
    static constexpr int QK = 256, BYTES = 84;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const uint8_t *scales = b, *q = b + 16;
        const float d = fp16(b + 80), min = fp16(b + 82);
        float s = 0.0f;
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const uint8_t sc = scales[is++];
                    const float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                    for (int l = 0; l < 16; ++l) s += (dl * ((q[l + 16 * half] >> shift) & 3) - ml) * *x++;
                }
                shift += 2;
            }
            q += 32;
        }
        return s;
    }
};

struct BlkQ3_K {
    static constexpr int QK = 256, BYTES = 110;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const uint8_t *hm = b, *q = b + 32, *raw = b + 96;
        const float d_all = fp16(b + 108);
        // unpack 16 6-bit scales (same bit shuffle as the reference's aux[] words)
        int8_t sc[16];
        for (int i = 0; i < 16; ++i) {
            const int lo = i < 8 ? (raw[i] & 0xF) : (raw[i - 8] >> 4);
            const int hi = (raw[8 + (i & 3)] >> (2 * (i >> 2))) & 3;
            sc[i] = (int8_t)(lo | (hi << 4));
        }
        float s = 0.0f;
        uint8_t m = 1;
        int is = 0;
        for (int nn = 0; nn < 256; nn += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                for (int half = 0; half < 2; ++half) {
                    const float dl = d_all * (sc[is++] - 32);
                    for (int l = 0; l < 16; ++l) {
                        const int o = l + 16 * half;
                        s += dl * (((q[o] >> shift) & 3) - ((hm[o] & m) ? 0 : 4)) * *x++;
                    }
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
        return s;
    }
};

struct BlkQ4_K {
    static constexpr int QK = 256, BYTES = 144;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b), min = fp16(b + 2);
        const uint8_t *scales = b + 4, *q = b + 16;
        float s = 0.0f;
        uint8_t sc, m;
        for (int is = 0; is < 8; is += 2, q += 32) {
            scale_min_k4(is, scales, sc, m);
            const float d1 = d * sc, m1 = min * m;
            scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) s += (d1 * (q[l] & 0xF) - m1) * x[l];
            for (int l = 0; l < 32; ++l) s += (d2 * (q[l] >> 4) - m2) * x[l + 32];
            x += 64;
        }
        return s;
    }
};

struct BlkQ5_K {
    static constexpr int QK = 256, BYTES = 176;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b), min = fp16(b + 2);
        const uint8_t *scales = b + 4, *qh = b + 16, *ql = b + 48;
        float s = 0.0f;
        uint8_t sc, m, u1 = 1, u2 = 2;
        for (int is = 0; is < 8; is += 2, ql += 32) {
            scale_min_k4(is, scales, sc, m);
            const float d1 = d * sc, m1 = min * m;
            scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) s += (d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1) * x[l];
            for (int l = 0; l < 32; ++l) s += (d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2) * x[l + 32];
            x += 64;
            u1 <<= 2;
            u2 <<= 2;
        }
        return s;
    }
};

struct BlkQ6_K {
    static constexpr int QK = 256, BYTES = 210;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d = fp16(b + 208);
        const uint8_t *ql = b, *qh = b + 128;
        const int8_t *sc = (const int8_t *)(b + 192);
        float s = 0.0f;
        for (int nn = 0; nn < 256; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                s += d * sc[is + 0] * q1 * x[l];
                s += d * sc[is + 2] * q2 * x[l + 32];
                s += d * sc[is + 4] * q3 * x[l + 64];
                s += d * sc[is + 6] * q4 * x[l + 96];
            }
            x += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
        return s;
    }
};

__constant__ int8_t k_rocmfp4_codebook[16] = {0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};

__device__ __forceinline__ float ue4m3_half(uint8_t e) {
    if (e > 0x7e) return 0.0f;
    const int ex = e >> 3, m = e & 7;
    return ex == 0 ? (float)m * 0x1p-10f : (float)(8 + m) * ldexpf(1.0f, ex - 11);
}

struct BlkROCmFP4 {
    static constexpr int QK = 32, BYTES = 18;
    __device__ static float dot(const uint8_t *b, const float *x) {
        const float d0 = ue4m3_half(b[16]), d1 = ue4m3_half(b[17]);
        float s0 = 0.0f, s1 = 0.0f;
        for (int j = 0; j < 16; ++j) {
            s0 += k_rocmfp4_codebook[b[j] & 0x0f] * x[j];
            s1 += k_rocmfp4_codebook[b[j] >> 4] * x[j + 16];
        }
        return s0 * d0 + s1 * d1;
    }
};

struct BlkROCmFP4Fast {
    static constexpr int QK = 32, BYTES = 17;
    __device__ static float dot(const uint8_t *b, const float *x) {
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            s += k_rocmfp4_codebook[b[j] & 0x0f] * x[j];
            s += k_rocmfp4_codebook[b[j] >> 4] * x[j + 16];
        }
        return s * ue4m3_half(b[16]);
    }
};

// ---------------------------------------------------------------- generic kernel

template <typename B>
__device__ void matvec_impl(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const uint8_t *wr = W + (long long)row * row_bytes;
    const int nb = K / B::QK;
    float acc = 0.0f;
    for (int ib = lane; ib < nb; ib += 32) acc += B::dot(wr + (long long)ib * B::BYTES, x + (long long)ib * B::QK);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = acc;
}

#define MATVEC_KERNEL(name, B)                                                                            \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)                                     \
        matvec_##name(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {    \
        matvec_impl<B>(W, row_bytes, x, y, N, K);                                                          \
    }

MATVEC_KERNEL(f32, BlkF32)
MATVEC_KERNEL(f16, BlkF16)
MATVEC_KERNEL(bf16, BlkBF16)
MATVEC_KERNEL(q4_0, BlkQ4_0)
MATVEC_KERNEL(q4_1, BlkQ4_1)
MATVEC_KERNEL(q5_0, BlkQ5_0)
MATVEC_KERNEL(q5_1, BlkQ5_1)
MATVEC_KERNEL(q8_0, BlkQ8_0)
MATVEC_KERNEL(q2_K, BlkQ2_K)
MATVEC_KERNEL(q3_K, BlkQ3_K)
MATVEC_KERNEL(q4_K, BlkQ4_K)
MATVEC_KERNEL(q5_K, BlkQ5_K)
MATVEC_KERNEL(q6_K, BlkQ6_K)
MATVEC_KERNEL(q4_0_rocmfp4, BlkROCmFP4)
MATVEC_KERNEL(q4_0_rocmfp4_fast, BlkROCmFP4Fast)
