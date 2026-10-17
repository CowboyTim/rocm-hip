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

#include "q8_device.h"

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

// Wave-cooperative Q6_K: the 32 lanes decode one 256-value block together, 8 values each (Phase 20 speed
// work). Lane l reads ql[l], ql[l + 32] and qh[l] of each 128-value half, plus x[l + {0, 32, 64, 96}],
// so both the weight and the activation loads coalesce. One lane per block (BlkQ6_K) read 210-byte
// blocks byte by byte at a 210-byte lane stride and reached 23 GB/s on ffn_down (K = 9216).
struct BlkQ6_KCoop {
    static constexpr int QK = 256, BYTES = 210;
    __device__ static float lane_dot(const uint8_t *b, const float *x, int l) {
        const int8_t *sc = (const int8_t *)(b + 192);
        const int is = l / 16;
        float s = 0.0f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint8_t *ql = b + 64 * h, *qh = b + 128 + 32 * h;
            const int8_t *scs = sc + 8 * h;
            const float *xh = x + 128 * h;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            s += scs[is + 0] * q1 * xh[l] + scs[is + 2] * q2 * xh[l + 32] + scs[is + 4] * q3 * xh[l + 64] +
                 scs[is + 6] * q4 * xh[l + 96];
        }
        return fp16(b + 208) * s;
    }
};

// Wave-cooperative Q4_K / Q5_K, same scheme: per 64-value chunk j, lane l takes values 64j + l (low
// nibble of q[32j + l]) and 64j + 32 + l (high nibble), plus Q5_K's high bits from qh[l].
struct BlkQ4_KCoop {
    static constexpr int QK = 256, BYTES = 144;
    __device__ static float lane_dot(const uint8_t *b, const float *x, int l) {
        const float d = fp16(b), min = fp16(b + 2);
        const uint8_t *scales = b + 4, *q = b + 16;
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc1, m1, sc2, m2;
            scale_min_k4(2 * j, scales, sc1, m1);
            scale_min_k4(2 * j + 1, scales, sc2, m2);
            const uint8_t v = q[32 * j + l];
            s += (d * sc1 * (v & 0xF) - min * m1) * x[64 * j + l];
            s += (d * sc2 * (v >> 4) - min * m2) * x[64 * j + 32 + l];
        }
        return s;
    }
};

struct BlkQ5_KCoop {
    static constexpr int QK = 256, BYTES = 176;
    __device__ static float lane_dot(const uint8_t *b, const float *x, int l) {
        const float d = fp16(b), min = fp16(b + 2);
        const uint8_t *scales = b + 4, *qh = b + 16, *ql = b + 48;
        const uint8_t h = qh[l];
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint8_t sc1, m1, sc2, m2;
            scale_min_k4(2 * j, scales, sc1, m1);
            scale_min_k4(2 * j + 1, scales, sc2, m2);
            const uint8_t v = ql[32 * j + l];
            const int lo = (v & 0xF) + ((h >> (2 * j)) & 1 ? 16 : 0);
            const int hi = (v >> 4) + ((h >> (2 * j + 1)) & 1 ? 16 : 0);
            s += (d * sc1 * lo - min * m1) * x[64 * j + l];
            s += (d * sc2 * hi - min * m2) * x[64 * j + 32 + l];
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

// Wave-cooperative variant: every lane works on every block (B::lane_dot gives the lane's share)
template <typename B>
__device__ void matvec_coop_impl(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const uint8_t *wr = W + (long long)row * row_bytes;
    const int nb = K / B::QK;
    float acc = 0.0f;
#pragma unroll 4
    for (int ib = 0; ib < nb; ++ib) acc += B::lane_dot(wr + (long long)ib * B::BYTES, x + (long long)ib * B::QK, lane);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = acc;
}

#define MATVEC_COOP_KERNEL(name, B)                                                                       \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)                                     \
        matvec_##name(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {    \
        matvec_coop_impl<B>(W, row_bytes, x, y, N, K);                                                     \
    }

#define MATVEC_KERNEL(name, B)                                                                            \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)                                     \
        matvec_##name(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {    \
        matvec_impl<B>(W, row_bytes, x, y, N, K);                                                          \
    }

MATVEC_KERNEL(f16, BlkF16)
// BF16: a dedicated kernel instead of the generic one-value-per-lane loop (that read 2 bytes per load and
// reached ~140 GB/s on the 35B's 1 GB BF16 LM head). Each lane loads 8 weights (16 bytes) and 8 activations at
// a time. The activations are rounded to BF16 (round to nearest even) first: BF16's vec_dot_type in ggml is
// BF16, so the reference converts the activations before the dot product, and this matches it.
__device__ __forceinline__ float bf16_round(float v) {
    uint32_t u = __float_as_uint(v);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(u);
}

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// F32 (e.g. the MoE router, 2 MB per layer): 16-byte loads, 4 weights and 4 activations per lane per step, exact
// FP32 like ggml (F32's vec_dot_type is F32). The generic one-value-per-lane loop reached ~57 GB/s.
extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matvec_f32(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const float *wr = (const float *)(W + (long long)row * row_bytes);
    float acc = 0.0f;
    int k0 = 0;
    if ((((uintptr_t)wr | (uintptr_t)x) & 15) == 0) {  // uniform per row
        const int n4 = K / 4;
        for (int i = lane; i < n4; i += 32) {
            const float4 w = ((const float4 *)wr)[i], v = ((const float4 *)x)[i];
            acc += w.x * v.x + w.y * v.y + w.z * v.z + w.w * v.w;
        }
        k0 = n4 * 4;
    }
    for (int k = k0 + lane; k < K; k += 32) acc += wr[k] * x[k];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = acc;
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matvec_bf16(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const uint8_t *wr = W + (long long)row * row_bytes;
    float acc = 0.0f;
    int k0 = 0;
    if ((((uintptr_t)wr | (uintptr_t)x) & 15) == 0) {  // uniform per row
        const int n8 = K / 8;
        for (int i = lane; i < n8; i += 32) {
            const uint4 u = ((const uint4 *)wr)[i];
            const float4 xa = ((const float4 *)x)[2 * i], xb = ((const float4 *)x)[2 * i + 1];
            acc += bf16_lo(u.x) * bf16_round(xa.x) + bf16_hi(u.x) * bf16_round(xa.y) + bf16_lo(u.y) * bf16_round(xa.z) +
                   bf16_hi(u.y) * bf16_round(xa.w) + bf16_lo(u.z) * bf16_round(xb.x) + bf16_hi(u.z) * bf16_round(xb.y) +
                   bf16_lo(u.w) * bf16_round(xb.z) + bf16_hi(u.w) * bf16_round(xb.w);
        }
        k0 = n8 * 8;
    }
    for (int k = k0 + lane; k < K; k += 32) acc += bf16(wr + 2 * k) * bf16_round(x[k]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = acc;
}
MATVEC_KERNEL(q4_0, BlkQ4_0)
MATVEC_KERNEL(q4_1, BlkQ4_1)
MATVEC_KERNEL(q5_0, BlkQ5_0)
MATVEC_KERNEL(q5_1, BlkQ5_1)
MATVEC_KERNEL(q8_0, BlkQ8_0)
MATVEC_KERNEL(q2_K, BlkQ2_K)
MATVEC_KERNEL(q3_K, BlkQ3_K)
MATVEC_COOP_KERNEL(q4_K, BlkQ4_KCoop)
MATVEC_COOP_KERNEL(q5_K, BlkQ5_KCoop)
MATVEC_COOP_KERNEL(q6_K, BlkQ6_KCoop)
MATVEC_KERNEL(q4_0_rocmfp4, BlkROCmFP4)
MATVEC_KERNEL(q4_0_rocmfp4_fast, BlkROCmFP4Fast)

// Float expert-indexed (MoE) kernels, for types without an int8 kernel (and QWEN_MATVEC_FP32): same grid and
// ids / x_per_slot convention as matvec_q8_id_<type> below, on the FP32 input rows.
template <typename B, bool COOP>
__device__ void matvec_id_impl(const uint8_t *W, long long row_bytes, const int *ids, int n_used, int x_per_slot,
                               const float *x, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const int slot = blockIdx.y, t = blockIdx.z, e = ids[t * n_used + slot];
    const uint8_t *wr = W + ((long long)e * N + row) * row_bytes;
    const float *xr = x + (x_per_slot ? (long long)t * n_used + slot : t) * K;
    const int nb = K / B::QK;
    float acc = 0.0f;
    if constexpr (COOP) {
        for (int ib = 0; ib < nb; ++ib) acc += B::lane_dot(wr + (long long)ib * B::BYTES, xr + (long long)ib * B::QK, lane);
    } else {
        for (int ib = lane; ib < nb; ib += 32) acc += B::dot(wr + (long long)ib * B::BYTES, xr + (long long)ib * B::QK);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[((long long)t * n_used + slot) * N + row] = acc;
}

#define MATVEC_ID_KERNEL(name, B, COOP)                                                                   \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)                                     \
        matvec_id_##name(const uint8_t *W, long long row_bytes, const int *ids, int n_used, int x_per_slot, \
                         const float *x, float *y, int N, int K) {                                        \
        matvec_id_impl<B, COOP>(W, row_bytes, ids, n_used, x_per_slot, x, y, N, K);                        \
    }

MATVEC_ID_KERNEL(f32, BlkF32, false)
MATVEC_ID_KERNEL(f16, BlkF16, false)
MATVEC_ID_KERNEL(bf16, BlkBF16, false)
MATVEC_ID_KERNEL(q4_0, BlkQ4_0, false)
MATVEC_ID_KERNEL(q4_1, BlkQ4_1, false)
MATVEC_ID_KERNEL(q5_0, BlkQ5_0, false)
MATVEC_ID_KERNEL(q5_1, BlkQ5_1, false)
MATVEC_ID_KERNEL(q8_0, BlkQ8_0, false)
MATVEC_ID_KERNEL(q2_K, BlkQ2_K, false)
MATVEC_ID_KERNEL(q3_K, BlkQ3_K, false)
MATVEC_ID_KERNEL(q4_K, BlkQ4_KCoop, true)
MATVEC_ID_KERNEL(q5_K, BlkQ5_KCoop, true)
MATVEC_ID_KERNEL(q6_K, BlkQ6_KCoop, true)
MATVEC_ID_KERNEL(q4_0_rocmfp4, BlkROCmFP4, false)
MATVEC_ID_KERNEL(q4_0_rocmfp4_fast, BlkROCmFP4Fast, false)

// ================================================================ int8 path (Phase 20 speed work)
//
// The float kernels above are ALU-bound (~166 GB/s against a 240 GB/s streaming roofline): every weight
// value costs a decode, an int->float conversion and an FMA. Like llama.cpp's GPU backends, this path
// quantizes the activation vector to int8 once (quantize_q8), decodes four weights at a time into a packed
// dword and multiplies with v_dot4_i32_i8, applying the float scales once per block. The activation
// formats are exactly what ggml's CPU reference uses for these weight types (vec_dot_type), with the same
// rounding, so each block's integer sum matches ggml's bit for bit:
//   Q8_0 (ROCmFP4): 32-value blocks, d = amax / 127 stored as FP16, q = round-half-even(x * 127 / amax)
//                   (the x86 SIMD quantize_row_q8_0 the reference runs, not the _ref variant).
//   Q8_K (K-quants): 256-value blocks, d = 1 / (-127 / max) with max the signed value of largest |x|
//                   (first one on ties), q = min(127, round-half-even(x * -127 / max)), plus the sum of
//                   every 16 q values (bsums).
//
// Kernels: matvec_q8_<type>(W, W2, N1, row_bytes, q8, d, bsums, y, y2, N, K), one wave per row as above. Rows
// 0..N1-1 come from W into y, rows N1..N-1 from W2 into y2 (same type and K): two small matvecs on the same
// input in one launch (DeltaNet's alpha and beta). With one matrix, W2 = W, y2 = y, N1 = N.

__device__ __forceinline__ int dot4(int a, int b, int c) { return __builtin_amdgcn_sudot4(true, a, true, b, c, false); }

__device__ __forceinline__ uint32_t ld32(const uint8_t *p) { return *(const uint32_t *)p; }               // 4-aligned
__device__ __forceinline__ uint32_t ld32_a2(const uint8_t *p) {                                          // 2-aligned
    const uint16_t *h = (const uint16_t *)p;
    return (uint32_t)h[0] | ((uint32_t)h[1] << 16);
}

// grid K / 256, block 256: one Q8_K block and its eight Q8_0 blocks (see q8_device.h). Producers of matvec
// inputs that can (add_rmsnorm, swiglu, attn_combine) quantize their own output instead.
extern "C" __global__ void __launch_bounds__(256)
quantize_q8(const float *x, Q8Out o) {
    __shared__ Q8Shared sh;
    const int t = threadIdx.x, b = blockIdx.x;
    q8_block_value(x[b * 256 + t], true, b, t, sh, o);
}

// Four ROCmFP4 nibbles (one per byte, 0..15) -> four int8 codebook values {0,1,2,3,4,6,8,10} with sign bit 3,
// by v_perm_b32 byte lookups into a positive and a negative 8-entry table
__device__ __forceinline__ int rocmfp4_dec4(uint32_t n) {
    const uint32_t sel = n & 0x07070707u;
    const uint32_t pos = __builtin_amdgcn_perm(0x0A080604u, 0x03020100u, sel);
    const uint32_t neg = __builtin_amdgcn_perm(0xF6F8FAFCu, 0xFDFEFF00u, sel);
    const uint32_t m = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (int)((pos & ~m) | (neg & m));
}

// One ROCmFP4 block (qs as 4 dwords, two UE4M3 half scales) against its Q8_0 block
__device__ __forceinline__ float rocmfp4_q8_block(const uint32_t qs[4], uint32_t e0, uint32_t e1, const int8_t *q8,
                                                  float dx) {
    const int *x8 = (const int *)q8;
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s0 = dot4(rocmfp4_dec4(qs[k] & 0x0F0F0F0Fu), x8[k], s0);
        s1 = dot4(rocmfp4_dec4((qs[k] >> 4) & 0x0F0F0F0Fu), x8[4 + k], s1);
    }
    return ue4m3_half((uint8_t)e0) * dx * (float)s0 + ue4m3_half((uint8_t)e1) * dx * (float)s1;
}

// Lanes per row: a row whose K gives fewer lane-tasks than 32 (e.g. the MoE experts' K = 512 down projection:
// 4 ROCmFP4_FAST block quads) would leave most of the wave idle, so the host picks L in {4, 8, 16, 32} lanes per
// row and the wave handles 32 / L rows. Lane gl = lane % L of its group strides over the row's tasks by L, and
// the reduction stays inside the group (xor offsets below L). Rows per workgroup: ROWS_PER_BLOCK * 32 / L.
#define Q8_ROW_INDEX                                                                \
    const int lane = threadIdx.x & 31, gl = lane & (L - 1);                         \
    const int row = (blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5)) * (32 / L) + lane / L;

#define Q8_REDUCE(acc)                                                              \
    for (int off = L / 2; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);

// Row `row` of the (W, y) / (W2, y2) pair: sets wr and yr; returns past the last row (a whole lane group)
#define Q8_ROW_SETUP                                                                \
    Q8_ROW_INDEX                                                                    \
    if (row >= N) return;                                                           \
    const bool second_mat = row >= N1;                                              \
    const uint8_t *wr = (second_mat ? W2 : W) + (long long)(second_mat ? row - N1 : row) * row_bytes; \
    float *yr = (second_mat ? y2 : y) + (second_mat ? row - N1 : row);

#define Q8_ARGS                                                                                            \
    const uint8_t *W, const uint8_t *W2, int N1, long long row_bytes, const int8_t *q8, const float *d8, \
        const int *bs, float *y, float *y2, int N, int K, int L

// ROCmFP4 x Q8_0: a lane takes a pair of 18-byte blocks (36 bytes, 4-aligned; the host requires K % 64 == 0)
__device__ __forceinline__ float mv_rocmfp4(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K, int lane, int L) {
    (void)bs;
    const int npairs = K / 64;
    float acc = 0.0f;
    for (int p = lane; p < npairs; p += L) {
        const uint8_t *b = wr + (long long)p * 36;
        uint32_t w[9];
#pragma unroll
        for (int k = 0; k < 9; ++k) w[k] = ld32(b + 4 * k);
        const uint32_t qa[4] = {w[0], w[1], w[2], w[3]};
        const uint32_t qb[4] = {(w[4] >> 16) | (w[5] << 16), (w[5] >> 16) | (w[6] << 16), (w[6] >> 16) | (w[7] << 16),
                                (w[7] >> 16) | (w[8] << 16)};
        const int ib = 2 * p;
        acc += rocmfp4_q8_block(qa, w[4] & 0xFF, (w[4] >> 8) & 0xFF, q8 + ib * 32, d8[ib]);
        acc += rocmfp4_q8_block(qb, (w[8] >> 16) & 0xFF, w[8] >> 24, q8 + (ib + 1) * 32, d8[ib + 1]);
    }
    return acc;
}

// ROCmFP4_FAST (17-byte blocks: 16 nibble bytes + one UE4M3 scale per 32 values) x Q8_0: a lane takes four
// blocks (68 bytes, 4-aligned; the host requires K % 128 == 0) and extracts each block's 16 bytes from the
// loaded dwords with v_alignbyte_b32. Same as ggml's rocmfp4_vec_dot_q4_0_fast_q8_0: d * (sumi0 + sumi1).
__device__ __forceinline__ uint32_t dword_at(const uint32_t *w, int off) {  // off: compile-time after unrolling
    return off % 4 == 0 ? w[off / 4] : __builtin_amdgcn_alignbyte(w[off / 4 + 1], w[off / 4], off % 4);
}

__device__ __forceinline__ void rocmfp4_fast_quad(const uint8_t *b, uint32_t qs[4][4], float e[4]) {
    uint32_t w[18];
#pragma unroll
    for (int k = 0; k < 17; ++k) w[k] = ld32(b + 4 * k);
    w[17] = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int k = 0; k < 4; ++k) qs[j][k] = dword_at(w, 17 * j + 4 * k);
        e[j] = ue4m3_half((uint8_t)(dword_at(w, 17 * j + 16) & 0xFF));
    }
}

__device__ __forceinline__ float rocmfp4_fast_q8_block(const uint32_t qs[4], float e, const int8_t *q8, float dx) {
    const int *x8 = (const int *)q8;
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s0 = dot4(rocmfp4_dec4(qs[k] & 0x0F0F0F0Fu), x8[k], s0);
        s1 = dot4(rocmfp4_dec4((qs[k] >> 4) & 0x0F0F0F0Fu), x8[4 + k], s1);
    }
    return e * dx * (float)(s0 + s1);
}

__device__ __forceinline__ float mv_rocmfp4_fast(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K, int lane, int L) {
    (void)bs;
    const int nquad = K / 128;
    float acc = 0.0f;
    for (int p = lane; p < nquad; p += L) {
        uint32_t qs[4][4];
        float e[4];
        rocmfp4_fast_quad(wr + (long long)p * 68, qs, e);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int ib = 4 * p + j;
            acc += rocmfp4_fast_q8_block(qs[j], e[j], q8 + ib * 32, d8[ib]);
        }
    }
    return acc;
}

// K-quant x Q8_K kernels: the wave works on 4 blocks at a time, 8 lanes per 256-value block, each lane on 16
// consecutive quant bytes (16 or 32 values that share one scale group per nibble half). One lane per block
// sub-group also adds that block's min / offset correction through the Q8_K group sums. (A first version
// with the whole wave on one block, one dword per lane, was slower than the float kernels: too few bytes in
// flight per iteration.)
#define KQ_BLOCKS_PER_ITER 4  // batched kernels; the matvec_q8 kernels use L / 8 (L lanes per row)

// Q6_K: lane s (0..7) of its block takes ql bytes 16s..16s+15: half h = s / 4, offset r = (s & 3) * 16. Values
// are 128h + (r >= 32 ? 32 : 0) + (r & 31) + 0..15 (low nibbles) and the same + 64 (high nibbles). The -32 offset
// of every value is applied through the group sums: sum (q - 32) x = sum q x - 32 * sum x.
__device__ __forceinline__ float mv_q6_K(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K, int lane, int L) {
    const int sub = lane >> 3, s = lane & 7;
    const int h = s >> 2, r = (s & 3) * 16, l = r & 31;
    const bool second = r >= 32;
    const int sh_lo = second ? 2 : 0;
    const int v_lo = 128 * h + (second ? 32 : 0) + l, v_hi = v_lo + 64;
    const int g_lo = 8 * h + l / 16 + (second ? 2 : 0), g_hi = g_lo + 4;
    const int nb = K / 256;
    float acc = 0.0f;
    for (int ib = sub; ib < nb; ib += L / 8) {
        const uint8_t *b = wr + (long long)ib * 210;  // 2-aligned
        const int8_t *sc = (const int8_t *)(b + 192);
        const int8_t *xb = q8 + (long long)ib * 256;
        int slo = 0, shi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t ql = ld32_a2(b + 64 * h + r + 4 * k);
            const uint32_t qh = ld32_a2(b + 128 + 32 * h + l + 4 * k);
            const int vlo = (int)((ql & 0x0F0F0F0Fu) | (((qh >> sh_lo) & 0x03030303u) << 4));
            const int vhi = (int)(((ql >> 4) & 0x0F0F0F0Fu) | (((qh >> (sh_lo + 4)) & 0x03030303u) << 4));
            slo = dot4(vlo, *(const int *)(xb + v_lo + 4 * k), slo);
            shi = dot4(vhi, *(const int *)(xb + v_hi + 4 * k), shi);
        }
        int isum = sc[g_lo] * slo + sc[g_hi] * shi;
        isum -= 32 * (sc[2 * s] * bs[ib * 16 + 2 * s] + sc[2 * s + 1] * bs[ib * 16 + 2 * s + 1]);
        acc += fp16(b + 208) * d8[ib] * (float)isum;
    }
    return acc;
}

// Q4_K / Q5_K: lane s (0..7) of its block takes qs bytes 16s..16s+15: chunk c = s / 2, offset l = (s & 1) * 16.
// Low nibbles are values 64c + l + 0..15 (sub-block 2c), high nibbles 64c + 32 + l + 0..15 (sub-block 2c + 1);
// Q5_K's fifth bits come from qh bytes l..l+15. Lane s also subtracts min s via the group sums.
template <bool Q5>
__device__ __forceinline__ float mv_kquant(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K, int lane, int L) {
    constexpr int BYTES = Q5 ? 176 : 144, QS = Q5 ? 48 : 16;
    const int sub = lane >> 3, s = lane & 7;
    const int c = s >> 1, l = (s & 1) * 16;
    const int nb = K / 256;
    float acc = 0.0f;
    for (int ib = sub; ib < nb; ib += L / 8) {
        const uint8_t *b = wr + (long long)ib * BYTES;  // 4-aligned (144 and 176 are multiples of 4)
        const int8_t *xb = q8 + (long long)ib * 256 + 64 * c + l;
        int slo = 0, shi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = ld32(b + QS + 32 * c + l + 4 * k);
            uint32_t lo = q & 0x0F0F0F0Fu, hi = (q >> 4) & 0x0F0F0F0Fu;
            if (Q5) {
                const uint32_t qh = ld32(b + 16 + l + 4 * k);
                lo |= ((qh >> (2 * c)) & 0x01010101u) << 4;
                hi |= ((qh >> (2 * c + 1)) & 0x01010101u) << 4;
            }
            slo = dot4((int)lo, *(const int *)(xb + 4 * k), slo);
            shi = dot4((int)hi, *(const int *)(xb + 32 + 4 * k), shi);
        }
        uint8_t sc1, m1, sc2, m2, sm, mm;
        scale_min_k4(2 * c, b + 4, sc1, m1);
        scale_min_k4(2 * c + 1, b + 4, sc2, m2);
        scale_min_k4(s, b + 4, sm, mm);
        const int isum = sc1 * slo + sc2 * shi;
        const int msum = mm * (bs[ib * 16 + 2 * s] + bs[ib * 16 + 2 * s + 1]);
        const float dx = d8[ib];
        acc += fp16(b) * dx * (float)isum - fp16(b + 2) * dx * (float)msum;
    }
    return acc;
}


__device__ __forceinline__ float mv_q4_K(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                         int lane, int L) {
    return mv_kquant<false>(wr, q8, d8, bs, K, lane, L);
}

__device__ __forceinline__ float mv_q5_K(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                         int lane, int L) {
    return mv_kquant<true>(wr, q8, d8, bs, K, lane, L);
}

// Expert-indexed (MoE) variant: grid (ceil(N / ROWS_PER_BLOCK), n_used, T). Slot j of token t uses expert
// e = ids[t * n_used + j], i.e. rows e * N .. e * N + N - 1 of W (all experts' rows stacked), and writes row
// t * n_used + j of y. Its input is row t of the quantized activations (x_per_slot = 0: every expert of a token
// reads the token's vector) or row t * n_used + j (x_per_slot = 1: each expert reads its own, e.g. the
// SwiGLU output going into ffn_down). The selected experts never leave the GPU.
#define Q8ID_ARGS                                                                                       \
    const uint8_t *W, long long row_bytes, const int *ids, int n_used, int x_per_slot, const int8_t *q8, \
        const float *d8, const int *bs, float *y, int N, int K, int L

#define MATVEC_Q8_KERNELS(name, FN, KQ)                                                                  \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matvec_q8_##name(Q8_ARGS) {        \
        Q8_ROW_SETUP                                                                                     \
        float acc = FN(wr, q8, d8, bs, K, gl, L);                                                        \
        Q8_REDUCE(acc)                                                                                   \
        if (gl == 0) *yr = acc;                                                                          \
    }                                                                                                    \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matvec_q8_id_##name(Q8ID_ARGS) {   \
        Q8_ROW_INDEX                                                                                     \
        if (row >= N) return;                                                                            \
        const int slot = blockIdx.y, t = blockIdx.z, e = ids[t * n_used + slot];                         \
        const uint8_t *wr = W + ((long long)e * N + row) * row_bytes;                                    \
        const long long in_row = x_per_slot ? (long long)t * n_used + slot : t;                          \
        float acc = FN(wr, q8 + in_row * K, d8 + in_row * (K / ((KQ) ? 256 : 32)), bs + in_row * (K / 16), \
                       K, gl, L);                                                                        \
        Q8_REDUCE(acc)                                                                                   \
        if (gl == 0) y[((long long)t * n_used + slot) * N + row] = acc;                                  \
    }

MATVEC_Q8_KERNELS(q4_0_rocmfp4, mv_rocmfp4, 0)
MATVEC_Q8_KERNELS(q4_0_rocmfp4_fast, mv_rocmfp4_fast, 0)
MATVEC_Q8_KERNELS(q4_K, mv_q4_K, 1)
MATVEC_Q8_KERNELS(q5_K, mv_q5_K, 1)
MATVEC_Q8_KERNELS(q6_K, mv_q6_K, 1)

// ================================================================ batched int8 matmul (prefill)
//
// Y[t][n] = sum_k W[n][k] X[t][k] for T tokens: the same decoders and lane layouts as the matvec_q8 kernels,
// but each lane decodes its weight bytes once and applies them to a tile of MM_TB tokens, so a weight block is
// read once per tile instead of once per token. The activations of the T tokens are quantized as one
// flattened [T * K] vector (qmatvec_quantize), which is the same thing block by block since K % 256 == 0.
// grid (ceil(N / ROWS_PER_BLOCK), ceil(T / MM_TB)), block 32 * ROWS_PER_BLOCK; y row t at y + t * ldy.
//
// Kernels: matmul_q8_<type>(W, row_bytes, q8, d8, bs, y, ldy, N, K, T)

#define MM_TB 16

#define MM_ARGS const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y, \
                long long ldy, int N, int K, int T

#define MM_ROW_SETUP                                                                \
    const int lane = threadIdx.x & 31;                                              \
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);               \
    if (row >= N) return;                                                           \
    const uint8_t *wr = W + (long long)row * row_bytes;                             \
    const int t0 = blockIdx.y * MM_TB, nt = min(MM_TB, T - t0);                     \
    float acc[MM_TB];                                                               \
    _Pragma("unroll") for (int j = 0; j < MM_TB; ++j) acc[j] = 0.0f;

#define MM_STORE                                                                    \
    _Pragma("unroll") for (int j = 0; j < MM_TB; ++j) {                             \
        float a = acc[j];                                                           \
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);         \
        if (lane == 0 && j < nt) y[(long long)(t0 + j) * ldy + row] = a;            \
    }

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matmul_q8_q4_0_rocmfp4(MM_ARGS) {
    (void)bs;
    MM_ROW_SETUP
    const int npairs = K / 64, nb = K / 32;
    for (int p = lane; p < npairs; p += 32) {
        const uint8_t *b = wr + (long long)p * 36;
        uint32_t w[9];
#pragma unroll
        for (int k = 0; k < 9; ++k) w[k] = ld32(b + 4 * k);
        const uint32_t qs[2][4] = {{w[0], w[1], w[2], w[3]},
                                   {(w[4] >> 16) | (w[5] << 16), (w[5] >> 16) | (w[6] << 16),
                                    (w[6] >> 16) | (w[7] << 16), (w[7] >> 16) | (w[8] << 16)}};
        const float e[2][2] = {{ue4m3_half((uint8_t)(w[4] & 0xFF)), ue4m3_half((uint8_t)((w[4] >> 8) & 0xFF))},
                               {ue4m3_half((uint8_t)((w[8] >> 16) & 0xFF)), ue4m3_half((uint8_t)(w[8] >> 24))}};
        int lo[2][4], hi[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[h][k] = rocmfp4_dec4(qs[h][k] & 0x0F0F0F0Fu);
                hi[h][k] = rocmfp4_dec4((qs[h][k] >> 4) & 0x0F0F0F0Fu);
            }
        // Not unrolled: 16 copies of the body cost more in registers than the loop does (600 vs 466 tokens/s)
        for (int j = 0; j < MM_TB; ++j) {
            if (j >= nt) break;
            const long long t = t0 + j;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int ib = 2 * p + h;
                const int *x8 = (const int *)(q8 + t * K + ib * 32);
                int s0 = 0, s1 = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s0 = dot4(lo[h][k], x8[k], s0);
                    s1 = dot4(hi[h][k], x8[4 + k], s1);
                }
                const float dx = d8[t * nb + ib];
                acc[j] += e[h][0] * dx * (float)s0 + e[h][1] * dx * (float)s1;
            }
        }
    }
    MM_STORE
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matmul_q8_q6_K(MM_ARGS) {
    MM_ROW_SETUP
    const int sub = lane >> 3, s = lane & 7;
    const int h = s >> 2, r = (s & 3) * 16, l = r & 31;
    const bool second = r >= 32;
    const int sh_lo = second ? 2 : 0;
    const int v_lo = 128 * h + (second ? 32 : 0) + l, v_hi = v_lo + 64;
    const int g_lo = 8 * h + l / 16 + (second ? 2 : 0), g_hi = g_lo + 4;
    const int nb = K / 256;
    for (int ib = sub; ib < nb; ib += KQ_BLOCKS_PER_ITER) {
        const uint8_t *b = wr + (long long)ib * 210;  // 2-aligned
        const int8_t *sc = (const int8_t *)(b + 192);
        int vlo[4], vhi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t ql = ld32_a2(b + 64 * h + r + 4 * k);
            const uint32_t qh = ld32_a2(b + 128 + 32 * h + l + 4 * k);
            vlo[k] = (int)((ql & 0x0F0F0F0Fu) | (((qh >> sh_lo) & 0x03030303u) << 4));
            vhi[k] = (int)(((ql >> 4) & 0x0F0F0F0Fu) | (((qh >> (sh_lo + 4)) & 0x03030303u) << 4));
        }
        const int sclo = sc[g_lo], schi = sc[g_hi], sc_a = sc[2 * s], sc_b = sc[2 * s + 1];
        const float dw = fp16(b + 208);
        // Not unrolled: 16 copies of the body cost more in registers than the loop does (600 vs 466 tokens/s)
        for (int j = 0; j < MM_TB; ++j) {
            if (j >= nt) break;
            const long long t = t0 + j;
            const int8_t *xb = q8 + t * K + (long long)ib * 256;
            const int *bst = bs + t * (K / 16) + ib * 16;
            int slo = 0, shi = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                slo = dot4(vlo[k], *(const int *)(xb + v_lo + 4 * k), slo);
                shi = dot4(vhi[k], *(const int *)(xb + v_hi + 4 * k), shi);
            }
            const int isum = sclo * slo + schi * shi - 32 * (sc_a * bst[2 * s] + sc_b * bst[2 * s + 1]);
            acc[j] += dw * d8[t * nb + ib] * (float)isum;
        }
    }
    MM_STORE
}

template <bool Q5>
__device__ void matmul_q8_kquant(MM_ARGS) {
    constexpr int BYTES = Q5 ? 176 : 144, QS = Q5 ? 48 : 16;
    MM_ROW_SETUP
    const int sub = lane >> 3, s = lane & 7;
    const int c = s >> 1, l = (s & 1) * 16;
    const int nb = K / 256;
    for (int ib = sub; ib < nb; ib += KQ_BLOCKS_PER_ITER) {
        const uint8_t *b = wr + (long long)ib * BYTES;
        int lo[4], hi[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint32_t q = ld32(b + QS + 32 * c + l + 4 * k);
            uint32_t a = q & 0x0F0F0F0Fu, h4 = (q >> 4) & 0x0F0F0F0Fu;
            if (Q5) {
                const uint32_t qh = ld32(b + 16 + l + 4 * k);
                a |= ((qh >> (2 * c)) & 0x01010101u) << 4;
                h4 |= ((qh >> (2 * c + 1)) & 0x01010101u) << 4;
            }
            lo[k] = (int)a;
            hi[k] = (int)h4;
        }
        uint8_t sc1, m1, sc2, m2, sm, mm;
        scale_min_k4(2 * c, b + 4, sc1, m1);
        scale_min_k4(2 * c + 1, b + 4, sc2, m2);
        scale_min_k4(s, b + 4, sm, mm);
        const float dw = fp16(b), dmin = fp16(b + 2);
        // Not unrolled: 16 copies of the body cost more in registers than the loop does (600 vs 466 tokens/s)
        for (int j = 0; j < MM_TB; ++j) {
            if (j >= nt) break;
            const long long t = t0 + j;
            const int8_t *xb = q8 + t * K + (long long)ib * 256 + 64 * c + l;
            const int *bst = bs + t * (K / 16) + ib * 16;
            int slo = 0, shi = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                slo = dot4(lo[k], *(const int *)(xb + 4 * k), slo);
                shi = dot4(hi[k], *(const int *)(xb + 32 + 4 * k), shi);
            }
            const int isum = sc1 * slo + sc2 * shi;
            const int msum = mm * (bst[2 * s] + bst[2 * s + 1]);
            const float dx = d8[t * nb + ib];
            acc[j] += dw * dx * (float)isum - dmin * dx * (float)msum;
        }
    }
    MM_STORE
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matmul_q8_q4_K(MM_ARGS) {
    matmul_q8_kquant<false>(W, row_bytes, q8, d8, bs, y, ldy, N, K, T);
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matmul_q8_q5_K(MM_ARGS) {
    matmul_q8_kquant<true>(W, row_bytes, q8, d8, bs, y, ldy, N, K, T);
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matmul_q8_q4_0_rocmfp4_fast(MM_ARGS) {
    (void)bs;
    MM_ROW_SETUP
    const int nquad = K / 128, nb = K / 32;
    for (int p = lane; p < nquad; p += 32) {
        uint32_t qs[4][4];
        float e[4];
        rocmfp4_fast_quad(wr + (long long)p * 68, qs, e);
        int lo[4][4], hi[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[j][k] = rocmfp4_dec4(qs[j][k] & 0x0F0F0F0Fu);
                hi[j][k] = rocmfp4_dec4((qs[j][k] >> 4) & 0x0F0F0F0Fu);
            }
        // Not unrolled: see matmul_q8_q4_0_rocmfp4
        for (int jt = 0; jt < MM_TB; ++jt) {
            if (jt >= nt) break;
            const long long t = t0 + jt;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4 * p + j;
                const int *x8 = (const int *)(q8 + t * K + ib * 32);
                int s0 = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    s0 = dot4(lo[j][k], x8[k], s0);
                    s0 = dot4(hi[j][k], x8[4 + k], s0);
                }
                acc[jt] += e[j] * d8[t * nb + ib] * (float)s0;
            }
        }
    }
    MM_STORE
}
