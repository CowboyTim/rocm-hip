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

// Several tokens (prefill on the float path, e.g. rows too short for the int8 kernels: the qwen4exp mixers'
// K = 320 up projections): grid (ceil(N / ROWS_PER_BLOCK), ceil(T / MMF_TT)). Each wave reads its row's blocks
// once per tile of MMF_TT tokens and sums every token exactly as matvec_impl does (same blocks per lane, same
// order, same reduction), so a prefill stays bit-identical to token by token.
#define MMF_TT 8
template <typename B>
__device__ void matmul_float_impl(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K,
                                  int T) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const int t0 = blockIdx.y * MMF_TT, nt = min(MMF_TT, T - t0);
    const uint8_t *wr = W + (long long)row * row_bytes;
    const int nb = K / B::QK;
    float acc[MMF_TT];
#pragma unroll
    for (int j = 0; j < MMF_TT; ++j) acc[j] = 0.0f;
    for (int ib = lane; ib < nb; ib += 32) {
        const uint8_t *wb = wr + (long long)ib * B::BYTES;
#pragma unroll
        for (int j = 0; j < MMF_TT; ++j)
            if (j < nt) acc[j] += B::dot(wb, x + (size_t)(t0 + j) * K + (long long)ib * B::QK);
    }
#pragma unroll
    for (int j = 0; j < MMF_TT; ++j) {
        float a = acc[j];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
        if (lane == 0 && j < nt) y[(size_t)(t0 + j) * N + row] = a;
    }
}

#define MATVEC_KERNEL(name, B)                                                                            \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)                                     \
        matvec_##name(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {    \
        matvec_impl<B>(W, row_bytes, x, y, N, K);                                                          \
    }                                                                                                     \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)                                     \
        matmul_f_##name(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K,     \
                        int T) {                                                                          \
        matmul_float_impl<B>(W, row_bytes, x, y, N, K, T);                                                 \
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

// F32 (e.g. the MoE router, 2 MB per layer, but only 256 rows): one row per 128-thread workgroup, so the GPU
// has enough waves in flight (one wave per row reached ~97 GB/s). Wave w sums quarter w of the row with
// 16-byte loads, 4 weights and 4 activations per lane per step, exact FP32 like ggml (F32's vec_dot_type is
// F32); the part sums are added in order. matmul_f32 (several tokens) uses the same partition and
// order, so prefill is bit-identical to token by token. Launch: grid N, block 128.
#define F32_PARTS 4

// w . v with explicit FMAs: HIP lets the compiler fuse a * b + c on its own, which it may do differently in
// matvec_f32 and matmul_f32; written out, both round identically
__device__ __forceinline__ float dot4f(const float4 w, const float4 v) {
    return fmaf(w.w, v.w, fmaf(w.z, v.z, fmaf(w.y, v.y, w.x * v.x)));
}

// Quarter `part` of row wr . x, reduced over the wave (vector path: 16-byte-aligned wr / x and K % 16 == 0)
__device__ __forceinline__ float f32_part_vec(const float4 *wr, const float4 *x, int n4, int part, int lane) {
    const int q = n4 / F32_PARTS;
    float acc = 0.0f;
#pragma unroll 4
    for (int i = part * q + lane; i < (part + 1) * q; i += 32) acc += dot4f(wr[i], x[i]);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    return acc;
}

extern "C" __global__ void __launch_bounds__(32 * F32_PARTS)
matvec_f32(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {
    __shared__ float part[F32_PARTS];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, row = blockIdx.x;
    if (row >= N) return;  // uniform per workgroup
    const float *wr = (const float *)(W + (long long)row * row_bytes);
    float acc;
    if ((((uintptr_t)wr | (uintptr_t)x) & 15) == 0 && K % (4 * F32_PARTS) == 0) {  // uniform
        acc = f32_part_vec((const float4 *)wr, (const float4 *)x, K / 4, w, lane);
    } else {
        acc = 0.0f;
        for (int k = K * w / F32_PARTS + lane; k < K * (w + 1) / F32_PARTS; k += 32) acc += wr[k] * x[k];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    }
    if (lane == 0) part[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = part[0];
#pragma unroll
        for (int k = 1; k < F32_PARTS; ++k) s += part[k];
        y[row] = s;
    }
}

// F32 for MF32_TT tokens at a time (prefill: the MoE router ran one launch per token). grid (N, ceil(T /
// MF32_TT)), block 128; each weight float4 is loaded once for the tile's tokens. Every token's sum has
// matvec_f32's vector-path partition, expression and order: bit-identical to it. The host guarantees
// 16-byte-aligned rows and x, and K % 16 == 0.
#define MF32_TT 8
extern "C" __global__ void __launch_bounds__(32 * F32_PARTS)
matmul_f32(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K, int T) {
    __shared__ float part[F32_PARTS][MF32_TT];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, row = blockIdx.x;
    if (row >= N) return;
    const int t0 = blockIdx.y * MF32_TT, nt = min(MF32_TT, T - t0);
    const float4 *wr = (const float4 *)(W + (long long)row * row_bytes);
    const float4 *xt = (const float4 *)(x + (size_t)t0 * K);
    const int n4 = K / 4, q = n4 / F32_PARTS;
    float acc[MF32_TT];
#pragma unroll
    for (int j = 0; j < MF32_TT; ++j) acc[j] = 0.0f;
    for (int i = w * q + lane; i < (w + 1) * q; i += 32) {
        const float4 wv = wr[i];
#pragma unroll
        for (int j = 0; j < MF32_TT; ++j)
            if (j < nt) acc[j] += dot4f(wv, xt[(size_t)j * n4 + i]);
    }
#pragma unroll
    for (int j = 0; j < MF32_TT; ++j) {
        float a = acc[j];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
        if (lane == 0) part[w][j] = a;
    }
    __syncthreads();
    if (threadIdx.x < nt) {
        float s = part[0][threadIdx.x];
#pragma unroll
        for (int k = 1; k < F32_PARTS; ++k) s += part[k][threadIdx.x];
        y[(size_t)(t0 + threadIdx.x) * N + row] = s;
    }
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

// grid ceil(n / 256), block 256: one Q8_K block and its eight Q8_0 blocks (see q8_device.h); n % 32 == 0. A
// partial last block has whole Q8_0 blocks (its Q8_K side is only read when every row is whole Q8_K blocks).
// Producers of matvec inputs that can (add_rmsnorm, swiglu, attn_combine) quantize their own output instead.
extern "C" __global__ void __launch_bounds__(256)
quantize_q8(const float *x, int n, Q8Out o) {
    __shared__ Q8Shared sh;
    const int t = threadIdx.x, b = blockIdx.x, i = b * 256 + t;
    q8_block_value(i < n ? x[i] : 0.0f, i < n, b, t, sh, o);
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

// Row `row` of up to four stacked matrices on the same input: rows 0 .. N1 - 1 from (W, y), N1 .. N2 - 1 from
// (W2, y2), N2 .. N3 - 1 from (W3, y3), N3 .. N - 1 from (W4, y4). Sets wr and yr; returns past the last row (a
// whole lane group).
#define Q8_ROW_SETUP                                                                \
    Q8_ROW_INDEX                                                                    \
    if (row >= N) return;                                                           \
    const int mat = (row >= N1) + (row >= N2) + (row >= N3);                        \
    const int base = mat == 0 ? 0 : mat == 1 ? N1 : mat == 2 ? N2 : N3;             \
    const uint8_t *wr = (mat == 0 ? W : mat == 1 ? W2 : mat == 2 ? W3 : W4) + (long long)(row - base) * row_bytes; \
    float *yr = (mat == 0 ? y : mat == 1 ? y2 : mat == 2 ? y3 : y4) + (row - base);

#define Q8_ARGS                                                                                            \
    const uint8_t *W, const uint8_t *W2, const uint8_t *W3, const uint8_t *W4, int N1, int N2, int N3,     \
        long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y, float *y2, float *y3, \
        float *y4, int N, int K, int L

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
    // The batched kernel (mmb_rocmfp4_fast) gives each row 32 lanes: lane l sums quads l, l + 32, ... and the
    // lanes are added by an xor tree. With L < 32 lanes per row, this lane stands in for the 32 / L virtual lanes
    // lane + k L, keeps their sums apart and adds them in that tree's order (offsets 16 .. L, then Q8_REDUCE
    // does L / 2 .. 1), so a token gives the same bits here as in a prefill batch.
    const int nquad = K / 128, V = 32 / L;
    float a[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        a[k] = 0.0f;
        if (k >= V) continue;
        for (int p = lane + k * L; p < nquad; p += 32) {
            uint32_t qs[4][4];
            float e[4];
            rocmfp4_fast_quad(wr + (long long)p * 68, qs, e);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4 * p + j;
                a[k] += rocmfp4_fast_q8_block(qs[j], e[j], q8 + ib * 32, d8[ib]);
            }
        }
    }
#pragma unroll
    for (int d = 4; d >= 1; d >>= 1)  // virtual-lane offset d L = 16 .. L
        if (d < V)
#pragma unroll
            for (int k = 0; k < 4; ++k)
                if (k < d) a[k] += a[k + d];
    return a[0];
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
// SwiGLU output going into ffn_down). The selected experts never leave the GPU. With grid (.., 2 n_used, T),
// slots n_used .. 2 n_used - 1 apply a second matrix of the same shape and type, W2 -> y2, to the same input
// (the experts' gate and up projections in one launch). An id of -1 selects the extra matrix Wx (W2x for the
// second): one more N-row matrix of the same type, e.g. the shared expert as one more slot.
#define Q8ID_ARGS                                                                                       \
    const uint8_t *W, const uint8_t *W2, long long row_bytes, const int *ids, int n_used, int x_per_slot,     \
        const int8_t *q8, const float *d8, const int *bs, float *y, float *y2, int N, int K, int L,           \
        const uint8_t *Wx, const uint8_t *W2x

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
        const bool m2 = (int)blockIdx.y >= n_used; /* grid y = 2 n_used: the second matrix (W2 -> y2) */  \
        const int slot = blockIdx.y - (m2 ? n_used : 0), t = blockIdx.z, e = ids[t * n_used + slot];       \
        const uint8_t *wr = e >= 0 ? (m2 ? W2 : W) + ((long long)e * N + row) * row_bytes                \
                                   : (m2 ? W2x : Wx) + (long long)row * row_bytes;                        \
        const long long in_row = x_per_slot ? (long long)t * n_used + slot : t;                          \
        float acc = FN(wr, q8 + in_row * K, d8 + in_row * (K / ((KQ) ? 256 : 32)), bs + in_row * (K / 16), \
                       K, gl, L);                                                                        \
        Q8_REDUCE(acc)                                                                                   \
        if (gl == 0) (m2 ? y2 : y)[((long long)t * n_used + slot) * N + row] = acc;                      \
    }

// matvec_q8_lr_q4_0_rocmfp4_fast(W, row_bytes, lo, n, K, scale, y, N, L): y = W x for one token, where x is
// SiLU(lo * scale) zero-padded from n to K values and quantized to Q8_0 by every workgroup into LDS first, with
// hc_lowrank's and q8_block_value's arithmetic (so the same bits as those two kernels and matvec_q8 after them).
// The qwen4exp mixers' up projection (K = 384 padded from the rank 320) without the separate hc_lowrank launch.
// K % 128 == 0, K <= LR_MAXK.
#define LR_MAXK 1024
extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matvec_q8_lr_q4_0_rocmfp4_fast(const uint8_t *W, long long row_bytes, const float *lo, int n, int K, float scale,
                               float *y, int N, int L) {
    __shared__ __attribute__((aligned(16))) int8_t xs[LR_MAXK];
    __shared__ float xd[LR_MAXK / 32];
    for (int i = threadIdx.x; i < K; i += 32 * ROWS_PER_BLOCK) {  // whole 32-value blocks per wave (K % 128)
        float v = 0.0f;
        if (i < n) {
            const float a = lo[i] * scale;
            v = a / (1.0f + __expf(-a));
        }
        float am = fabsf(v);
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor(am, off, 32));
        const float id0 = am != 0.0f ? 127.0f / am : 0.0f;
        xs[i] = (int8_t)(int)rintf(v * id0);
        if ((threadIdx.x & 31) == 0) xd[i / 32] = __half2float(__float2half(am / 127.0f));
    }
    __syncthreads();
    Q8_ROW_INDEX
    if (row >= N) return;
    float acc = mv_rocmfp4_fast(W + (long long)row * row_bytes, xs, xd, nullptr, K, gl, L);
    Q8_REDUCE(acc)
    if (gl == 0) y[row] = acc;
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

// Arguments of the kernel bodies below (device templates over a row map RM, see the kernels at the end)
#define MM_BARGS const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y, \
                 long long ldy, int N, int K

// Which input / output rows a workgroup's tile of up to MM_TB tokens covers, and where its weight rows are
struct RowsLinear {  // prefill: tokens t0 .. t0 + nt - 1, rows t of the input and of y
    long long t0;
    int nt;
    __device__ const uint8_t *row_ptr(const uint8_t *W, int row, long long rb, int) const { return W + (long long)row * rb; }
    __device__ long long in(int j) const { return t0 + j; }
    __device__ long long out(int j) const { return t0 + j; }
};

struct RowsExpert {  // MoE prefill: one expert's tile of (token, slot) entries, rows from lists
    int e, nt;
    const int *in_rows, *out_rows;
    __device__ const uint8_t *row_ptr(const uint8_t *W, int row, long long rb, int N) const {
        return W + ((long long)e * N + row) * rb;
    }
    __device__ long long in(int j) const { return in_rows[j]; }
    __device__ long long out(int j) const { return out_rows[j]; }
};

#define MM_ROW_SETUP                                                                \
    const int lane = threadIdx.x & 31;                                              \
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);               \
    if (row >= N) return;                                                           \
    const uint8_t *wr = rm.row_ptr(W, row, row_bytes, N);                           \
    const int nt = rm.nt;                                                           \
    float acc[MM_TB];                                                               \
    _Pragma("unroll") for (int j = 0; j < MM_TB; ++j) acc[j] = 0.0f;

#define MM_STORE                                                                    \
    _Pragma("unroll") for (int j = 0; j < MM_TB; ++j) {                             \
        float a = acc[j];                                                           \
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);         \
        if (lane == 0 && j < nt) y[rm.out(j) * ldy + row] = a;                      \
    }

template <class RM>
__device__ __forceinline__ void mmb_rocmfp4(MM_BARGS, const RM &rm) {
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
            const long long t = rm.in(j);
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

template <class RM>
__device__ __forceinline__ void mmb_q6_K(MM_BARGS, const RM &rm) {
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
            const long long t = rm.in(j);
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

template <bool Q5, class RM>
__device__ __forceinline__ void mmb_kquant(MM_BARGS, const RM &rm) {
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
            const long long t = rm.in(j);
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

template <class RM>
__device__ __forceinline__ void mmb_rocmfp4_fast(MM_BARGS, const RM &rm) {
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
            const long long t = rm.in(jt);
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

__device__ __forceinline__ void mmb_q4_K_lin(MM_BARGS, const RowsLinear &rm) { mmb_kquant<false>(W, row_bytes, q8, d8, bs, y, ldy, N, K, rm); }
__device__ __forceinline__ void mmb_q5_K_lin(MM_BARGS, const RowsLinear &rm) { mmb_kquant<true>(W, row_bytes, q8, d8, bs, y, ldy, N, K, rm); }
__device__ __forceinline__ void mmb_q4_K_exp(MM_BARGS, const RowsExpert &rm) { mmb_kquant<false>(W, row_bytes, q8, d8, bs, y, ldy, N, K, rm); }
__device__ __forceinline__ void mmb_q5_K_exp(MM_BARGS, const RowsExpert &rm) { mmb_kquant<true>(W, row_bytes, q8, d8, bs, y, ldy, N, K, rm); }

// matmul_q8_<type>: grid (ceil(N / ROWS_PER_BLOCK), ceil(T / MM_TB)) over T consecutive tokens.
// matmul_q8_id_<type> (MoE prefill): grid (ceil(N / ROWS_PER_BLOCK), max_tiles). Tile k (k < *n_tiles) is up to
// MM_TB (token, slot) entries that chose the same expert tile_e[k]: entries tile_start[k] .. + tile_n[k] - 1 of
// the in_rows / out_rows lists (built by moe_group in gpu_ops.hip). W stacks all experts' N rows; y has N columns.
// So each expert's weights are decoded once per tile of its tokens instead of once per token.
#define MMID_ARGS                                                                                         \
    const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y, int N, int K, \
        const int *n_tiles, const int *tile_e, const int *tile_start, const int *tile_n, const int *in_rows,          \
        const int *out_rows

#define MATMUL_Q8_KERNELS(name, BODY_LIN, BODY_EXP)                                                       \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matmul_q8_##name(MM_ARGS) {         \
        const int t0 = blockIdx.y * MM_TB;                                                                \
        const RowsLinear rm{t0, min(MM_TB, T - t0)};                                                      \
        BODY_LIN(W, row_bytes, q8, d8, bs, y, ldy, N, K, rm);                                             \
    }                                                                                                     \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matmul_q8_id_##name(MMID_ARGS) {    \
        const int k = blockIdx.y;                                                                         \
        if (k >= *n_tiles) return;                                                                        \
        const RowsExpert rm{tile_e[k], tile_n[k], in_rows + tile_start[k], out_rows + tile_start[k]};     \
        BODY_EXP(W, row_bytes, q8, d8, bs, y, N, N, K, rm);                                               \
    }

MATMUL_Q8_KERNELS(q4_0_rocmfp4, mmb_rocmfp4<RowsLinear>, mmb_rocmfp4<RowsExpert>)
MATMUL_Q8_KERNELS(q4_0_rocmfp4_fast, mmb_rocmfp4_fast<RowsLinear>, mmb_rocmfp4_fast<RowsExpert>)
MATMUL_Q8_KERNELS(q6_K, mmb_q6_K<RowsLinear>, mmb_q6_K<RowsExpert>)
MATMUL_Q8_KERNELS(q4_K, mmb_q4_K_lin, mmb_q4_K_exp)
MATMUL_Q8_KERNELS(q5_K, mmb_q5_K_lin, mmb_q5_K_exp)

// ================================================================ WMMA tiled matmul (prefill, Phase 20 W2)
//
// Y[t][n] = sum_k W[n][k] X[t][k] for T tokens with v_wmma_i32_16x16x16_iu8 (55.5 TOPS on gfx1151, vs 29 for
// v_dot4). A workgroup of 4 waves computes a 64 weight-row x 64 token tile; wave w owns rows 16w..16w+15 and
// all 64 tokens (4 float accumulator tiles of 16 x 16). Per 128-value K chunk the workgroup decodes the 64
// rows' weights to int8 into LDS once (with their scales) and copies the 64 tokens' Q8_0 bytes and scales into
// LDS; each 16-value step is then one WMMA per 16 x 16 sub-tile, an exact int32 dot over one ROCmFP4
// half-block, accumulated as (float)dot * e_w[row] * d_x[token] (the same integer sums and scale products as
// ggml; only the float accumulation order differs from the dot4 kernels).
// WMMA wave32 layout (tests/wmma_layout_verify): A row i / B column i in lanes i and i + 16, 16 int8 each;
// C column l % 16, row 2r + l / 16.
// grid (ceil(N / 64), ceil(T / 64)), block 128. Requires K % 128 == 0.

typedef int v4i_t __attribute__((ext_vector_type(4)));
typedef int v8i_t __attribute__((ext_vector_type(8)));

#define WM_TILE 64
#define WM_KC 128

// Weight staging policies for the Q8_0-activation WMMA kernel. Thread tid stages weight row tid / 2, blocks
// wb = 2 * (tid & 1) and wb + 1 of the chunk's 4 blocks: load() reads them from global memory into registers
// (so the next chunk's loads overlap this chunk's WMMAs), store() decodes them to int8 in LDS and writes the two
// half-block scales of each block.
struct WStageFP4 {  // Q4_0_ROCMFP4: 18-byte blocks (2-aligned), a UE4M3 scale per 16 values
    uint32_t q[2][5];
    __device__ void load(const uint8_t *W, long long rb, int row, bool ok, int b0, int wb) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (ok) {
                const uint8_t *b = W + (long long)row * rb + (long long)(b0 + wb + j) * 18;
#pragma unroll
                for (int k = 0; k < 4; ++k) q[j][k] = ld32_a2(b + 4 * k);
                q[j][4] = *(const uint16_t *)(b + 16);
            } else {
#pragma unroll
                for (int k = 0; k < 5; ++k) q[j][k] = 0;
            }
        }
    }
    __device__ void store(int8_t *srow, float *erow, int wb) const {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t *dst = (uint32_t *)(srow + (wb + j) * 32);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst[k] = (uint32_t)rocmfp4_dec4(q[j][k] & 0x0F0F0F0Fu);
                dst[4 + k] = (uint32_t)rocmfp4_dec4((q[j][k] >> 4) & 0x0F0F0F0Fu);
            }
            erow[(wb + j) * 2] = ue4m3_half((uint8_t)(q[j][4] & 0xFF));
            erow[(wb + j) * 2 + 1] = ue4m3_half((uint8_t)(q[j][4] >> 8));
        }
    }
};

struct WStageFast {  // Q4_0_ROCMFP4_FAST: 17-byte blocks, one UE4M3 scale per 32 values. The chunk's 4 blocks
                     // of a row are a 68-byte, 4-aligned quad: load it whole, extract bytes with v_alignbyte.
    uint32_t w[18];
    __device__ void load(const uint8_t *W, long long rb, int row, bool ok, int b0, int) {
        const uint8_t *b = W + (long long)row * rb + (long long)b0 * 17;  // b0 % 4 == 0
#pragma unroll
        for (int k = 0; k < 17; ++k) w[k] = ok ? ld32(b + 4 * k) : 0;
        w[17] = 0;
    }
    __device__ void store(int8_t *srow, float *erow, int wb) const {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int off = 17 * (wb + j);
            uint32_t *dst = (uint32_t *)(srow + (wb + j) * 32);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t qd = dword_at(w, off + 4 * k);
                dst[k] = (uint32_t)rocmfp4_dec4(qd & 0x0F0F0F0Fu);
                dst[4 + k] = (uint32_t)rocmfp4_dec4((qd >> 4) & 0x0F0F0F0Fu);
            }
            const float e = ue4m3_half((uint8_t)(dword_at(w, off + 16) & 0xFF));
            erow[(wb + j) * 2] = e;
            erow[(wb + j) * 2 + 1] = e;
        }
    }
};

// Token-row maps for the WMMA kernels: which weight rows (an expert's, for MoE) and which input / output token
// rows a workgroup's tile covers
struct WRowsLinear {  // prefill: tokens t0 .. t0 + nt - 1
    int t0, nt;
    __device__ const uint8_t *base(const uint8_t *W, long long, int) const { return W; }
    __device__ long long in(int j) const { return t0 + j; }
    __device__ long long out(int j) const { return t0 + j; }
};

struct WRowsExpert {  // MoE prefill: one expert's tile of (token, slot) entries (moe_group)
    int e, nt;
    const int *in_rows, *out_rows;
    __device__ const uint8_t *base(const uint8_t *W, long long rb, int N) const { return W + (long long)e * N * rb; }
    __device__ long long in(int j) const { return in_rows[j]; }
    __device__ long long out(int j) const { return out_rows[j]; }
};

// NT: 16-token sub-tiles per workgroup (4: 64 tokens for dense prefill; 1: 16 entries for an expert tile)
template <class WS, int NT, class RM>
__device__ __forceinline__ void wmma_q80(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8,
                                         float *y, long long ldy, int N, int K, const RM &rm) {
    constexpr int TOK = 16 * NT;
    // Rows padded by 16 bytes: the WMMA operand loads read 16 bytes from 16 consecutive rows, and an unpadded
    // 128-byte row stride would put them all in the same LDS banks
    __shared__ __attribute__((aligned(16))) int8_t sw[WM_TILE][WM_KC + 16];  // decoded weights
    __shared__ __attribute__((aligned(16))) int8_t sx[TOK][WM_KC + 16];      // token activations
    __shared__ float sew[WM_TILE][WM_KC / 16];                               // weight half-block scales
    __shared__ float sdx[TOK][WM_KC / 32];                                   // token block scales
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int r0 = blockIdx.x * WM_TILE, T = rm.nt;
    const int nb = K / 32;
    const uint8_t *Wb = rm.base(W, row_bytes, N);

    float acc[NT][8];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[n][r] = 0.0f;

    // One K chunk's global data in registers: weights per the staging policy; activations: NT uint4 per thread
    // (task tid + 128 c: token task / 8, bytes 16 (task % 8))
    WS ws;
    uint4 gx[NT];
    const int wr_row = tid >> 1, wr_b = 2 * (tid & 1);
    auto load_chunk = [&](int k0) {
        ws.load(Wb, row_bytes, r0 + wr_row, r0 + wr_row < N, k0 / 32, wr_b);
#pragma unroll
        for (int c = 0; c < NT; ++c) {
            const int task = tid + 128 * c, tt = task >> 3;
            gx[c] = tt < T ? *(const uint4 *)(q8 + rm.in(tt) * K + k0 + 16 * (task & 7)) : uint4{0, 0, 0, 0};
        }
    };
    auto store_chunk = [&](int k0) {
        const int b0 = k0 / 32;
        ws.store(&sw[wr_row][0], &sew[wr_row][0], wr_b);
#pragma unroll
        for (int c = 0; c < NT; ++c) {
            const int task = tid + 128 * c;
            *(uint4 *)&sx[task >> 3][16 * (task & 7)] = gx[c];
        }
        // token block scales: TOK tokens x 4 blocks
        for (int task = tid; task < TOK * 4; task += 128) {
            const int tt = task >> 2, c = task & 3;
            sdx[tt][c] = tt < T ? d8[rm.in(tt) * nb + b0 + c] : 0.0f;
        }
    };

    load_chunk(0);
    const int i = lane & 15, hi = lane >> 4;
    for (int k0 = 0; k0 < K; k0 += WM_KC) {
        store_chunk(k0);
        __syncthreads();
        if (k0 + WM_KC < K) load_chunk(k0 + WM_KC);  // in flight during the WMMAs below
        // Not unrolled: unrolling all 8 steps let the compiler hoist every step's operand loads and spill
        // (256 VGPRs + 260 spilled)
#pragma unroll 1
        for (int h = 0; h < WM_KC / 16; ++h) {
            const v4i_t a = *(const v4i_t *)&sw[16 * w + i][16 * h];
            v8i_t c[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {  // issue the WMMAs first, then their epilogues
                const v4i_t bx = *(const v4i_t *)&sx[16 * n + i][16 * h];
                c[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, bx, v8i_t{}, false);
            }
            float ew[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) ew[r] = sew[16 * w + 2 * r + hi][h];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const float dx = sdx[16 * n + i][h / 2];
#pragma unroll
                for (int r = 0; r < 8; ++r) acc[n][r] += (float)c[n][r] * (ew[r] * dx);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int tt = 16 * n + i;
        if (tt >= T) continue;
        const long long orow = rm.out(tt);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = r0 + 16 * w + 2 * r + hi;
            if (row < N) y[orow * ldy + row] = acc[n][r];
        }
    }
}

// matmul_wmma_<type>: grid (ceil(N / 64), ceil(T / 64)). matmul_wmma_id_<type> (MoE prefill): grid
// (ceil(N / 64), max_tiles) over moe_group's expert tiles, 16 entries wide; same arguments as matmul_q8_id_<type>.
#define MATMUL_WMMA_Q80_KERNELS(name, WS)                                                                    \
    extern "C" __global__ void __launch_bounds__(128)                                                        \
        matmul_wmma_##name(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, float *y, \
                           long long ldy, int N, int K, int T) {                                             \
        const int t0 = blockIdx.y * WM_TILE;                                                                 \
        const WRowsLinear rm{t0, min(WM_TILE, T - t0)};                                                      \
        wmma_q80<WS, 4>(W, row_bytes, q8, d8, y, ldy, N, K, rm);                                             \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) matmul_wmma_id_##name(MMID_ARGS) {                      \
        (void)bs;                                                                                            \
        const int k = blockIdx.y;                                                                            \
        if (k >= *n_tiles) return;                                                                           \
        const WRowsExpert rm{tile_e[k], tile_n[k], in_rows + tile_start[k], out_rows + tile_start[k]};       \
        wmma_q80<WS, 1>(W, row_bytes, q8, d8, y, N, N, K, rm);                                               \
    }

MATMUL_WMMA_Q80_KERNELS(q4_0_rocmfp4, WStageFP4)
MATMUL_WMMA_Q80_KERNELS(q4_0_rocmfp4_fast, WStageFast)

// ---------------------------------------------------------------- WMMA for K-quants x Q8_K (Phase 20 W3)
//
// Same tiling as wmma_q80 (64 rows x 64 tokens, 4 waves, 128-value chunks = half a 256-value super-block), with
// the K-quant arithmetic of ggml's vec_dot_q*_K_q8_K: per chunk, 8 WMMAs of 16 values each (Q6_K: one 16-value
// scale group each; Q4_K/Q5_K: two per 32-value sub-block) accumulate into int32 as isum += sc[row][g] * dot,
// then one float step per chunk: acc += d_x * (d_w * isum - dmin_w * msum), msum = sum over the chunk's sub-blocks
// of m[row][sub] * (bs[2 sub] + bs[2 sub + 1]) (Q4_K/Q5_K mins; Q6_K has none: its -32 offset is applied while
// decoding, since q - 32 fits int8).
// Staging: thread tid takes weight row tid / 2, half `part` = tid & 1 of the chunk (64 values).

// Per byte: b - 32 for b in 0..63, as a signed int8, without carries between bytes
__device__ __forceinline__ uint32_t sub32_bytes(uint32_t b) {
    const uint32_t r = b ^ 0x20202020u;
    const uint32_t m = r & 0x20202020u;  // set where b < 32: the result is negative
    return r | (m << 1) | (m << 2);
}

template <int KT>  // 4: Q4_K, 5: Q5_K, 6: Q6_K
struct WStageK {
    static constexpr int BYTES = KT == 6 ? 210 : KT == 5 ? 176 : 144;
    uint32_t q[16];   // Q4/5: 8 dwords of qs for this part's 64-value sub-chunk; Q6: 16 dwords of ql (both parts)
    uint32_t qh[8];   // Q5 / Q6 high bits (32 bytes at the chunk's qh offset)
    uint32_t sc[3];   // Q4/5: the 12 scale bytes; Q6: 8 group scales (2 dwords)
    uint32_t dd;      // fp16 d (and dmin for Q4/5)
    __device__ void load(const uint8_t *W, long long rb, int row, bool ok, int k0, int part) {
        const int hh = (k0 / 128) & 1;  // which half of the super-block
        const uint8_t *b = W + (long long)row * rb + (long long)(k0 / 256) * BYTES;
        if (!ok) {
#pragma unroll
            for (int k = 0; k < 16; ++k) q[k] = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) qh[k] = 0;
            sc[0] = sc[1] = sc[2] = 0;
            dd = 0;
            return;
        }
        if (KT == 6) {  // 2-aligned blocks
#pragma unroll
            for (int k = 0; k < 16; ++k) q[k] = ld32_a2(b + 64 * hh + 4 * k);
#pragma unroll
            for (int k = 0; k < 8; ++k) qh[k] = ld32_a2(b + 128 + 32 * hh + 4 * k);
            sc[0] = ld32_a2(b + 192 + 8 * hh);
            sc[1] = ld32_a2(b + 196 + 8 * hh);
            sc[2] = 0;
            dd = *(const uint16_t *)(b + 208);
        } else {  // 4-aligned blocks
            const int c = 2 * hh + part;  // 64-value chunk of the super-block this thread decodes
#pragma unroll
            for (int k = 0; k < 8; ++k) q[k] = ld32(b + (KT == 5 ? 48 : 16) + 32 * c + 4 * k);
#pragma unroll
            for (int k = 8; k < 16; ++k) q[k] = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k) qh[k] = KT == 5 ? ld32(b + 16 + 4 * k) : 0;
#pragma unroll
            for (int k = 0; k < 3; ++k) sc[k] = ld32(b + 4 + 4 * k);
            dd = ld32(b);
        }
    }
    // Decodes this thread's 64 values into srow[64 * part ..], and (part 0) the row's group scales, d and mins
    __device__ void store(int8_t *srow, int *gsc, float *dw, float *dmin, int *mins, int k0, int part) const {
        const int hh = (k0 / 128) & 1;
        uint32_t *dst = (uint32_t *)(srow + 64 * part);
        if (KT == 6) {
            // local values 64 part + 32 j + l: quarter qd = 2 part + j, ql byte l + 32 (qd & 1), nibble qd >> 1
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int qd = 2 * part + j;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const uint32_t qlw = q[(qd & 1) * 8 + k];
                    const uint32_t nib = (qd >> 1) ? (qlw >> 4) & 0x0F0F0F0Fu : qlw & 0x0F0F0F0Fu;
                    const uint32_t hb = (qh[k] >> (2 * qd)) & 0x03030303u;
                    dst[8 * j + k] = sub32_bytes(nib | (hb << 4));
                }
            }
            if (part == 0) {
#pragma unroll
                for (int g = 0; g < 8; ++g) gsc[g] = (int)(int8_t)((sc[g / 4] >> (8 * (g % 4))) & 0xFF);
                *dw = __half2float(__ushort_as_half((uint16_t)dd));
                *dmin = 0.0f;
            }
        } else {
            const int c = 2 * hh + part;
#pragma unroll
            for (int j = 0; j < 2; ++j)  // j = 0: low nibbles (sub-block 2c), 1: high nibbles (2c + 1)
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    uint32_t v = j ? (q[k] >> 4) & 0x0F0F0F0Fu : q[k] & 0x0F0F0F0Fu;
                    if (KT == 5) v |= ((qh[k] >> (2 * c + j)) & 0x01010101u) << 4;
                    dst[8 * j + k] = v;
                }
            uint8_t sb[12];
#pragma unroll
            for (int k = 0; k < 12; ++k) sb[k] = (uint8_t)(sc[k / 4] >> (8 * (k % 4)));
#pragma unroll
            for (int s2 = 0; s2 < 2; ++s2) {  // this part's two sub-blocks 2c, 2c + 1: local groups 4 part + 2 s2 (+1)
                uint8_t s, mm;
                scale_min_k4(2 * c + s2, sb, s, mm);
                gsc[4 * part + 2 * s2] = s;
                gsc[4 * part + 2 * s2 + 1] = s;
                mins[2 * part + s2] = mm;
            }
            if (part == 0) {
                *dw = __half2float(__ushort_as_half((uint16_t)(dd & 0xFFFF)));
                *dmin = __half2float(__ushort_as_half((uint16_t)(dd >> 16)));
            }
        }
    }
};

template <int KT>
__device__ __forceinline__ void wmma_q8k(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8,
                                         const int *bs, float *y, long long ldy, int N, int K, int T) {
    __shared__ __attribute__((aligned(16))) int8_t sw[WM_TILE][WM_KC + 16];
    __shared__ __attribute__((aligned(16))) int8_t sx[WM_TILE][WM_KC + 16];
    __shared__ int sgsc[WM_TILE][8];   // integer scale of each local 16-value group
    __shared__ int smin[WM_TILE][4];   // Q4/5: mins of the chunk's 4 sub-blocks
    __shared__ float sdw[WM_TILE], sdm[WM_TILE];
    __shared__ float sdx[WM_TILE];     // token Q8_K scale of this super-block
    __shared__ int sbs[WM_TILE][4];    // Q4/5: token group sums per sub-block (bs[2 sub] + bs[2 sub + 1])
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int r0 = blockIdx.x * WM_TILE, t0 = blockIdx.y * WM_TILE;
    const int nbk = K / 256;

    float acc[4][8];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[n][r] = 0.0f;

    WStageK<KT> ws;
    uint4 gx[4];
    const int wr_row = tid >> 1, part = tid & 1;
    const int xt0 = tid >> 1, xc = (tid & 1) * 64;
    auto load_chunk = [&](int k0) {
        ws.load(W, row_bytes, r0 + wr_row, r0 + wr_row < N, k0, part);
        const int t = t0 + xt0;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            gx[c] = t < T ? *(const uint4 *)(q8 + (long long)t * K + k0 + xc + 16 * c) : uint4{0, 0, 0, 0};
    };
    auto store_chunk = [&](int k0) {
        ws.store(&sw[wr_row][0], &sgsc[wr_row][0], &sdw[wr_row], &sdm[wr_row], &smin[wr_row][0], k0, part);
#pragma unroll
        for (int c = 0; c < 4; ++c) *(uint4 *)&sx[xt0][xc + 16 * c] = gx[c];
        const int bk = k0 / 256, hh = (k0 / 128) & 1;
        if (tid < 64) {
            const int t = t0 + tid;
            sdx[tid] = t < T ? d8[(long long)t * nbk + bk] : 0.0f;
        } else if (KT != 6) {
            const int tt = tid - 64, t = t0 + tt;
#pragma unroll
            for (int s2 = 0; s2 < 4; ++s2) {
                const int sub = 4 * hh + s2;
                sbs[tt][s2] = t < T ? bs[(long long)t * (K / 16) + bk * 16 + 2 * sub] +
                                          bs[(long long)t * (K / 16) + bk * 16 + 2 * sub + 1]
                                    : 0;
            }
        }
    };

    load_chunk(0);
    const int i = lane & 15, hi = lane >> 4;
    for (int k0 = 0; k0 < K; k0 += WM_KC) {
        store_chunk(k0);
        __syncthreads();
        if (k0 + WM_KC < K) load_chunk(k0 + WM_KC);

        int isum[4][8];
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int r = 0; r < 8; ++r) isum[n][r] = 0;
#pragma unroll 1
        for (int g = 0; g < WM_KC / 16; ++g) {
            const v4i_t a = *(const v4i_t *)&sw[16 * w + i][16 * g];
            v8i_t c[4];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const v4i_t bx = *(const v4i_t *)&sx[16 * n + i][16 * g];
                c[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, bx, v8i_t{}, false);
            }
            int gs[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) gs[r] = sgsc[16 * w + 2 * r + hi][g];
#pragma unroll
            for (int n = 0; n < 4; ++n)
#pragma unroll
                for (int r = 0; r < 8; ++r) isum[n][r] += gs[r] * c[n][r];
        }
        // float step for this chunk
        float dwr[8], dmr[8];
        int mr[8][4];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int rr = 16 * w + 2 * r + hi;
            dwr[r] = sdw[rr];
            dmr[r] = sdm[rr];
#pragma unroll
            for (int s2 = 0; s2 < 4; ++s2) mr[r][s2] = KT == 6 ? 0 : smin[rr][s2];
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int tt = 16 * n + i;
            const float dx = sdx[tt];
            int bsp[4];
#pragma unroll
            for (int s2 = 0; s2 < 4; ++s2) bsp[s2] = KT == 6 ? 0 : sbs[tt][s2];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                int msum = 0;
                if (KT != 6)
#pragma unroll
                    for (int s2 = 0; s2 < 4; ++s2) msum += mr[r][s2] * bsp[s2];
                acc[n][r] += dx * (dwr[r] * (float)isum[n][r] - dmr[r] * (float)msum);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int t = t0 + 16 * n + i;
        if (t >= T) continue;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int row = r0 + 16 * w + 2 * r + hi;
            if (row < N) y[(long long)t * ldy + row] = acc[n][r];
        }
    }
}

extern "C" __global__ void __launch_bounds__(128)
matmul_wmma_q4_K(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y,
                 long long ldy, int N, int K, int T) {
    wmma_q8k<4>(W, row_bytes, q8, d8, bs, y, ldy, N, K, T);
}

extern "C" __global__ void __launch_bounds__(128)
matmul_wmma_q5_K(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y,
                 long long ldy, int N, int K, int T) {
    wmma_q8k<5>(W, row_bytes, q8, d8, bs, y, ldy, N, K, T);
}

extern "C" __global__ void __launch_bounds__(128)
matmul_wmma_q6_K(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y,
                 long long ldy, int N, int K, int T) {
    wmma_q8k<6>(W, row_bytes, q8, d8, bs, y, ldy, N, K, T);
}

// ================================================================ int8 copy of a float LM head (opt-in)
//
// The 35B's LM head is a 1 GB BF16 matrix, 5 ms of every generated token at ~200 GB/s. On request
// (inference_set_lm_head_q8) the engine keeps a Q8_0 copy of it: per 32 values d = amax / 127 (stored as
// FP16) and q = round(x / d), as ggml's quantize_row_q8_0_ref (what llama-quantize writes for a Q8_0 tensor),
// in a split layout: values qs [N][K] and scales d [N][K / 32]. Logits then differ slightly from the
// float head's.

// quantize_rows_q8_0  grid N, block 256: row n of W (F32 = 0, F16 = 1 or BF16 = 30, ggml type ids) -> qs, d.
// Each wave quantizes one 32-value block per step. K % 256 == 0.
extern "C" __global__ void __launch_bounds__(256)
quantize_rows_q8_0(const uint8_t *W, long long row_bytes, int type, int K, int8_t *qs, __half *d) {
    const int n = blockIdx.x, lane = threadIdx.x & 31;
    const uint8_t *wr = W + (long long)n * row_bytes;
    for (int k = threadIdx.x; k < K; k += 256) {
        const float x = type == 0 ? ((const float *)wr)[k] : type == 1 ? fp16(wr + 2 * k) : bf16(wr + 2 * k);
        float am = fabsf(x);
        for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor(am, off, 32));
        const float dd = am / 127.0f, id = dd != 0.0f ? 1.0f / dd : 0.0f;
        qs[(long long)n * K + k] = (int8_t)(int)roundf(x * id);
        if (lane == 0) d[(long long)n * (K / 32) + k / 32] = __float2half(dd);
    }
}

// matvec_q8_0s  grid ceil(N / ROWS_PER_BLOCK), block 32 * ROWS_PER_BLOCK: one wave per row, lane by 32-value
// block; Q8_0 x Q8_0 as ggml's vec_dot_q8_0_q8_0: sum_b (float)isum_b * (d_w * d_x). K % 32 == 0, qs rows
// 16-byte aligned.
extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matvec_q8_0s(const int8_t *qs, const __half *d, const int8_t *q8, const float *d8, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const int nb = K / 32;
    const uint4 *wr = (const uint4 *)(qs + (long long)row * K);
    const __half *dr = d + (long long)row * nb;
    float acc = 0.0f;
#pragma unroll 2
    for (int b = lane; b < nb; b += 32) {
        const uint4 w0 = wr[2 * b], w1 = wr[2 * b + 1];
        const int4 x0 = ((const int4 *)q8)[2 * b], x1 = ((const int4 *)q8)[2 * b + 1];
        int s = 0;
        s = dot4((int)w0.x, x0.x, s);
        s = dot4((int)w0.y, x0.y, s);
        s = dot4((int)w0.z, x0.z, s);
        s = dot4((int)w0.w, x0.w, s);
        s = dot4((int)w1.x, x1.x, s);
        s = dot4((int)w1.y, x1.y, s);
        s = dot4((int)w1.z, x1.z, s);
        s = dot4((int)w1.w, x1.w, s);
        acc += (float)s * (__half2float(dr[b]) * d8[b]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = acc;
}
