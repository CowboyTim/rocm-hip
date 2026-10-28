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

// F16: dedicated kernels instead of the generic one-value-per-lane loop (2-byte loads: 56-142 GB/s on the
// Qwen3.5-0.8B's F16 matrices and tied LM head). Each lane takes chunks of 8 weights (16 bytes) and 8 activations:
// chunks lane, lane + 32, ... of the row, f16_dot8 each, then the wave reduction. The activations are rounded to
// F16 first (round to nearest even): F16's vec_dot_type in ggml is F16, so the reference converts them, and the
// products are then exact in FP32. matmul_f_f16 sums every token exactly so (same chunks per lane, same
// f16_dot8, same reduction): prefill gives decoding's bits. The vector path needs 16-byte-aligned rows and
// activations and K % 8 == 0 (uniform per launch); otherwise one value per lane, in the same order for both.
__device__ __forceinline__ float f16_lo(uint32_t w) { return __half2float(__ushort_as_half((uint16_t)(w & 0xFFFF))); }
__device__ __forceinline__ float f16_hi(uint32_t w) { return __half2float(__ushort_as_half((uint16_t)(w >> 16))); }
__device__ __forceinline__ float f16_round(float v) { return __half2float(__float2half(v)); }

__device__ __forceinline__ float f16_dot8(const uint4 w, const float4 xa, const float4 xb) {  // explicit FMAs
    float s = f16_lo(w.x) * f16_round(xa.x);
    s = fmaf(f16_hi(w.x), f16_round(xa.y), s);
    s = fmaf(f16_lo(w.y), f16_round(xa.z), s);
    s = fmaf(f16_hi(w.y), f16_round(xa.w), s);
    s = fmaf(f16_lo(w.z), f16_round(xb.x), s);
    s = fmaf(f16_hi(w.z), f16_round(xb.y), s);
    s = fmaf(f16_lo(w.w), f16_round(xb.z), s);
    return fmaf(f16_hi(w.w), f16_round(xb.w), s);
}

__device__ __forceinline__ bool f16_vec_ok(const uint8_t *W, long long row_bytes, const float *x, int K) {
    return (((uintptr_t)W | (uintptr_t)row_bytes | (uintptr_t)x) & 15) == 0 && K % 8 == 0;
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matvec_f16(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const uint8_t *wr = W + (long long)row * row_bytes;
    float acc = 0.0f;
    if (f16_vec_ok(W, row_bytes, x, K)) {
        const int n8 = K / 8;
        for (int i = lane; i < n8; i += 32) {
            const uint4 u = ((const uint4 *)wr)[i];
            acc += f16_dot8(u, ((const float4 *)x)[2 * i], ((const float4 *)x)[2 * i + 1]);
        }
    } else {
        for (int k = lane; k < K; k += 32) acc += fp16(wr + 2 * k) * f16_round(x[k]);
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = acc;
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matmul_f_f16(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K, int T) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const int t0 = blockIdx.y * MMF_TT, nt = min(MMF_TT, T - t0);
    const uint8_t *wr = W + (long long)row * row_bytes;
    float acc[MMF_TT];
#pragma unroll
    for (int j = 0; j < MMF_TT; ++j) acc[j] = 0.0f;
    if (f16_vec_ok(W, row_bytes, x, K)) {
        const int n8 = K / 8;
        for (int i = lane; i < n8; i += 32) {
            const uint4 u = ((const uint4 *)wr)[i];
#pragma unroll
            for (int j = 0; j < MMF_TT; ++j)
                if (j < nt) {
                    const float4 *xt = (const float4 *)(x + (size_t)(t0 + j) * K);
                    acc[j] += f16_dot8(u, xt[2 * i], xt[2 * i + 1]);
                }
        }
    } else {
        for (int k = lane; k < K; k += 32) {
            const float w = fp16(wr + 2 * k);
#pragma unroll
            for (int j = 0; j < MMF_TT; ++j)
                if (j < nt) acc[j] += w * f16_round(x[(size_t)(t0 + j) * K + k]);
        }
    }
#pragma unroll
    for (int j = 0; j < MMF_TT; ++j) {
        float a = acc[j];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
        if (lane == 0 && j < nt) y[(size_t)(t0 + j) * N + row] = a;
    }
}
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

// F32 for MF32_TT tokens and MF32_ROWS rows at a time (prefill: the MoE router). grid (ceil(N / MF32_ROWS),
// ceil(T / MF32_TT)), block 128: wave w takes rows 4w..4w+3 of the workgroup's 16, the tokens' activations are
// staged in LDS in chunks (read once per 16 rows instead of once per row). Every (row, token) sum has
// matvec_f32's vector-path partition, expression and order: a wave computes the 4 quarters one after another,
// each lane over i = quarter start + lane + 32 m, reduced over the wave the same way, and adds them in order;
// bit-identical to it. The host guarantees 16-byte-aligned rows and x, and K % 16 == 0.
#define MF32_TT 8
#define MF32_RW 4                      // rows per wave
#define MF32_ROWS (MF32_RW * F32_PARTS)  // rows per workgroup (4 waves)
#define MF32_CH 256                    // float4 per token per LDS chunk (a multiple of 32)
extern "C" __global__ void __launch_bounds__(32 * F32_PARTS)
matmul_f32(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K, int T) {
    __shared__ float4 sx[MF32_TT][MF32_CH];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int row0 = blockIdx.x * MF32_ROWS + w * MF32_RW;
    const int t0 = blockIdx.y * MF32_TT, nt = min(MF32_TT, T - t0);
    const int n4 = K / 4, q = n4 / F32_PARTS;
    const float4 *x4 = (const float4 *)x;
    const float4 *wr[MF32_RW];
#pragma unroll
    for (int r = 0; r < MF32_RW; ++r)
        wr[r] = (const float4 *)(W + (long long)min(row0 + r, N - 1) * row_bytes);  // rows >= N: not stored
    float res[MF32_RW][MF32_TT];
    for (int p = 0; p < F32_PARTS; ++p) {
        float acc[MF32_RW][MF32_TT];
#pragma unroll
        for (int r = 0; r < MF32_RW; ++r)
#pragma unroll
            for (int t = 0; t < MF32_TT; ++t) acc[r][t] = 0.0f;
        for (int c0 = p * q; c0 < (p + 1) * q; c0 += MF32_CH) {
            const int cn = min(MF32_CH, (p + 1) * q - c0);
            __syncthreads();  // the previous chunk is consumed
            for (int idx = tid; idx < MF32_TT * MF32_CH; idx += 32 * F32_PARTS) {
                const int t = idx / MF32_CH, k = idx % MF32_CH;
                sx[t][k] = t < nt && k < cn ? x4[(size_t)(t0 + t) * n4 + c0 + k] : float4{0, 0, 0, 0};
            }
            __syncthreads();
            for (int k = lane; k < cn; k += 32) {
                float4 wv[MF32_RW];
#pragma unroll
                for (int r = 0; r < MF32_RW; ++r) wv[r] = wr[r][c0 + k];
#pragma unroll
                for (int t = 0; t < MF32_TT; ++t) {
                    const float4 xv = sx[t][k];
#pragma unroll
                    for (int r = 0; r < MF32_RW; ++r) acc[r][t] += dot4f(wv[r], xv);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < MF32_RW; ++r)
#pragma unroll
            for (int t = 0; t < MF32_TT; ++t) {
                float a = acc[r][t];
                for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
                res[r][t] = p == 0 ? a : res[r][t] + a;
            }
    }
    if (lane != 0) return;
#pragma unroll
    for (int r = 0; r < MF32_RW; ++r)
#pragma unroll
        for (int t = 0; t < MF32_TT; ++t)
            if (row0 + r < N && t < nt) y[(size_t)(t0 + t) * N + row0 + r] = res[r][t];
}

// F32 prefill (the MoE router's logits) when the summation order may differ from matvec_f32 (the WMMA mode,
// as the int8 WMMA kernels): a register-blocked tile of FT_ROWS rows x FT_TOK tokens per 256-thread workgroup,
// thread (ty, tx) computing rows 4 ty .. + 3 x tokens 4 tx .. + 3. Per chunk of FT_KC values both operands are
// staged in LDS transposed ([k][row], [k][token]) so that a thread reads its 4 rows / 4 tokens as one float4;
// each chunk's 16 dot products are summed with FMAs in k order and then added to the accumulators (fewer
// roundings of large partial sums). Each weight is read once per 64 tokens and each activation once per 64
// rows: matmul_f32 (16 rows x 8 tokens, matvec_f32's order) moved ~5x the bytes through L2. K % 32 == 0.
#define FT_ROWS 64
#define FT_TOK 64
#define FT_KC 32
extern "C" __global__ void __launch_bounds__(256)
matmul_f32_tile(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K, int T) {
    __shared__ __attribute__((aligned(16))) float sw[FT_KC][FT_ROWS + 4];
    __shared__ __attribute__((aligned(16))) float sx[FT_KC][FT_TOK + 4];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int r0 = blockIdx.x * FT_ROWS, t0 = blockIdx.y * FT_TOK;
    float acc[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[r][c] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += FT_KC) {
        // stage: 64 rows / tokens x 32 values = 512 float4 each, 2 per thread
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int idx = tid + 256 * j, rr = idx / (FT_KC / 4), k4 = 4 * (idx % (FT_KC / 4));
            const float4 wv = r0 + rr < N ? *(const float4 *)((const float *)(W + (long long)(r0 + rr) * row_bytes) +
                                                                k0 + k4)
                                          : float4{0, 0, 0, 0};
            const float4 xv = t0 + rr < T ? *(const float4 *)(x + (size_t)(t0 + rr) * K + k0 + k4) : float4{0, 0, 0, 0};
            sw[k4][rr] = wv.x, sw[k4 + 1][rr] = wv.y, sw[k4 + 2][rr] = wv.z, sw[k4 + 3][rr] = wv.w;
            sx[k4][rr] = xv.x, sx[k4 + 1][rr] = xv.y, sx[k4 + 2][rr] = xv.z, sx[k4 + 3][rr] = xv.w;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) s[r][c] = 0.0f;
#pragma unroll 8
        for (int k = 0; k < FT_KC; ++k) {
            const float4 wv = *(const float4 *)&sw[k][4 * ty], xv = *(const float4 *)&sx[k][4 * tx];
            const float wr[4] = {wv.x, wv.y, wv.z, wv.w}, xc[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
                for (int c = 0; c < 4; ++c) s[r][c] = fmaf(wr[r], xc[c], s[r][c]);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[r][c] += s[r][c];
        __syncthreads();
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int t = t0 + 4 * tx + c;
        if (t >= T) continue;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = r0 + 4 * ty + r;
            if (row < N) y[(size_t)t * N + row] = acc[r][c];
        }
    }
}

// One lane's share of a BF16 row's dot product with x (the wave reduction follows): 16-byte chunks lane,
// lane + 32, ... when the row and x are 16-byte aligned (the same for every lane), then the rest one value per
// lane. Every BF16 kernel sums a (row, token) this way, so prefill and decoding give the same bits.
__device__ __forceinline__ float bf16_lane_dot(const uint8_t *wr, const float *x, int K, int lane) {
    float acc = 0.0f;
    int k0 = 0;
    if ((((uintptr_t)wr | (uintptr_t)x) & 15) == 0) {
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
    return acc;
}

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matvec_bf16(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    float acc = bf16_lane_dot(W + (long long)row * row_bytes, x, K, lane);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[row] = acc;
}

// Several tokens: grid (ceil(N / ROWS_PER_BLOCK), ceil(T / MMF_TT)), each wave's row read once per tile (from
// cache for all but the first token: the lanes' chunks are the same) and every token summed as matvec_bf16
extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matmul_f_bf16(const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K, int T) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const int t0 = blockIdx.y * MMF_TT, nt = min(MMF_TT, T - t0);
    const uint8_t *wr = W + (long long)row * row_bytes;
    for (int j = 0; j < nt; ++j) {
        float a = bf16_lane_dot(wr, x + (size_t)(t0 + j) * K, K, lane);
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
        if (lane == 0) y[(size_t)(t0 + j) * N + row] = a;
    }
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

// BF16 experts: as matvec_bf16 (bf16_lane_dot), so that the expert-tiled batch below gives decoding's bits
extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK)
matvec_id_bf16(const uint8_t *W, long long row_bytes, const int *ids, int n_used, int x_per_slot, const float *x,
               float *y, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= N) return;
    const int slot = blockIdx.y, t = blockIdx.z, e = ids[t * n_used + slot];
    const float *xr = x + (x_per_slot ? (long long)t * n_used + slot : t) * K;
    float acc = bf16_lane_dot(W + ((long long)e * N + row) * row_bytes, xr, K, lane);
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (lane == 0) y[((long long)t * n_used + slot) * N + row] = acc;
}

// Float experts for a batch (MoE prefill on the float path): grid (ceil(N / ROWS_PER_BLOCK), max_tiles) over
// moe_group's tiles of up to MMF_ID_TB (token, slot) entries of one expert (the int8 matmul_q8_id_<type>
// convention: x rows in_rows[], y rows out_rows[]), so each expert row is read once per tile; each entry is
// summed as matvec_id_<type> sums it.
#define MMF_ID_TB 16  // QMV_MM_TB in qmatvec.cpp
#define MMF_ID_ARGS                                                                                         \
    const uint8_t *W, long long row_bytes, const float *x, float *y, int N, int K, const int *n_tiles,          \
        const int *tile_e, const int *tile_start, const int *tile_n, const int *in_rows, const int *out_rows

template <typename B>
__device__ void matmul_f_id_impl(MMF_ID_ARGS) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5), k = blockIdx.y;
    if (row >= N || k >= *n_tiles) return;
    const int nt = tile_n[k], *in = in_rows + tile_start[k], *out = out_rows + tile_start[k];
    const uint8_t *wr = W + ((long long)tile_e[k] * N + row) * row_bytes;
    const int nb = K / B::QK;
    float acc[MMF_ID_TB];
#pragma unroll
    for (int j = 0; j < MMF_ID_TB; ++j) acc[j] = 0.0f;
    for (int ib = lane; ib < nb; ib += 32) {
        const uint8_t *wb = wr + (long long)ib * B::BYTES;
#pragma unroll
        for (int j = 0; j < MMF_ID_TB; ++j)
            if (j < nt) acc[j] += B::dot(wb, x + (long long)in[j] * K + (long long)ib * B::QK);
    }
#pragma unroll
    for (int j = 0; j < MMF_ID_TB; ++j) {
        float a = acc[j];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
        if (lane == 0 && j < nt) y[(long long)out[j] * N + row] = a;
    }
}

#define MATMUL_F_ID_KERNEL(name, B)                                                                       \
    extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matmul_f_id_##name(MMF_ID_ARGS) {   \
        matmul_f_id_impl<B>(W, row_bytes, x, y, N, K, n_tiles, tile_e, tile_start, tile_n, in_rows, out_rows); \
    }

MATMUL_F_ID_KERNEL(f32, BlkF32)
MATMUL_F_ID_KERNEL(f16, BlkF16)
MATMUL_F_ID_KERNEL(q4_0, BlkQ4_0)
MATMUL_F_ID_KERNEL(q4_1, BlkQ4_1)
MATMUL_F_ID_KERNEL(q5_0, BlkQ5_0)
MATMUL_F_ID_KERNEL(q5_1, BlkQ5_1)
MATMUL_F_ID_KERNEL(q8_0, BlkQ8_0)
MATMUL_F_ID_KERNEL(q2_K, BlkQ2_K)
MATMUL_F_ID_KERNEL(q3_K, BlkQ3_K)
MATMUL_F_ID_KERNEL(q4_0_rocmfp4, BlkROCmFP4)
MATMUL_F_ID_KERNEL(q4_0_rocmfp4_fast, BlkROCmFP4Fast)

extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matmul_f_id_bf16(MMF_ID_ARGS) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROWS_PER_BLOCK + (threadIdx.x >> 5), k = blockIdx.y;
    if (row >= N || k >= *n_tiles) return;
    const int nt = tile_n[k], *in = in_rows + tile_start[k], *out = out_rows + tile_start[k];
    const uint8_t *wr = W + ((long long)tile_e[k] * N + row) * row_bytes;
    for (int j = 0; j < nt; ++j) {
        float a = bf16_lane_dot(wr, x + (long long)in[j] * K, K, lane);
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
        if (lane == 0) y[(long long)out[j] * N + row] = a;
    }
}

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

// The 18-byte 4-bit block types, read as 4-aligned pairs of blocks (36 bytes, 9 dwords; K % 64 == 0): values j
// (0..15) in the low nibbles of the 16 value bytes, j + 16 in the high ones. A policy P gives pair() (the two
// blocks' value dwords and scales from the 9 dwords), dec4() (four nibbles -> four int8 values) and block() (a
// block's float sum from its two half-block int dots), so every kernel family is written once for both:
//  ROCmFP4: 16 value bytes, then two UE4M3 scales (one per 16 values); FP4 codebook values.
//  Q4_0: an F16 scale, then 16 value bytes; values nibble - 8. As ggml's vec_dot_q4_0_q8_0: (d_w d_x) * sumi.
struct P4ROCmFP4 {
    static constexpr bool BLOCK_SCALE = false;  // (WMMA: a scale per half-block)
    __device__ static void pair(const uint32_t w[9], uint32_t qs[2][4], float e[2][2]) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            qs[0][k] = w[k];
            qs[1][k] = (w[4 + k] >> 16) | (w[5 + k] << 16);
        }
        e[0][0] = ue4m3_half((uint8_t)(w[4] & 0xFF));
        e[0][1] = ue4m3_half((uint8_t)((w[4] >> 8) & 0xFF));
        e[1][0] = ue4m3_half((uint8_t)((w[8] >> 16) & 0xFF));
        e[1][1] = ue4m3_half((uint8_t)(w[8] >> 24));
    }
    __device__ static int dec4(uint32_t n) { return rocmfp4_dec4(n); }
    __device__ static float block(int s0, int s1, const float e[2], float dx) {
        return e[0] * dx * (float)s0 + e[1] * dx * (float)s1;
    }
};

struct P4Q4_0 {
    static constexpr bool BLOCK_SCALE = true;
    __device__ static void pair(const uint32_t w[9], uint32_t qs[2][4], float e[2][2]) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            qs[0][k] = __builtin_amdgcn_alignbyte(w[k + 1], w[k], 2);  // block 0's values: bytes 2..17
            qs[1][k] = w[5 + k];                                       // block 1's: bytes 20..35
        }
        e[0][0] = e[0][1] = __half2float(__ushort_as_half((uint16_t)(w[0] & 0xFFFF)));
        e[1][0] = e[1][1] = __half2float(__ushort_as_half((uint16_t)(w[4] >> 16)));
    }
    // nibbles 0..15 -> -8..7 per byte (no borrow between bytes: each is >= 0x80 before the subtraction)
    __device__ static int dec4(uint32_t n) { return (int)(((n | 0x80808080u) - 0x08080808u) ^ 0x80808080u); }
    __device__ static float block(int s0, int s1, const float e[2], float dx) { return e[0] * dx * (float)(s0 + s1); }
};

template <class P>
__device__ __forceinline__ void p4_pair(const uint8_t *b, uint32_t qs[2][4], float e[2][2]) {
    uint32_t w[9];
#pragma unroll
    for (int k = 0; k < 9; ++k) w[k] = ld32(b + 4 * k);
    P::pair(w, qs, e);
}

// One block (its value dwords and scales) against its Q8_0 block
template <class P>
__device__ __forceinline__ float p4_q8_block(const uint32_t qs[4], const float e[2], const int8_t *q8, float dx) {
    const int *x8 = (const int *)q8;
    int s0 = 0, s1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        s0 = dot4(P::dec4(qs[k] & 0x0F0F0F0Fu), x8[k], s0);
        s1 = dot4(P::dec4((qs[k] >> 4) & 0x0F0F0F0Fu), x8[4 + k], s1);
    }
    return P::block(s0, s1, e, dx);
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

// ROCmFP4 / Q4_0 x Q8_0: a lane takes a pair of 18-byte blocks (36 bytes, 4-aligned; the host requires K % 64 == 0)
template <class P>
__device__ __forceinline__ float mv_p4(const uint8_t *wr, const int8_t *q8, const float *d8, int K, int lane, int L) {
    const int npairs = K / 64;
    float acc = 0.0f;
    for (int p = lane; p < npairs; p += L) {
        uint32_t qs[2][4];
        float e[2][2];
        p4_pair<P>(wr + (long long)p * 36, qs, e);
        const int ib = 2 * p;
        acc += p4_q8_block<P>(qs[0], e[0], q8 + ib * 32, d8[ib]);
        acc += p4_q8_block<P>(qs[1], e[1], q8 + (ib + 1) * 32, d8[ib + 1]);
    }
    return acc;
}

__device__ __forceinline__ float mv_rocmfp4(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                            int lane, int L) {
    return mv_p4<P4ROCmFP4>(wr, q8, d8, K, lane, L);
}

__device__ __forceinline__ float mv_q4_0(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                         int lane, int L) {
    return mv_p4<P4Q4_0>(wr, q8, d8, K, lane, L);
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

// Q8_0 (34-byte blocks: an F16 scale, then 32 int8 values) x Q8_0: a lane takes a pair of blocks (68 bytes,
// 4-aligned; the host requires K % 64 == 0). Of the pair's 17 dwords, block 0's scale is the low half of w[0] and
// its values bytes 2..33 (v_alignbyte), block 1's scale the high half of w[8] and its values w[9..16]. As ggml's
// vec_dot_q8_0_q8_0: (d_w * d_x) * (float)sumi per block.
__device__ __forceinline__ void q8_0_pair(const uint8_t *b, uint32_t qs[2][8], float e[2]) {
    uint32_t w[17];
#pragma unroll
    for (int k = 0; k < 17; ++k) w[k] = ld32(b + 4 * k);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        qs[0][k] = __builtin_amdgcn_alignbyte(w[k + 1], w[k], 2);
        qs[1][k] = w[9 + k];
    }
    e[0] = __half2float(__ushort_as_half((uint16_t)(w[0] & 0xFFFF)));
    e[1] = __half2float(__ushort_as_half((uint16_t)(w[8] >> 16)));
}

__device__ __forceinline__ float q8_0_q8_block(const uint32_t qs[8], float e, const int8_t *q8, float dx) {
    const int *x8 = (const int *)q8;
    int s = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) s = dot4((int)qs[k], x8[k], s);
    return e * dx * (float)s;
}

// L lanes per row: lane l sums pairs l, l + L, ...; the host only takes L < 32 when every lane has at most one
// pair, so the sums are those of the batched kernel's 32 lanes (plus exact zeros)
__device__ __forceinline__ float mv_q8_0(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                         int lane, int L) {
    (void)bs;
    const int npairs = K / 64;
    float acc = 0.0f;
    for (int p = lane; p < npairs; p += L) {
        uint32_t qs[2][8];
        float e[2];
        q8_0_pair(wr + (long long)p * 68, qs, e);
        const int ib = 2 * p;
        acc += q8_0_q8_block(qs[0], e[0], q8 + ib * 32, d8[ib]);
        acc += q8_0_q8_block(qs[1], e[1], q8 + (ib + 1) * 32, d8[ib + 1]);
    }
    return acc;
}

// Q5_1 (24-byte blocks, 4-aligned: F16 d, F16 m, the 32 fifth bits qh, 16 bytes of nibbles) x Q8_0: values
// j (0..15) are the low nibbles with bit j of qh as bit 4, values j + 16 the high nibbles with bit j + 16; each
// q (0..31) * d + m. As ggml's vec_dot_q5_1_q8_1: (d_w * d_x) * (float)sumi + m_w * (d_x * sum of the block's
// x q) per block, that product as ggml's Q8_1 s from the quantizer (qmatvec_q8.s0: F16(d * sum), d unrounded).
__device__ __forceinline__ uint32_t q5_1_hbits(uint32_t h) {  // 4 bits -> bit 4 of 4 bytes
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

__device__ __forceinline__ void q5_1_decode(const uint32_t w[6], uint32_t q[8], float &d, float &m) {
    d = __half2float(__ushort_as_half((uint16_t)(w[0] & 0xFFFF)));
    m = __half2float(__ushort_as_half((uint16_t)(w[0] >> 16)));
    const uint32_t qh = w[1];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        q[k] = (w[2 + k] & 0x0F0F0F0Fu) | q5_1_hbits(qh >> (4 * k));
        q[4 + k] = ((w[2 + k] >> 4) & 0x0F0F0F0Fu) | q5_1_hbits(qh >> (16 + 4 * k));
    }
}

// The 32-value block types read a block per lane: Q5_1, Q4_1 (an offset m per block: MIN) and Q5_0. A policy gives
// BYTES, load() (the block as DW dwords) and decode() (its 32 values as int8 bytes, d and m), so every kernel family
// is written once for all three:
//  Q4_1: 20 bytes (4-aligned): F16 d, F16 m, 16 bytes of nibbles; values q (0..15) * d + m (ggml: as Q5_1's).
//  Q5_0: 22 bytes (2-aligned): F16 d, the fifth bits qh, 16 bytes of nibbles; values (q - 16) * d, q 0..31 as Q5_1's.
//        As ggml's vec_dot_q5_0_q8_0: (d_w * d_x) * (float)sumi.
struct PBQ5_1 {
    static constexpr int BYTES = 24, DW = 6;
    static constexpr bool MIN = true;
    __device__ static void load(const uint8_t *b, uint32_t w[DW]) {
#pragma unroll
        for (int k = 0; k < DW; ++k) w[k] = ld32(b + 4 * k);
    }
    __device__ static void decode(const uint32_t w[DW], uint32_t q[8], float &d, float &m) { q5_1_decode(w, q, d, m); }
};

struct PBQ4_1 {
    static constexpr int BYTES = 20, DW = 5;
    static constexpr bool MIN = true;
    __device__ static void load(const uint8_t *b, uint32_t w[DW]) {
#pragma unroll
        for (int k = 0; k < DW; ++k) w[k] = ld32(b + 4 * k);
    }
    __device__ static void decode(const uint32_t w[DW], uint32_t q[8], float &d, float &m) {
        d = __half2float(__ushort_as_half((uint16_t)(w[0] & 0xFFFF)));
        m = __half2float(__ushort_as_half((uint16_t)(w[0] >> 16)));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            q[k] = w[1 + k] & 0x0F0F0F0Fu;
            q[4 + k] = (w[1 + k] >> 4) & 0x0F0F0F0Fu;
        }
    }
};

struct PBQ5_0 {
    static constexpr int BYTES = 22, DW = 6;
    static constexpr bool MIN = false;
    __device__ static void load(const uint8_t *b, uint32_t w[DW]) {  // w: d, qh, the 4 nibble dwords
        w[0] = *(const uint16_t *)b;
        w[1] = ld32_a2(b + 2);
#pragma unroll
        for (int k = 0; k < 4; ++k) w[2 + k] = ld32_a2(b + 6 + 4 * k);
    }
    __device__ static void decode(const uint32_t w[DW], uint32_t q[8], float &d, float &m) {
        uint32_t u[6] = {w[0], w[1], w[2], w[3], w[4], w[5]};
        q5_1_decode(u, q, d, m);  // (its m: the high half of w[0], zero here)
#pragma unroll
        for (int k = 0; k < 8; ++k) q[k] = ((q[k] | 0x80808080u) - 0x10101010u) ^ 0x80808080u;  // q - 16 per byte
        m = 0.0f;
    }
};

// One block (its values, d, m) against its Q8_0 block (dx, sx: its scale and the sum of its q)
template <class P>
__device__ __forceinline__ float pb_q8_block(const uint32_t q[8], float d, float m, const int8_t *q8, float dx, int sx) {
    const int *x8 = (const int *)q8;
    int s = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) s = dot4((int)q[k], x8[k], s);
    if constexpr (P::MIN) return d * dx * (float)s + m * __int_as_float(sx);  // sx: Q8_1's s (qmatvec_q8.s0)
    else return d * dx * (float)s;
}

template <class P>
__device__ __forceinline__ void pb_block(const uint8_t *b, uint32_t q[8], float &d, float &m) {
    uint32_t w[P::DW];
    P::load(b, w);
    P::decode(w, q, d, m);
}

// bs: the Q8_0 block sums (s0). L lanes per row, a block per lane and step, as mv_q8_0
template <class P>
__device__ __forceinline__ float mv_pb(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                       int lane, int L) {
    const int nb = K / 32;
    float acc = 0.0f;
    for (int ib = lane; ib < nb; ib += L) {
        uint32_t q[8];
        float d, m;
        pb_block<P>(wr + (long long)ib * P::BYTES, q, d, m);
        acc += pb_q8_block<P>(q, d, m, q8 + ib * 32, d8[ib], bs[ib]);
    }
    return acc;
}

__device__ __forceinline__ float mv_q5_1(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                         int lane, int L) {
    return mv_pb<PBQ5_1>(wr, q8, d8, bs, K, lane, L);
}
__device__ __forceinline__ float mv_q4_1(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                         int lane, int L) {
    return mv_pb<PBQ4_1>(wr, q8, d8, bs, K, lane, L);
}
__device__ __forceinline__ float mv_q5_0(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                         int lane, int L) {
    return mv_pb<PBQ5_0>(wr, q8, d8, bs, K, lane, L);
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
        float acc = FN(wr, q8 + in_row * K, d8 + in_row * (K / ((KQ) == 1 ? 256 : 32)),                \
                       bs + in_row * (K / ((KQ) == 2 ? 32 : 16)), K, gl, L);                             \
        Q8_REDUCE(acc)                                                                                   \
        if (gl == 0) (m2 ? y2 : y)[((long long)t * n_used + slot) * N + row] = acc;                      \
    }

// The same for NT tokens (2..4): each lane decodes its quads once and applies them to every token's Q8_0 input
// (q8[t] / d8[t]), keeping each token's sums exactly as mv_rocmfp4_fast does, so the
// results are those of NT single-token launches (speculative decoding's verify pass) while the weights are
// read once.
template <int NT>
__device__ __forceinline__ void mv_rocmfp4_fast_tp(const uint8_t *wr, const int8_t *const q8[NT],
                                                   const float *const d8[NT], int K, int lane, int L, float out[NT]) {
    const int nquad = K / 128, V = 32 / L;
    float a[NT][8];
#pragma unroll
    for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int k = 0; k < 8; ++k) a[t][k] = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k >= V) continue;
        for (int p = lane + k * L; p < nquad; p += 32) {
            uint32_t qs[4][4];
            float e[4];
            rocmfp4_fast_quad(wr + (long long)p * 68, qs, e);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int ib = 4 * p + j;
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    a[t][k] += rocmfp4_fast_q8_block(qs[j], e[j], q8[t] + ib * 32, d8[t][ib]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int d = 4; d >= 1; d >>= 1)
            if (d < V)
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    if (k < d) a[t][k] += a[t][k + d];
        out[t] = a[t][0];
    }
}

// Token t's input at rows t * K of q8 / t * K / 32 of d8
template <int NT>
__device__ __forceinline__ void mv_rocmfp4_fast_t(const uint8_t *wr, const int8_t *q8, const float *d8, int K,
                                                  int lane, int L, float out[NT]) {
    const int8_t *q[NT];
    const float *d[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) q[t] = q8 + (long long)t * K, d[t] = d8 + (long long)t * (K / 32);
    mv_rocmfp4_fast_tp<NT>(wr, q, d, K, lane, L, out);
}

template <int NT>
__device__ __forceinline__ void matvec_q8t_fast_body(Q8_ARGS) {
    Q8_ROW_SETUP
    const int nm = mat == 0 ? N1 : mat == 1 ? N2 - N1 : mat == 2 ? N3 - N2 : N - N3;  // this matrix's rows
    float acc[NT];
    mv_rocmfp4_fast_t<NT>(wr, q8, d8, K, gl, L, acc);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
        Q8_REDUCE(v)
        if (gl == 0) yr[(long long)t * nm] = v;
    }
}

// Q6_K for NT tokens: mv_q6_K with every token's sums kept as there (Q8_K inputs: rows t * K of q8, t * K / 256
// of d8, t * K / 16 of bs)
template <int NT>
__device__ __forceinline__ void mv_q6_K_t(const uint8_t *wr, const int8_t *q8, const float *d8, const int *bs, int K,
                                          int lane, int L, float out[NT]) {
    const int sub = lane >> 3, s = lane & 7;
    const int h = s >> 2, r = (s & 3) * 16, l = r & 31;
    const bool second = r >= 32;
    const int sh_lo = second ? 2 : 0;
    const int v_lo = 128 * h + (second ? 32 : 0) + l, v_hi = v_lo + 64;
    const int g_lo = 8 * h + l / 16 + (second ? 2 : 0), g_hi = g_lo + 4;
    const int nb = K / 256;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int ib = sub; ib < nb; ib += L / 8) {
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
        const float dw = fp16(b + 208);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t *xb = q8 + (long long)t * K + (long long)ib * 256;
            const int *bst = bs + (long long)t * (K / 16);
            int slo = 0, shi = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                slo = dot4(vlo[k], *(const int *)(xb + v_lo + 4 * k), slo);
                shi = dot4(vhi[k], *(const int *)(xb + v_hi + 4 * k), shi);
            }
            int isum = sc[g_lo] * slo + sc[g_hi] * shi;
            isum -= 32 * (sc[2 * s] * bst[ib * 16 + 2 * s] + sc[2 * s + 1] * bst[ib * 16 + 2 * s + 1]);
            acc[t] += dw * d8[(long long)t * (K / 256) + ib] * (float)isum;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) out[t] = acc[t];
}

template <int NT>
__device__ __forceinline__ void matvec_q8t_q6_K_body(Q8_ARGS) {
    Q8_ROW_SETUP
    const int nm = mat == 0 ? N1 : mat == 1 ? N2 - N1 : mat == 2 ? N3 - N2 : N - N3;
    float acc[NT];
    mv_q6_K_t<NT>(wr, q8, d8, bs, K, gl, L, acc);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
        Q8_REDUCE(v)
        if (gl == 0) yr[(long long)t * nm] = v;
    }
}

// matvec_q8t_<type>(Q8_ARGS, T): matvec_q8_<type> for T = 2..4 tokens (input rows t of q8; output row t of each
// matrix's y at y + t * its rows)
#define Q8T_ARGS W, W2, W3, W4, N1, N2, N3, row_bytes, q8, d8, bs, y, y2, y3, y4, N, K, L
extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matvec_q8t_q4_0_rocmfp4_fast(Q8_ARGS, int T) {
    if (T == 2) matvec_q8t_fast_body<2>(Q8T_ARGS);
    else if (T == 3) matvec_q8t_fast_body<3>(Q8T_ARGS);
    else matvec_q8t_fast_body<4>(Q8T_ARGS);
}
extern "C" __global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) matvec_q8t_q6_K(Q8_ARGS, int T) {
    if (T == 2) matvec_q8t_q6_K_body<2>(Q8T_ARGS);
    else if (T == 3) matvec_q8t_q6_K_body<3>(Q8T_ARGS);
    else matvec_q8t_q6_K_body<4>(Q8T_ARGS);
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
MATVEC_Q8_KERNELS(q4_0, mv_q4_0, 0)
MATVEC_Q8_KERNELS(q4_0_rocmfp4_fast, mv_rocmfp4_fast, 0)
MATVEC_Q8_KERNELS(q8_0, mv_q8_0, 0)
MATVEC_Q8_KERNELS(q5_1, mv_q5_1, 2)
MATVEC_Q8_KERNELS(q4_1, mv_q4_1, 2)
MATVEC_Q8_KERNELS(q5_0, mv_q5_0, 2)
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

// ROCmFP4 / Q4_0 (P: the 18-byte block policy)
template <class P, class RM>
__device__ __forceinline__ void mmb_p4(MM_BARGS, const RM &rm) {
    (void)bs;
    MM_ROW_SETUP
    const int npairs = K / 64, nb = K / 32;
    for (int p = lane; p < npairs; p += 32) {
        uint32_t qs[2][4];
        float e[2][2];
        p4_pair<P>(wr + (long long)p * 36, qs, e);
        int lo[2][4], hi[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[h][k] = P::dec4(qs[h][k] & 0x0F0F0F0Fu);
                hi[h][k] = P::dec4((qs[h][k] >> 4) & 0x0F0F0F0Fu);
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
                acc[j] += P::block(s0, s1, e[h], d8[t * nb + ib]);
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

template <class RM>
__device__ __forceinline__ void mmb_q8_0(MM_BARGS, const RM &rm) {
    (void)bs;
    MM_ROW_SETUP
    const int npairs = K / 64, nb = K / 32;
    for (int p = lane; p < npairs; p += 32) {
        uint32_t qs[2][8];
        float e[2];
        q8_0_pair(wr + (long long)p * 68, qs, e);
        // Not unrolled: see matmul_q8_q4_0_rocmfp4
        for (int j = 0; j < MM_TB; ++j) {
            if (j >= nt) break;
            const long long t = rm.in(j);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int ib = 2 * p + h;
                acc[j] += q8_0_q8_block(qs[h], e[h], q8 + t * K + ib * 32, d8[t * nb + ib]);
            }
        }
    }
    MM_STORE
}

// Q5_1 / Q4_1 / Q5_0 (P: the block-per-lane policy). bs: the Q8_0 block sums (s0), [T][K / 32]
template <class P, class RM>
__device__ __forceinline__ void mmb_pb(MM_BARGS, const RM &rm) {
    MM_ROW_SETUP
    const int nb = K / 32;
    for (int ib = lane; ib < nb; ib += 32) {
        uint32_t q[8];
        float d, m;
        pb_block<P>(wr + (long long)ib * P::BYTES, q, d, m);
        // Not unrolled: see matmul_q8_q4_0_rocmfp4
        for (int j = 0; j < MM_TB; ++j) {
            if (j >= nt) break;
            const long long t = rm.in(j);
            acc[j] += pb_q8_block<P>(q, d, m, q8 + t * K + ib * 32, d8[t * nb + ib], bs[t * nb + ib]);
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

MATMUL_Q8_KERNELS(q4_0_rocmfp4, (mmb_p4<P4ROCmFP4, RowsLinear>), (mmb_p4<P4ROCmFP4, RowsExpert>))
MATMUL_Q8_KERNELS(q4_0, (mmb_p4<P4Q4_0, RowsLinear>), (mmb_p4<P4Q4_0, RowsExpert>))
MATMUL_Q8_KERNELS(q4_0_rocmfp4_fast, mmb_rocmfp4_fast<RowsLinear>, mmb_rocmfp4_fast<RowsExpert>)
MATMUL_Q8_KERNELS(q8_0, mmb_q8_0<RowsLinear>, mmb_q8_0<RowsExpert>)
MATMUL_Q8_KERNELS(q5_1, (mmb_pb<PBQ5_1, RowsLinear>), (mmb_pb<PBQ5_1, RowsExpert>))
MATMUL_Q8_KERNELS(q4_1, (mmb_pb<PBQ4_1, RowsLinear>), (mmb_pb<PBQ4_1, RowsExpert>))
MATMUL_Q8_KERNELS(q5_0, (mmb_pb<PBQ5_0, RowsLinear>), (mmb_pb<PBQ5_0, RowsExpert>))
MATMUL_Q8_KERNELS(q6_K, mmb_q6_K<RowsLinear>, mmb_q6_K<RowsExpert>)
MATMUL_Q8_KERNELS(q4_K, mmb_q4_K_lin, mmb_q4_K_exp)
MATMUL_Q8_KERNELS(q5_K, mmb_q5_K_lin, mmb_q5_K_exp)

// ================================================================ WMMA tiled matmul (prefill, Phase 20 W2)
//
// Y[t][n] = sum_k W[n][k] X[t][k] for T tokens with v_wmma_i32_16x16x16_iu8 (55.5 TOPS on gfx1151, vs 29 for
// v_dot4). A workgroup of 8 waves computes a 128 weight-row x 64 token tile (WM_ROWS; the K-quant kernels below:
// 4 waves, 64 rows); wave w owns rows 16w..16w+15 and all 64 tokens (4 float accumulator tiles of 16 x 16). 128
// rows read each token chunk for twice the rows (experts 8-10% faster than 64). Per 128-value K chunk the
// workgroup decodes the rows' weights to int8 into LDS once (with their scales) and copies the 64 tokens' Q8_0
// bytes and scales into LDS; each 16-value step is then one WMMA per 16 x 16 sub-tile, an exact int32 dot over one ROCmFP4
// half-block, accumulated as (float)dot * e_w[row] * d_x[token]. ROCmFP4_FAST (one scale per 32-value block)
// adds a block's two dots in int32 first and scales once, (e_w * d_x) * (float)dot, the dot4 kernels' per-block
// term; for it the tokens are operand A, so that a lane's results share its weight row's scale and a token's
// outputs are stored as 16 consecutive floats. The same integer sums and scale products as ggml; only the
// float accumulation order differs from the dot4 kernels.
// WMMA wave32 layout (tests/wmma_layout_verify): A row i / B column i in lanes i and i + 16, 16 int8 each;
// C column l % 16, row 2r + l / 16.
// Q5_1 (BLOCK_SCALE with MIN): the block's offset m adds m * d_x * (sum of the block's x q) per block, from the
// quantizer's block sums. grid (ceil(N / WM_ROWS), ceil(T / 64)), block WM_THREADS. K % 128 == 0, or K % 64 == 0
// for the policies of 2 blocks per thread (all but ROCmFP4_FAST): the last chunk is then half, the rest zeros.

typedef int v4i_t __attribute__((ext_vector_type(4)));
typedef int v8i_t __attribute__((ext_vector_type(8)));

#define WM_TILE 64
#define WM_KC 128

// Weight staging policies for the Q8_0-activation WMMA kernel. Thread tid stages weight row tid / 2, blocks
// wb = 2 * (tid & 1) and wb + 1 of the chunk's 4 blocks: load() reads them from global memory into registers
// (so the next chunk's loads overlap this chunk's WMMAs), store() decodes them to int8 in LDS and writes the two
// half-block scales of each block.
template <class P>
struct WStageP4 {  // ROCmFP4 / Q4_0: a thread's two 18-byte blocks (wb = 0 or 2) are the 36-byte, 4-aligned pair at
                   // byte 18 wb of the chunk's 4 blocks; a scale per half-block (ROCmFP4) or per block (Q4_0)
    static constexpr bool BLOCK_SCALE = P::BLOCK_SCALE, MIN = false;
    uint32_t w[9];
    __device__ void load(const uint8_t *W, long long rb, int row, bool ok, int b0, int wb) {
        const uint8_t *b = W + (long long)row * rb + (long long)(b0 + wb) * 18;  // b0 % 4 == 0
#pragma unroll
        for (int k = 0; k < 9; ++k) w[k] = ok ? ld32(b + 4 * k) : 0;
    }
    __device__ void store(int8_t *srow, float *erow, int wb) const {
        uint32_t qs[2][4];
        float e[2][2];
        P::pair(w, qs, e);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t *dst = (uint32_t *)(srow + (wb + j) * 32);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                dst[k] = (uint32_t)P::dec4(qs[j][k] & 0x0F0F0F0Fu);
                dst[4 + k] = (uint32_t)P::dec4((qs[j][k] >> 4) & 0x0F0F0F0Fu);
            }
            erow[(wb + j) * 2] = e[j][0];
            erow[(wb + j) * 2 + 1] = e[j][1];
        }
    }
};

struct WStageFast {  // Q4_0_ROCMFP4_FAST: 17-byte blocks, one UE4M3 scale per 32 values. The chunk's 4 blocks
                     // of a row are a 68-byte, 4-aligned quad; a thread's two blocks (wb = 0 or 2) are its bytes
                     // 17 wb .. 17 wb + 33: load the 9 dwords from byte 16 wb on, extract with v_alignbyte.
    static constexpr bool BLOCK_SCALE = true, MIN = false;  // both half-blocks share it: one int32 dot per block
    uint32_t w[10];
    int sh;  // byte offset of the thread's first block in w: 17 wb - 16 wb
    __device__ void load(const uint8_t *W, long long rb, int row, bool ok, int b0, int wb) {
        const uint8_t *b = W + (long long)row * rb + (long long)b0 * 17 + 16 * wb;  // b0 % 4 == 0
#pragma unroll
        for (int k = 0; k < 9; ++k) w[k] = ok ? ld32(b + 4 * k) : 0;
        w[9] = 0;
        sh = wb;
    }
    __device__ void store(int8_t *srow, float *erow, int wb) const {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            // bytes off .. of w, off = 17 j + sh: the dword index is the same for both sh, only the shift differs
            const int s = (17 * j + sh) & 3, d = (17 * j) >> 2;
            uint32_t *dst = (uint32_t *)(srow + (wb + j) * 32);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t qd = __builtin_amdgcn_alignbyte(w[d + k + 1], w[d + k], s);
                dst[k] = (uint32_t)rocmfp4_dec4(qd & 0x0F0F0F0Fu);
                dst[4 + k] = (uint32_t)rocmfp4_dec4((qd >> 4) & 0x0F0F0F0Fu);
            }
            const float e = ue4m3_half((uint8_t)(__builtin_amdgcn_alignbyte(w[d + 5], w[d + 4], s) & 0xFF));
            erow[(wb + j) * 2] = e;
            erow[(wb + j) * 2 + 1] = e;
        }
    }
};

struct WStageQ8_0 {  // Q8_0: 34-byte blocks, an F16 scale per 32 int8 values (no decode: copied as they are).
                     // A thread's two blocks (wb = 0 or 2) are the 68-byte, 4-aligned pair at byte 34 wb of the
                     // chunk's 4 blocks.
    static constexpr bool BLOCK_SCALE = true, MIN = false;
    uint32_t qs[2][8];
    float e[2];
    __device__ void load(const uint8_t *W, long long rb, int row, bool ok, int b0, int wb) {
        if (ok) {
            q8_0_pair(W + (long long)row * rb + (long long)(b0 + wb) * 34, qs, e);  // b0 % 4 == 0
        } else {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                e[j] = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) qs[j][k] = 0;
            }
        }
    }
    __device__ void store(int8_t *srow, float *erow, int wb) const {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t *dst = (uint32_t *)(srow + (wb + j) * 32);
#pragma unroll
            for (int k = 0; k < 8; ++k) dst[k] = qs[j][k];
            erow[(wb + j) * 2] = e[j];
            erow[(wb + j) * 2 + 1] = e[j];
        }
    }
};

template <class P>
struct WStagePB {  // Q5_1 / Q4_1 / Q5_0: a scale d (and offset m: MIN) per 32 values, decoded to int8 as the
                   // decode kernels do. store_min writes the blocks' m.
    static constexpr bool BLOCK_SCALE = true, MIN = P::MIN;
    uint32_t w[2][P::DW];
    __device__ void load(const uint8_t *W, long long rb, int row, bool ok, int b0, int wb) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            if (ok) {
                P::load(W + (long long)row * rb + (long long)(b0 + wb + j) * P::BYTES, w[j]);
            } else {
#pragma unroll
                for (int k = 0; k < P::DW; ++k) w[j][k] = 0;
            }
        }
    }
    __device__ void store(int8_t *srow, float *erow, int wb) const {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t q[8];
            float d, m;
            P::decode(w[j], q, d, m);
            uint32_t *dst = (uint32_t *)(srow + (wb + j) * 32);
#pragma unroll
            for (int k = 0; k < 8; ++k) dst[k] = q[k];
            erow[(wb + j) * 2] = d;
            erow[(wb + j) * 2 + 1] = d;
        }
    }
    __device__ void store_min(float *mrow, int wb) const {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            uint32_t q[8];
            float d, m;
            P::decode(w[j], q, d, m);
            mrow[wb + j] = m;
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

__device__ __forceinline__ void token_scales(const float *p, float dx[8]) {  // 8 adjacent floats, 16-aligned
    const float4 a = ((const float4 *)p)[0], b = ((const float4 *)p)[1];
    dx[0] = a.x, dx[1] = a.y, dx[2] = a.z, dx[3] = a.w, dx[4] = b.x, dx[5] = b.y, dx[6] = b.z, dx[7] = b.w;
}

// The kernel's LDS, declared once by the __global__ function: the expert kernel instantiates wmma_q80 for 1..4
// sub-tiles, whose own __shared__ arrays would each take their space. Rows padded by 16 bytes: the WMMA operand
// loads read 16 bytes from 16 consecutive rows, and an unpadded 128-byte row stride would put them all in the
// same LDS banks.
#define WM_ROWS 128                // weight rows per workgroup of the Q8_0-activation kernels: 8 waves
#define WM_THREADS (2 * WM_ROWS)   // two threads stage each row
struct WmmaSmem {
    int8_t sw[WM_ROWS][WM_KC + 16];  // decoded weights
    int8_t sx[WM_TILE][WM_KC + 16];  // token activations
    float sew[WM_ROWS][WM_KC / 16];  // weight half-block scales
    // token block scales, block c of token 16 n + 2 r + hi at [c][16 n + 8 hi + r]: a lane's 8 C rows are adjacent
    float sdx[WM_KC / 32][WM_TILE];
};
// Policies with offsets (MIN): also the weight blocks' m and the tokens' d_x * (block sum), laid out as sew / sdx
struct WmmaSmemMin : WmmaSmem {
    float smw[WM_ROWS][WM_KC / 32];
    float sxs[WM_KC / 32][WM_TILE];
};
template <bool MIN> struct WmmaSmemSel { typedef WmmaSmem type; };
template <> struct WmmaSmemSel<true> { typedef WmmaSmemMin type; };

// NT: 16-token sub-tiles per workgroup (4: 64 tokens of a dense prefill; an expert tile of up to 64 entries takes
// the fewest that cover it). No per-sub-tile branches: uniform branches between the WMMAs cost a dense
// prefill 17%. Each value is the same sum in the same order whichever NT computes it.
template <class WS, int NT, class RM, class SM>
__device__ __forceinline__ void wmma_q80(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8,
                                         const int *bs, float *y, long long ldy, int N, int K, const RM &rm, SM &sm) {
    constexpr int TOK = 16 * NT;
    auto &sw = sm.sw;
    auto &sx = sm.sx;
    auto &sew = sm.sew;
    auto &sdx = sm.sdx;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int r0 = blockIdx.x * WM_ROWS, T = rm.nt;
    const int nb = K / 32;
    const uint8_t *Wb = rm.base(W, row_bytes, N);

    float acc[NT][8];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[n][r] = 0.0f;

    // One K chunk's global data in registers: weights per the staging policy; activations: XC uint4 per thread
    // (task tid + WM_THREADS c < TOK * 8: token task / 8, bytes 16 (task % 8))
    constexpr int XC = (TOK * 8 + WM_THREADS - 1) / WM_THREADS;
    WS ws;
    uint4 gx[XC];
    const int wr_row = tid >> 1, wr_b = 2 * (tid & 1);
    // (K % 128 == 64: the last chunk's blocks 2 and 3, a thread's both or neither, and their activations are
    // zeros)
    auto load_chunk = [&](int k0) {
        ws.load(Wb, row_bytes, r0 + wr_row, r0 + wr_row < N && k0 / 32 + wr_b < nb, k0 / 32, wr_b);
#pragma unroll
        for (int c = 0; c < XC; ++c) {
            const int task = tid + WM_THREADS * c, tt = task >> 3, kk = k0 + 16 * (task & 7);
            gx[c] = tt < T && kk < K ? *(const uint4 *)(q8 + rm.in(tt) * K + kk) : uint4{0, 0, 0, 0};
        }
    };
    auto store_chunk = [&](int k0) {
        const int b0 = k0 / 32;
        ws.store(&sw[wr_row][0], &sew[wr_row][0], wr_b);
        if constexpr (WS::MIN) ws.store_min(&sm.smw[wr_row][0], wr_b);
#pragma unroll
        for (int c = 0; c < XC; ++c) {
            const int task = tid + WM_THREADS * c;
            if (task < TOK * 8) *(uint4 *)&sx[task >> 3][16 * (task & 7)] = gx[c];
        }
        // token block scales (MIN: and d_x * block sum): TOK tokens x 4 blocks
        for (int task = tid; task < TOK * 4; task += WM_THREADS) {
            const int tt = task >> 2, c = task & 3, at = (tt & ~15) + 8 * (tt & 1) + ((tt & 15) >> 1);
            const bool ok = tt < T && b0 + c < nb;
            const float dx = ok ? d8[rm.in(tt) * nb + b0 + c] : 0.0f;
            sdx[c][at] = dx;
            if constexpr (WS::MIN) sm.sxs[c][at] = ok ? __int_as_float(bs[rm.in(tt) * nb + b0 + c]) : 0.0f;
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
        if constexpr (WS::BLOCK_SCALE) {
#pragma unroll 1
            for (int j = 0; j < WM_KC / 32; ++j) {
                // one scale per 32-value block: its two 16-value steps' dots are added in int32 (the WMMA's
                // accumulator), then scaled once, as the dot4 kernels' e * dx * (float)(s0 + s1)
                const v4i_t a0 = *(const v4i_t *)&sw[16 * w + i][32 * j];
                const v4i_t a1 = *(const v4i_t *)&sw[16 * w + i][32 * j + 16];
                v8i_t c[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) {  // issue the WMMAs first, then their epilogues
                    const v4i_t b0 = *(const v4i_t *)&sx[16 * n + i][32 * j];
                    const v4i_t b1 = *(const v4i_t *)&sx[16 * n + i][32 * j + 16];
                    c[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b0, true, a0, v8i_t{}, false);
                    c[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b1, true, a1, c[n], false);
                }
                const float ew = sew[16 * w + i][2 * j];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    float dx[8];
                    token_scales(&sdx[j][16 * n + 8 * hi], dx);
#pragma unroll
                    for (int r = 0; r < 8; ++r) acc[n][r] += (ew * dx[r]) * (float)c[n][r];
                    if constexpr (WS::MIN) {  // + m_w * d_x * (block sum)
                        const float mw = sm.smw[16 * w + i][j];
                        float xs[8];
                        token_scales(&sm.sxs[j][16 * n + 8 * hi], xs);
#pragma unroll
                        for (int r = 0; r < 8; ++r) acc[n][r] += mw * xs[r];
                    }
                }
            }
        } else {
            // A scale per 16 values: the block's two half-block dots scaled by their own e_w, then by the
            // token's d_x once, acc += d_x (e_w0 dot0 + e_w1 dot1) (5 VALU ops per pair of results, not 6)
#pragma unroll 1
            for (int j = 0; j < WM_KC / 32; ++j) {
                const v4i_t a0 = *(const v4i_t *)&sw[16 * w + i][32 * j];
                const v4i_t a1 = *(const v4i_t *)&sw[16 * w + i][32 * j + 16];
                v8i_t c0[NT], c1[NT];
#pragma unroll
                for (int n = 0; n < NT; ++n) {  // issue the WMMAs first, then their epilogues
                    const v4i_t b0 = *(const v4i_t *)&sx[16 * n + i][32 * j];
                    const v4i_t b1 = *(const v4i_t *)&sx[16 * n + i][32 * j + 16];
                    c0[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, b0, v8i_t{}, false);
                    c1[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, v8i_t{}, false);
                }
                float ew0[8], ew1[8];
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    ew0[r] = sew[16 * w + 2 * r + hi][2 * j];
                    ew1[r] = sew[16 * w + 2 * r + hi][2 * j + 1];
                }
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const float dx = sdx[j][16 * n + 8 * (i & 1) + (i >> 1)];  // token 16 n + i
#pragma unroll
                    for (int r = 0; r < 8; ++r) acc[n][r] += dx * ((float)c0[n][r] * ew0[r] + (float)c1[n][r] * ew1[r]);
                }
            }
        }
        __syncthreads();
    }

    if constexpr (WS::BLOCK_SCALE) {  // C column l % 16 = weight row 16 w + l % 16, row 2 r + l / 16 = token
        const int row = r0 + 16 * w + i;
        if (row >= N) return;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int tt = 16 * n + 2 * r + hi;
                if (tt < T) y[rm.out(tt) * ldy + row] = acc[n][r];
            }
        }
    } else {  // C column l % 16 = token, row 2 r + l / 16 = weight row
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
}

// matmul_wmma_<type>: grid (ceil(N / WM_ROWS), ceil(T / 64)). matmul_wmma_id_<type> (MoE prefill): grid
// (ceil(N / WM_ROWS), max_tiles) over moe_group's expert tiles of up to 64 entries (QMV_WMMA_ID_TB in qmatvec.cpp), so a
// batch reads each expert's weights once per 64 of its entries; same arguments as matmul_q8_id_<type>.
#define MATMUL_WMMA_Q80_KERNELS(name, WS)                                                                    \
    extern "C" __global__ void __launch_bounds__(WM_THREADS)                                                 \
        matmul_wmma_##name(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8,          \
                           const int *bs, float *y, long long ldy, int N, int K, int T) {                    \
        __shared__ __attribute__((aligned(16))) WmmaSmemSel<WS::MIN>::type sm;                               \
        const int t0 = blockIdx.y * WM_TILE;                                                                 \
        const WRowsLinear rm{t0, min(WM_TILE, T - t0)};                                                      \
        wmma_q80<WS, 4>(W, row_bytes, q8, d8, bs, y, ldy, N, K, rm, sm);                                     \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(WM_THREADS) matmul_wmma_id_##name(MMID_ARGS) {               \
        __shared__ __attribute__((aligned(16))) WmmaSmemSel<WS::MIN>::type sm;                               \
        const int k = blockIdx.y;                                                                            \
        if (k >= *n_tiles) return;                                                                           \
        const WRowsExpert rm{tile_e[k], tile_n[k], in_rows + tile_start[k], out_rows + tile_start[k]};       \
        if (rm.nt <= 16)                                                                                     \
            wmma_q80<WS, 1>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);                                   \
        else if (rm.nt <= 32)                                                                                \
            wmma_q80<WS, 2>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);                                   \
        else if (rm.nt <= 48)                                                                                \
            wmma_q80<WS, 3>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);                                   \
        else                                                                                                 \
            wmma_q80<WS, 4>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);                                   \
    }

MATMUL_WMMA_Q80_KERNELS(q4_0_rocmfp4, WStageP4<P4ROCmFP4>)
MATMUL_WMMA_Q80_KERNELS(q4_0, WStageP4<P4Q4_0>)
MATMUL_WMMA_Q80_KERNELS(q4_0_rocmfp4_fast, WStageFast)
MATMUL_WMMA_Q80_KERNELS(q8_0, WStageQ8_0)
MATMUL_WMMA_Q80_KERNELS(q5_1, WStagePB<PBQ5_1>)
MATMUL_WMMA_Q80_KERNELS(q4_1, WStagePB<PBQ4_1>)
MATMUL_WMMA_Q80_KERNELS(q5_0, WStagePB<PBQ5_0>)

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

// The K-quant kernels' LDS, declared once by the __global__ function (the expert kernel instantiates wmma_q8k for
// 1..4 sub-tiles, whose own __shared__ arrays would each take their space)
struct WmmaKSmem {
    int8_t sw[WM_TILE][WM_KC + 16];
    int8_t sx[WM_TILE][WM_KC + 16];
    int sgsc[WM_TILE][8];   // integer scale of each local 16-value group
    int smin[WM_TILE][4];   // Q4/5: mins of the chunk's 4 sub-blocks
    float sdw[WM_TILE], sdm[WM_TILE];
    float sdx[WM_TILE];     // token Q8_K scale of this super-block
    int sbs[WM_TILE][4];    // Q4/5: token group sums per sub-block (bs[2 sub] + bs[2 sub + 1])
};

// RM: the tile's tokens (WRowsLinear, or WRowsExpert: one expert's entries, MoE prefill), as in wmma_q80. NT: the
// 16-token sub-tiles computed (4 for a dense tile; an expert tile takes the fewest that cover its entries: a few
// entries in 4 sub-tiles made the kernel slower than the dot4 one below ~300 tokens per batch)
template <int KT, int NT, class RM>
__device__ __forceinline__ void wmma_q8k(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8,
                                         const int *bs, float *y, long long ldy, int N, int K, const RM &rm,
                                         WmmaKSmem &sm) {
    auto &sw = sm.sw;
    auto &sx = sm.sx;
    auto &sgsc = sm.sgsc;
    auto &smin = sm.smin;
    auto &sdw = sm.sdw;
    auto &sdm = sm.sdm;
    auto &sdx = sm.sdx;
    auto &sbs = sm.sbs;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int r0 = blockIdx.x * WM_TILE, T = rm.nt;
    const int nbk = K / 256;
    const uint8_t *Wb = rm.base(W, row_bytes, N);

    float acc[NT][8];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[n][r] = 0.0f;

    WStageK<KT> ws;
    uint4 gx[4];
    const int wr_row = tid >> 1, part = tid & 1;
    const int xt0 = tid >> 1, xc = (tid & 1) * 64;
    auto load_chunk = [&](int k0) {
        ws.load(Wb, row_bytes, r0 + wr_row, r0 + wr_row < N, k0, part);
        const long long t = xt0 < T ? rm.in(xt0) : 0;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            gx[c] = xt0 < T ? *(const uint4 *)(q8 + t * K + k0 + xc + 16 * c) : uint4{0, 0, 0, 0};
    };
    auto store_chunk = [&](int k0) {
        ws.store(&sw[wr_row][0], &sgsc[wr_row][0], &sdw[wr_row], &sdm[wr_row], &smin[wr_row][0], k0, part);
#pragma unroll
        for (int c = 0; c < 4; ++c) *(uint4 *)&sx[xt0][xc + 16 * c] = gx[c];
        const int bk = k0 / 256, hh = (k0 / 128) & 1;
        if (tid < 64) {
            sdx[tid] = tid < T ? d8[rm.in(tid) * nbk + bk] : 0.0f;
        } else if (KT != 6) {
            const int tt = tid - 64;
            const long long t = tt < T ? rm.in(tt) : 0;
#pragma unroll
            for (int s2 = 0; s2 < 4; ++s2) {
                const int sub = 4 * hh + s2;
                sbs[tt][s2] = tt < T ? bs[t * (K / 16) + bk * 16 + 2 * sub] + bs[t * (K / 16) + bk * 16 + 2 * sub + 1]
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

        int isum[NT][8];
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int r = 0; r < 8; ++r) isum[n][r] = 0;
        // Q4_K / Q5_K: one scale per 32-value sub-block, so its two 16-value WMMAs chain in the int32 accumulator
        // and are scaled once (the same integer sums); Q6_K: a scale per 16 values
        constexpr int GW = KT == 6 ? 1 : 2;  // WMMAs per scale group
#pragma unroll 1
        for (int g = 0; g < WM_KC / 16; g += GW) {
            v8i_t c[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) c[n] = v8i_t{};
#pragma unroll
            for (int j = 0; j < GW; ++j) {
                const v4i_t a = *(const v4i_t *)&sw[16 * w + i][16 * (g + j)];
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const v4i_t bx = *(const v4i_t *)&sx[16 * n + i][16 * (g + j)];
                    c[n] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, bx, c[n], false);
                }
            }
            int gs[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) gs[r] = sgsc[16 * w + 2 * r + hi][g];
#pragma unroll
            for (int n = 0; n < NT; ++n)
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
        for (int n = 0; n < NT; ++n) {
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

extern "C" __global__ void __launch_bounds__(128)
matmul_wmma_q4_K(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y,
                 long long ldy, int N, int K, int T) {
    __shared__ __attribute__((aligned(16))) WmmaKSmem sm;
    const int t0 = blockIdx.y * WM_TILE;
    wmma_q8k<4, 4>(W, row_bytes, q8, d8, bs, y, ldy, N, K, WRowsLinear{t0, min(WM_TILE, T - t0)}, sm);
}

// MoE prefill: grid (ceil(N / 64), max_tiles) over moe_group's expert tiles of up to 64 entries (QMV_WMMA_ID_TB),
// block 128; same arguments as matmul_q8_id_<type>
extern "C" __global__ void __launch_bounds__(128) matmul_wmma_id_q4_K(MMID_ARGS) {
    const int k = blockIdx.y;
    if (k >= *n_tiles) return;
    __shared__ __attribute__((aligned(16))) WmmaKSmem sm;
    const WRowsExpert rm{tile_e[k], tile_n[k], in_rows + tile_start[k], out_rows + tile_start[k]};
    if (rm.nt <= 16) wmma_q8k<4, 1>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else if (rm.nt <= 32) wmma_q8k<4, 2>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else if (rm.nt <= 48) wmma_q8k<4, 3>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else wmma_q8k<4, 4>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
}

extern "C" __global__ void __launch_bounds__(128)
matmul_wmma_q5_K(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y,
                 long long ldy, int N, int K, int T) {
    __shared__ __attribute__((aligned(16))) WmmaKSmem sm;
    const int t0 = blockIdx.y * WM_TILE;
    wmma_q8k<5, 4>(W, row_bytes, q8, d8, bs, y, ldy, N, K, WRowsLinear{t0, min(WM_TILE, T - t0)}, sm);
}

// MoE prefill: grid (ceil(N / 64), max_tiles) over moe_group's expert tiles of up to 64 entries (QMV_WMMA_ID_TB),
// block 128; same arguments as matmul_q8_id_<type>
extern "C" __global__ void __launch_bounds__(128) matmul_wmma_id_q5_K(MMID_ARGS) {
    const int k = blockIdx.y;
    if (k >= *n_tiles) return;
    __shared__ __attribute__((aligned(16))) WmmaKSmem sm;
    const WRowsExpert rm{tile_e[k], tile_n[k], in_rows + tile_start[k], out_rows + tile_start[k]};
    if (rm.nt <= 16) wmma_q8k<5, 1>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else if (rm.nt <= 32) wmma_q8k<5, 2>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else if (rm.nt <= 48) wmma_q8k<5, 3>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else wmma_q8k<5, 4>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
}

extern "C" __global__ void __launch_bounds__(128)
matmul_wmma_q6_K(const uint8_t *W, long long row_bytes, const int8_t *q8, const float *d8, const int *bs, float *y,
                 long long ldy, int N, int K, int T) {
    __shared__ __attribute__((aligned(16))) WmmaKSmem sm;
    const int t0 = blockIdx.y * WM_TILE;
    wmma_q8k<6, 4>(W, row_bytes, q8, d8, bs, y, ldy, N, K, WRowsLinear{t0, min(WM_TILE, T - t0)}, sm);
}

// MoE prefill: grid (ceil(N / 64), max_tiles) over moe_group's expert tiles of up to 64 entries (QMV_WMMA_ID_TB),
// block 128; same arguments as matmul_q8_id_<type>
extern "C" __global__ void __launch_bounds__(128) matmul_wmma_id_q6_K(MMID_ARGS) {
    const int k = blockIdx.y;
    if (k >= *n_tiles) return;
    __shared__ __attribute__((aligned(16))) WmmaKSmem sm;
    const WRowsExpert rm{tile_e[k], tile_n[k], in_rows + tile_start[k], out_rows + tile_start[k]};
    if (rm.nt <= 16) wmma_q8k<6, 1>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else if (rm.nt <= 32) wmma_q8k<6, 2>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else if (rm.nt <= 48) wmma_q8k<6, 3>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
    else wmma_q8k<6, 4>(W, row_bytes, q8, d8, bs, y, N, N, K, rm, sm);
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
