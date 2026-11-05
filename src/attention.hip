// attention.hip -- gated full-attention token mixer kernels, one token (Phase 20 speed work).
//
// Gemma-4 (attention_forward_opts): head_dim 512 as well (attn_prep / attn_partial / attn_combine only), a cache row
// wider than the layer's n_kv * D (kv_stride), RoPE on the first rot_pairs NeoX pairs only, an RMSNorm of V before it
// is cached (v_norm), a sliding window (a token attends to its last `window` positions) and any score scale.
//
// Queued between the q/k/v projections and attn_output, so an attention layer needs one CPU sync. Same math
// as the CPU reference in qwen_ops.cpp (qwen_rmsnorm, qwen_rope_neox, qwen_attention), which follows
// llama.cpp's qwen35 graph. Compiled for head_dim D = 256 (Qwen3.5 and later: gated, attn_q holds [Q | gate] per
// head) and D = 128 (Qwen3: no gate, attn_q is Q): kernel names end in _256 / _128; `gated` selects the layout.
//
// All three take T tokens at positions pos0 .. pos0 + T - 1 (prefill; T = 1 when generating) through the
// last grid dimension; token z attends to positions 0 .. pos0 + z (causal). Per-token rows: qfull
// [(gated ? 2 : 1) * n_q * D], k / v [n_kv * D], q / out [n_q * D].
//
// attn_prep     grid (n_q + n_kv, T), block D. Workgroup h < n_q: Q head h (first half of attn_q's [Q | gate]
//               head), RMSNorm with q_norm, NeoX RoPE over the first n_rot dims -> q. Workgroup n_q + g:
//               K head g, RMSNorm with k_norm, RoPE -> FP16 into the K cache row of this position, and V
//               head g -> FP16 into the V cache row. The CPU never touches the KV cache.
// attn_partial  grid (n_q, n_chunks, T), block CH = 256 (positions per chunk), n_chunks for the last token;
//               chunks past a token's last position contribute nothing (m = -inf, l = 0, acc = 0). Flash-decoding: thread t scores
//               position chunk * CH + t against q (q . K / sqrt(D)), the workgroup takes the chunk's max m
//               and sum l of exp(score - m), then thread d < D accumulates sum_t exp(score_t - m) * V[t][d]
//               (coalesced over d). Writes m, l and acc[D] per (head, chunk).
// attn_combine  grid (n_q / (256 / D), T), block 256: 256 / D heads per workgroup (one 256-value block). Merges
//               the chunks (rescaling each by exp(m_c - M)), divides by the total weight and applies the output
//               gate sigmoid(gate) (gated) -> out. out feeds attn_output, so it is also quantized for the int8
//               matvec when o.qK != NULL (the workgroup's 256 values = one Q8_K block).
// Query head h reads KV head h / (n_q / n_kv), as ggml's broadcast does.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q8_device.h"

#define CH 256  // positions per attn_partial chunk (ATTN_CHUNK in attention.h)

// KV cache row of position p, in __half units from the layer's row 0: blocks of 2^kv_shift positions, kv_bstride
// apart, rows kv_stride apart inside a block (kv_cache.h). A contiguous cache is kv_shift = 30.
__device__ __forceinline__ size_t kv_row(int p, int kv_stride, int kv_shift, long long kv_bstride) {
    return (size_t)(p >> kv_shift) * (size_t)kv_bstride + (size_t)(p & ((1 << kv_shift) - 1)) * (size_t)kv_stride;
}

// Quantized KV cache rows (attention_opts.kv_quant, KQ): d_row values, then their fp16 scales, one per 32 values.
// KQ 1, Q8_0: int8 values (as ggml's q8_0: d = amax / 127, q = round(x / d)); kv_stride (in halves) = d_row * 17 /
// 32. KQ 2, Q4_0: 4-bit values q + 8 (as ggml's q4_0: d = max / -8, max the value of largest magnitude), byte 4 j + k
// (k < 4) holding values 8 j + k (low nibble) and 8 j + k + 4 (high: 4 bytes unpack to 8 consecutive values with
// two masks); kv_stride = d_row * 9 / 32. Element e: q[e] * scale[e / 32].
template <int KQ>
__device__ __forceinline__ int kvq_drow(int kv_stride) { return kv_stride * 32 / (KQ == 2 ? 9 : 17); }
template <int KQ>
__device__ __forceinline__ const __half* kvq_scales(const __half* row, int drow) {
    return (const __half*)((const int8_t*)row + (KQ == 2 ? drow / 2 : drow));
}

// halves (1024 + 128 + byte k of w, k = sel's bytes) - bias, times sc: the packed dequantization below
__device__ __forceinline__ __half2 kvq_pair(uint32_t w, uint32_t sel, __half2 bias, __half2 sc) {
    const uint32_t h = __builtin_amdgcn_perm(0x64646464u, w, sel);
    return __hmul2(__hsub2(*(const __half2*)&h, bias), sc);
}

// 8 consecutive values e .. e + 7 (e % 8 == 0) of a cache row in two steps: kv_fetch loads the raw words (and the
// block's scale) without using them, so that the loads of a whole block can be in flight together (a dequantizing
// load would wait for its data where it stands); kv_unpack turns them into 8 fp16 values in a uint4
struct KvRaw {
    uint4 u;   // F16: the values; Q8_0: u.x, u.y the 8 int8 values; Q4_0: u.x the 8 nibbles
    __half s;  // the block's scale (quantized)
};
template <int KQ>
__device__ __forceinline__ KvRaw kv_fetch(const __half* row, int e, int drow) {
    KvRaw r;
    if constexpr (KQ == 0) {
        r.u = *(const uint4*)(row + e);
    } else if constexpr (KQ == 2) {
        r.u.x = *(const uint32_t*)((const uint8_t*)row + e / 2);
        r.s = kvq_scales<2>(row, drow)[e / 32];
    } else {
        const uint2 q = *(const uint2*)((const int8_t*)row + e);
        r.u.x = q.x, r.u.y = q.y;
        r.s = kvq_scales<1>(row, drow)[e / 32];
    }
    return r;
}
template <int KQ>
__device__ __forceinline__ uint4 kv_unpack(const KvRaw& raw) {
    if constexpr (KQ == 0) {
        return raw.u;
    } else if constexpr (KQ == 2) {  // nibble n under the exponent of 1024 is the half 1024 + n: (that - 1032) * scale
        const __half2 sc = __half2half2(raw.s);
        const __half2 bias = __float2half2_rn(1032.0f);
        const uint32_t lo = raw.u.x & 0x0F0F0F0Fu, hi = (raw.u.x >> 4) & 0x0F0F0F0Fu;  // values 0 .. 3, 4 .. 7
        uint4 r;
        __half2* h = (__half2*)&r;
        h[0] = kvq_pair(lo, 0x04010400u, bias, sc), h[1] = kvq_pair(lo, 0x04030402u, bias, sc);
        h[2] = kvq_pair(hi, 0x04010400u, bias, sc), h[3] = kvq_pair(hi, 0x04030402u, bias, sc);
        return r;
    } else {
        // packed: byte q (as q + 128) under the exponent of 1024 is the half 1024 + 128 + q, so (that - 1152) *
        // scale is q * scale rounded once, as (float)q * scale then to half (both products are exact)
        const __half2 sc = __half2half2(raw.s);
        const __half2 bias = __float2half2_rn(1152.0f);
        const uint32_t w[2] = {raw.u.x ^ 0x80808080u, raw.u.y ^ 0x80808080u};
        uint4 r;
        __half2* h = (__half2*)&r;
#pragma unroll
        for (int k = 0; k < 2; ++k)
            h[2 * k] = kvq_pair(w[k], 0x04010400u, bias, sc), h[2 * k + 1] = kvq_pair(w[k], 0x04030402u, bias, sc);
        return r;
    }
}
template <int KQ>
__device__ __forceinline__ uint4 kv_load8(const __half* row, int e, int drow) {
    return kv_unpack<KQ>(kv_fetch<KQ>(row, e, drow));
}

// One value of a cache row as float
template <int KQ>
__device__ __forceinline__ float kv_value(const __half* row, int e, int drow) {
    if constexpr (KQ == 0) {
        return __half2float(row[e]);
    } else if constexpr (KQ == 2) {
        const uint8_t byte = ((const uint8_t*)row)[(e >> 3) * 4 + (e & 3)];
        const int n = (e & 4) ? byte >> 4 : byte & 15;
        return (float)(n - 8) * __half2float(kvq_scales<2>(row, drow)[e / 32]);
    } else {
        return (float)((const int8_t*)row)[e] * __half2float(kvq_scales<1>(row, drow)[e / 32]);
    }
}

__device__ __forceinline__ float wave_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    return v;
}

// v from the lane whose index differs in bits M (M < 16: within the lane's DPP row of 16): DPP row_xmask, an ALU
// move instead of __shfl_xor's ds_bpermute through the LDS crossbar
template <int M>
__device__ __forceinline__ float row_xor(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x160 | M, 0xf, 0xf, false));
}
__device__ __forceinline__ float row16_max(float v) {
    v = fmaxf(v, row_xor<1>(v));
    v = fmaxf(v, row_xor<2>(v));
    v = fmaxf(v, row_xor<4>(v));
    return fmaxf(v, row_xor<8>(v));
}
__device__ __forceinline__ float row16_sum(float v) {
    v += row_xor<1>(v);
    v += row_xor<2>(v);
    v += row_xor<4>(v);
    return v + row_xor<8>(v);
}

__device__ __forceinline__ float wave_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 32));
    return v;
}

// Writes value x as element e = 32 b + (lane) of a quantized cache row (all 32 lanes of a wave together: one
// block): Q8_0 or Q4_0 (kq), the block's scale by lane 0
__device__ __forceinline__ void kvq_store(__half* row, int e, float x, int kq, int kv_stride) {
    const int lane = e & 31;
    if (kq == 1) {
        const int drow = kvq_drow<1>(kv_stride);
        const float d = wave_max(fabsf(x)) / 127.0f;
        ((int8_t*)row)[e] = (int8_t)(int)rintf(d != 0.0f ? x * (1.0f / d) : 0.0f);
        if (lane == 0) ((__half*)((int8_t*)row + drow))[e / 32] = __float2half(d);
    } else {
        const int drow = kvq_drow<2>(kv_stride);
        const float amax = wave_max(fabsf(x));
        const float mx = wave_max(fabsf(x) == amax ? x : -INFINITY);  // the value of largest magnitude
        const float d = mx / -8.0f;
        const int n = min(15, (int)(d != 0.0f ? x * (1.0f / d) + 8.5f : 8.0f));
        const int hi = __shfl_down(n, 4, 32);  // value e + 4 shares a byte with e (e % 8 < 4)
        if ((e & 4) == 0) ((uint8_t*)row)[(e >> 3) * 4 + (e & 3)] = (uint8_t)(n | (hi << 4));
        if (lane == 0) ((__half*)((int8_t*)row + drow / 2))[e / 32] = __float2half(d);
    }
}

// Sum / max over the NTH threads of a workgroup (wave32)
template <int NTH>
__device__ float block_sum(float v) {
    __shared__ float part[NTH / 32];
    v = wave_sum(v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NTH / 32; ++i) s += part[i];
    __syncthreads();
    return s;
}

template <int NTH>
__device__ float block_max(float v) {
    __shared__ float part[NTH / 32];
    v = wave_max(v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < NTH / 32; ++i) m = fmaxf(m, part[i]);
    __syncthreads();
    return m;
}

// RMSNorm (gamma w; none if NULL; no norm at all unless norm) then RoPE of one head held one value per thread
// (D threads); returns this thread's value. NeoX: pairs (i, i + n_rot / 2); rope_norm (llama.cpp's
// LLAMA_ROPE_TYPE_NORM: Llama): pairs (2 i, 2 i + 1). Pair i < rot_pairs rotates by theta_i = pos *
// base^(-2 i / n_rot) / ff[i] (ff: rope_freqs, Llama 3's frequency divisors; NULL: 1), the others stay as they are;
// n_rot 0: no RoPE.
template <int D>
__device__ float norm_rope(float x, const float* w, int n_rot, int rot_pairs, int pos, float freq_base, float eps,
                           bool norm = true, int rope_norm = 0, const float* ff = nullptr) {
    __shared__ float v[D];
    const int d = threadIdx.x;
    if (norm) {  // uniform
        const float ss = block_sum<D>(x * x);
        x = x * rsqrtf(ss / D + eps) * (w ? w[d] : 1.0f);
    }
    v[d] = x;
    __syncthreads();
    const int half = n_rot / 2;
    const int i = rope_norm ? d / 2 : d < half ? d : d - half;  // this dim's pair
    if (d < n_rot && i < rot_pairs) {
        float theta = (float)pos * powf(freq_base, -2.0f * (float)i / (float)n_rot);
        if (ff) theta /= ff[i];
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const int i0 = rope_norm ? 2 * i : i, i1 = rope_norm ? 2 * i + 1 : i + half;
        const float x0 = v[i0], x1 = v[i1];
        x = d == i0 ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    __syncthreads();
    return x;
}

// Orthonormal Walsh-Hadamard transform (symmetric, its own inverse) of NT values held one per thread, in blocks of
// N, as llama.cpp rotates attention for a quantized cache (src/llama-kv-cache.cpp, attn_rot): Q and K by blocks of
// HK(D) (64, doubled while it divides D: the whole head for 64 / 128 / 256), so Q.K is unchanged; V by blocks of 64,
// the attention output rotated back (attn_combine). The outliers of a row are spread over its block before the 4- /
// 8-bit quantization (Qwen3-0.6B, Q4_0 cache: ppl 132.7 -> 84.7). Butterflies in LDS; all NT threads take part.
__host__ __device__ constexpr int hadamard_k(int D) {
    int r = 64;
    while (D % (2 * r) == 0) r *= 2;
    return r;
}
// attn_prep's kv_quant: the cache type (0 F16, 1 Q8_0, 2 Q4_0) | ATTN_KV_ROT (rotate; the host sets it for a
// quantized cache and D % 64 == 0 unless INFERENCE_KV_ROT=0)
#define ATTN_KV_ROT 4
template <int NT, int N>
__device__ float hadamard_rot(float x) {
    __shared__ float v[NT];
    const int d = threadIdx.x;
#pragma unroll
    for (int s = 1; s < N; s <<= 1) {
        v[d] = x;
        __syncthreads();
        const float y = v[d ^ s];
        x = (d & s) ? y - x : x + y;
        __syncthreads();
    }
    return x * (1.0f / __builtin_sqrtf((float)N));
}

template <int D>
__device__ __forceinline__ void attn_prep_body(const float* qfull, const float* k, const float* v, const float* q_norm,
                                               const float* k_norm, float* q_out, __half* K, __half* V, int kv_stride,
                                               int kv_shift, long long kv_bstride, int n_q, int n_kv, int n_rot,
                                               int pos0, float freq_base, float eps, int gated, int rot_pairs,
                                               int v_norm, int qk_norm, int kv_quant, const float* bq, const float* bk,
                                               const float* bv, int rope_norm, const float* ff) {
    const int b = blockIdx.x, d = threadIdx.x, z = blockIdx.y, pos = pos0 + z;
    const int qs = gated ? 2 * D : D;  // a head's stride in attn_q's output
    qfull += (size_t)z * n_q * qs;
    q_out += (size_t)z * n_q * D;
    k += (size_t)z * n_kv * D;
    v += (size_t)z * n_kv * D;
    __half* k_row = K + kv_row(pos, kv_stride, kv_shift, kv_bstride);
    __half* v_row = V + kv_row(pos, kv_stride, kv_shift, kv_bstride);
    if (b < n_q) {
        const float x = qfull[(size_t)b * qs + d] + (bq ? bq[(size_t)b * qs + d] : 0.0f);
        const float y = norm_rope<D>(x, q_norm, n_rot, rot_pairs, pos, freq_base, eps, qk_norm, rope_norm, ff);
        q_out[b * D + d] = y;
        if (kv_quant & ATTN_KV_ROT) q_out[b * D + d] = hadamard_rot<D, hadamard_k(D)>(y);  // (uniform; K the same)
    } else {
        const int g = b - n_q;
        float x = norm_rope<D>(k[g * D + d] + (bk ? bk[g * D + d] : 0.0f), k_norm, n_rot, rot_pairs, pos, freq_base,
                               eps, qk_norm, rope_norm, ff);
        if (kv_quant & ATTN_KV_ROT) x = hadamard_rot<D, hadamard_k(D)>(x);  // (uniform)
        const float vv = v[g * D + d] + (bv ? bv[g * D + d] : 0.0f);
        float vx = v_norm ? norm_rope<D>(vv, nullptr, 0, 0, pos, freq_base, eps) : vv;
        if (kv_quant & ATTN_KV_ROT) vx = hadamard_rot<D, 64>(vx);  // (uniform; attn_combine rotates back)
        if (kv_quant & 3) {  // uniform: a wave's 32 dims are one block
            kvq_store(k_row, g * D + d, x, kv_quant & 3, kv_stride);
            kvq_store(v_row, g * D + d, vx, kv_quant & 3, kv_stride);
        } else {
            k_row[g * D + d] = __float2half(x);
            v_row[g * D + d] = __float2half(vx);
        }
    }
}

template <int D, int KQ = 0>
__device__ __forceinline__ void attn_partial_body(const float* q, const __half* K, const __half* V, int kv_stride,
                                                  int kv_shift, long long kv_bstride, int pos0, int n_q, int n_kv_heads,
                                                  float scale, float* part_ml, float* part_acc, int window) {
    __shared__ float sq[D], sw[CH];
    const int h = blockIdx.x, c = blockIdx.y, t = threadIdx.x, n_chunks = gridDim.y, z = blockIdx.z;
    const int n_kv = pos0 + z + 1;
    const int lo = window > 0 ? n_kv - window : 0;  // first position this token sees (sliding window)
    q += (size_t)z * n_q * D;
    part_ml += (size_t)2 * z * n_q * n_chunks;
    part_acc += (size_t)z * n_q * n_chunks * D;
    const int g = h / (n_q / n_kv_heads);
    const size_t pc = (size_t)h * n_chunks + c;
    if (c * CH + CH <= lo) {  // the whole chunk is before the window (uniform): it contributes nothing
        for (int d = t; d < D; d += CH) part_acc[pc * D + d] = 0.0f;
        if (t == 0) part_ml[2 * pc] = -INFINITY, part_ml[2 * pc + 1] = 0.0f;
        return;
    }
    for (int d = t; d < D; d += CH) sq[d] = q[h * D + d];
    __syncthreads();

    // Score of position p = c * CH + t (thread per position); K row read as 8 halves at a time
    const int p = c * CH + t;
    float score = -INFINITY;
    if (p < n_kv && p >= lo) {
        float s = 0.0f;
        if constexpr (KQ == 1) {  // per 32-value block: the int8 dot with q, times the block's scale
            const int8_t* kb = (const int8_t*)(K + kv_row(p, kv_stride, kv_shift, kv_bstride));
            const __half* ks = (const __half*)(kb + kvq_drow<1>(kv_stride)) + g * (D / 32);
            kb += g * D;
#pragma unroll 2
            for (int blk = 0; blk < D / 32; ++blk) {
                const uint4 u0 = ((const uint4*)(kb + 32 * blk))[0], u1 = ((const uint4*)(kb + 32 * blk))[1];
                const int8_t* q0 = (const int8_t*)&u0;
                const int8_t* q1 = (const int8_t*)&u1;
                float sb = 0.0f;
#pragma unroll
                for (int j = 0; j < 16; ++j) sb += (float)q0[j] * sq[32 * blk + j] + (float)q1[j] * sq[32 * blk + 16 + j];
                s += sb * __half2float(ks[blk]);
            }
        } else if constexpr (KQ == 2) {  // per 8 values: the 4-bit values' dot with q, times the block's scale
            const __half* kr = K + kv_row(p, kv_stride, kv_shift, kv_bstride);
            const __half* ks = kvq_scales<2>(kr, kvq_drow<2>(kv_stride)) + g * (D / 32);
            const uint32_t* kw = (const uint32_t*)kr + g * D / 8;
#pragma unroll 4
            for (int i = 0; i < D / 8; ++i) {
                const uint32_t w = kw[i];
                float sb = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    sb += (float)((int)((w >> (8 * j)) & 15) - 8) * sq[8 * i + j] +
                          (float)((int)((w >> (8 * j + 4)) & 15) - 8) * sq[8 * i + 4 + j];
                s += sb * __half2float(ks[i / 4]);
            }
        } else {
            const __half* kr = K + kv_row(p, kv_stride, kv_shift, kv_bstride);
#pragma unroll 4
            for (int i = 0; i < D / 8; ++i) {
                const uint4 u = *(const uint4*)(kr + g * D + 8 * i);
                const __half2* h2 = (const __half2*)&u;
#pragma unroll
                for (int k2 = 0; k2 < 4; ++k2) {
                    const float2 f = __half22float2(h2[k2]);
                    s += f.x * sq[8 * i + 2 * k2] + f.y * sq[8 * i + 2 * k2 + 1];
                }
            }
        }
        score = s * scale;
    }
    const float m = block_max<CH>(score);
    const float e = score > -INFINITY ? __expf(score - m) : 0.0f;
    const float l = block_sum<CH>(e);
    sw[t] = e;
    __syncthreads();

    // acc[d] over this chunk's positions (thread per dim; D > CH: CH apart)
    const int p0 = c * CH, i0 = max(0, lo - p0), np = max(0, min(CH, n_kv - p0));
    for (int d = t; d < D; d += CH) {
        float acc = 0.0f;
        if constexpr (KQ) {
            const int drow = kvq_drow<KQ>(kv_stride);
#pragma unroll 8
            for (int i = i0; i < np; ++i)
                acc += sw[i] * kv_value<KQ>(V + kv_row(p0 + i, kv_stride, kv_shift, kv_bstride), g * D + d, drow);
        } else {
            const __half* vb = V + (size_t)g * D + d;
#pragma unroll 8  // (8 positions' loads in flight)
            for (int i = i0; i < np; ++i) acc += sw[i] * __half2float(vb[kv_row(p0 + i, kv_stride, kv_shift, kv_bstride)]);
        }
        part_acc[pc * D + d] = acc;
    }
    if (t == 0) {
        part_ml[2 * pc] = m;
        part_ml[2 * pc + 1] = l;
    }
}

template <int D>
__device__ __forceinline__ void attn_combine_body(const float* part_ml, const float* part_acc, int n_chunks, int n_q,
                                                  const float* qfull, float* out, Q8Out o, int gated, int vrot) {
    __shared__ Q8Shared sh;
    // workgroup: 256 values of the token's n_q * D (256 / D heads, or 256 / D of one head for D 512); the last one
    // partly past n_q * D when that is not a multiple of 256 (no int8 output then)
    const int e = blockIdx.x * 256 + threadIdx.x, z = blockIdx.y;
    const bool active = e < n_q * D;
    const int h = active ? e / D : 0, d = e % D;
    const int nb = (n_q * D + 255) / 256;  // 256-value blocks per token
    part_ml += (size_t)2 * z * n_q * n_chunks;
    part_acc += (size_t)z * n_q * n_chunks * D;
    qfull += (size_t)z * (gated ? 2 : 1) * n_q * D;
    out += (size_t)z * n_q * D;
    const float* ml = part_ml + (size_t)2 * h * n_chunks;
    // (unrolled: 8 chunks' loads in flight; a long context has hundreds of chunks per head)
    float M = -INFINITY;
#pragma unroll 8
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, ml[2 * c]);
    float num = 0.0f, den = 0.0f;
#pragma unroll 8
    for (int c = 0; c < n_chunks; ++c) {
        const float w = __expf(ml[2 * c] - M);
        num += w * part_acc[((size_t)h * n_chunks + c) * D + d];
        den += w * ml[2 * c + 1];
    }
    float r = num / den;
    if (vrot) r = hadamard_rot<256, 64>(r);  // (uniform; V was cached rotated: attn_prep)
    if (gated) r /= 1.0f + __expf(-qfull[(size_t)h * 2 * D + D + d]);
    if (active) out[h * D + d] = r;
    if (q8_wanted(o)) q8_block_value(r, active, z * nb + blockIdx.x, threadIdx.x, sh, o);  // uniform: one 256-block
}

// ---------------------------------------------------------------- prefill: query-tiled flash attention
//
// attn_flash  grid (n_q / NH * DS, ceil(T / (FT * TT)), n_chunks), block 32 * NH * TT. The same output as
//             attn_partial (m, l, acc per (token, head, chunk), merged by attn_combine), for TT tiles of FT = 16
//             tokens and NH query heads of one KV head at once (NG > 1: of NG KV heads), so each K/V chunk is
//             read once per 16 NH TT query rows instead of once per row. A chunk is `chunk` positions (a
//             multiple of FB, chosen by the host so that a token has at most ATTN_MAX_SPLITS chunks). Wave w
//             takes head NH * (blockIdx.x / DS) + w % NH and token tile w / NH; n_q / n_kv must be a multiple
//             of NH / NG (the host checks). DS > 1 splits the head's D output dims over DS workgroups
//             (blockIdx.x % DS picks the slice; all compute the same scores): D 512 would not fit a wave's
//             registers otherwise.
//             Per block of FB = 16 positions: the K block and the transposed V block (its slice) are staged in
//             LDS (FP16, as cached), S = Q K^T (16 tokens x 16 positions, FP16 WMMA, FP32 accumulation; Q rounded
//             to FP16 as ggml's flash attention does), causal mask (and the sliding window: a token at position
//             pz sees pz - window < p <= pz), online softmax per token row (running max m and sum l; the row's
//             previous acc is rescaled by exp(m_old - m_new)), then acc += P V with P in FP16.
//             Operand layouts as tests/wmma_layout_verify: A row i / B column i in lanes i and i + 16; C
//             column lane % 16, element r = row 2 r + lane / 16.

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define FT 16          // query tokens per tile
#define FB 16          // positions per block
// NG: KV heads per workgroup (its NH query heads read KV head h / (n_q / n_kv)): 1 when the ratio n_q / n_kv
// is a multiple of NH, 2 for ratio 2 with NH 4 (Qwen3). Each block stages the NG KV heads' K / V rows; wave w
// reads those of its own KV head (local index (w % NH) * NG / NH).
template <int D, int NG, int NH, int TT, int DS, int KQ = 0, int SP = 1>
__device__ __forceinline__ void attn_flash_body(const float* q, const __half* K, const __half* V, int kv_stride,
                                                int kv_shift, long long kv_bstride, int pos0, int T, int n_q,
                                                int n_kv_heads, int chunk, float scale, float* part_ml,
                                                float* part_acc, int window) {
    // SP waves per (head, token tile) split its D: the Q K^T sum (each a 1 / SP of the dims, added through LDS) and
    // the output dims (DH / SP each), so a wave holds 1 / SP of the acc registers (D 256: 64 VGPRs, not 128 at the
    // 256 limit), which leaves room for the prefetch and gives SP times the waves per workgroup
    constexpr int NT = NH * TT, NW = NT * SP, NTH = 32 * NW, DH = D / DS;  // tiles, waves, threads, output dims
    constexpr int DT = DH / 16 / SP;                                        // a wave's 16-dim output tiles
    // KG: K is not staged; each lane reads its position's K row from global memory (L2) as the WMMA operand,
    // which frees LDS for more workgroups per WGP (the Q rows then padded by 4 halves, not 8). D 512 (2 -> 3
    // workgroups per WGP: Gemma-4 prefill +5%) and D 128 (Qwen3-0.6B +9%); neutral for D 256, which keeps the
    // staging (half the L2 reads of K). Not for a quantized D 128 cache: every wave would dequantize the same K
    // rows inside its WMMA loop; staged, they are dequantized once (and prefetched)
    constexpr bool KG = DS > 1 || (D <= 128 && KQ == 0);
    constexpr int QP = D + (KG ? 4 : 8);  // padded LDS row of Q / K (halves)
    __shared__ __attribute__((aligned(16))) _Float16 sQ[NT][FT][QP];
    __shared__ __attribute__((aligned(16))) _Float16 sK[KG ? 1 : NG][KG ? 1 : FB][KG ? 8 : QP];
    __shared__ __attribute__((aligned(16))) _Float16 sVt[NG][DH][FB];  // V blocks transposed: [dim][position]
    __shared__ __attribute__((aligned(16))) _Float16 sP[NT][FT][FB];
    __shared__ float sS[SP > 1 ? NW : 1][8][32];  // the waves' partial scores (SP > 1)
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, i = lane & 15, hi = lane >> 4;
    const int sp = w % SP, wt = w / SP;  // this wave's part of the tile's D, its tile
    const int hg = blockIdx.x / DS, ds = blockIdx.x % DS, hw = wt % NH, tw = wt / NH;
    const int h = hg * NH + hw, g0 = hg * NH / (n_q / n_kv_heads), gw = hw * NG / NH;
    const int zt = blockIdx.y * FT * TT, z0 = zt + FT * tw, c = blockIdx.z, n_chunks = gridDim.z;
    const int zmax = min(T, zt + FT * TT) - 1;              // last token of the workgroup's tiles
    const int npos = min(chunk, pos0 + zmax + 1 - c * chunk);  // chunk positions any token of the tiles sees
    const int nb = npos > 0 ? (npos + FB - 1) / FB : 0;
    const int lo0 = window > 0 ? pos0 + zt + 1 - window : 0;  // the first token's first position
    const int drow = KQ ? kvq_drow<KQ>(kv_stride) : 0;        // quantized rows: values per row

    // This tile's Q rows (tokens z0 .. z0 + 15 of head h) to FP16 (its SP waves share them); rows past T are zero
    for (int idx = lane + 32 * sp; idx < FT * D / 4; idx += 32 * SP) {
        const int rr = idx / (D / 4), d = 4 * (idx % (D / 4));
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (z0 + rr < T) x = *(const float4*)(q + ((size_t)(z0 + rr) * n_q + h) * D + d);
        sQ[wt][rr][d] = (_Float16)x.x, sQ[wt][rr][d + 1] = (_Float16)x.y;
        sQ[wt][rr][d + 2] = (_Float16)x.z, sQ[wt][rr][d + 3] = (_Float16)x.w;
    }

    float m[8], l[8];
    v8f acc[DT];
#pragma unroll
    for (int r = 0; r < 8; ++r) m[r] = -INFINITY, l[r] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) acc[dt] = (v8f){};

    // Stage the NG KV heads' K blocks (16 positions x D dims; 8 halves per uint4, consecutive lanes on consecutive
    // dims) and V blocks transposed (16 x the DH dims of slice ds): a thread takes positions 2 pp and 2 pp + 1 of 8
    // dims and writes each dim's pair as one dword, consecutive lanes on consecutive pairs (single halves from lanes
    // 8 dims apart hit one LDS bank 32 times). Positions no token of the tiles sees are zeroed: they are masked, but
    // 0 * (stale V) must stay 0. Block b + 1's raw words are fetched into registers before block b is computed
    // (kfetch / vfetch), so that their loads are in flight meanwhile: a workgroup takes most of a CU's LDS, so no
    // other one hides them. The loads are unconditional (the last position's row is read instead, then dropped): a
    // load in a branch is waited for before the branch ends, which would chain them.
    // PF: prefetch block b + 1 during block b. Not with DH > 128: those kernels hold 128 VGPRs of acc and are at the
    // 256 limit already (the prefetch registers spilled: Qwen3.6-35B prefill -5%); Qwen3-30B (D 128) +17% at 31k
    constexpr bool PF = DH / SP <= 128;
    constexpr int NKF = KG ? 0 : NG * FB * D / 8 / NTH;                  // K uint4 per thread per block
    constexpr int NVF = (NG * (FB / 2) * (DH / 8) + NTH - 1) / NTH;      // V position pairs per thread per block
    const int plast = pos0 + zmax;
    KvRaw kraw[NKF > 0 ? NKF : 1], vraw[NVF][2];
    auto fetch = [&](int P0) {
#pragma unroll
        for (int k = 0; k < NKF; ++k) {
            const int idx = tid + NTH * k, gl = idx / (FB * D / 8), r = idx % (FB * D / 8);
            const int p = r / (D / 8), d = 8 * (r % (D / 8));
            kraw[k] = kv_fetch<KQ>(K + kv_row(min(P0 + p, plast), kv_stride, kv_shift, kv_bstride), (g0 + gl) * D + d,
                                   drow);
        }
#pragma unroll
        for (int k = 0; k < NVF; ++k) {
            const int idx = min(tid + NTH * k, NG * (FB / 2) * (DH / 8) - 1);  // (D 64: fewer tasks than threads)
            const int pp = idx % (FB / 2), r = idx / (FB / 2);
            const int gl = r / (DH / 8), d = 8 * (r % (DH / 8));
            const int col = (g0 + gl) * D + ds * DH + d;
            vraw[k][0] = kv_fetch<KQ>(V + kv_row(min(P0 + 2 * pp, plast), kv_stride, kv_shift, kv_bstride), col, drow);
            vraw[k][1] = kv_fetch<KQ>(V + kv_row(min(P0 + 2 * pp + 1, plast), kv_stride, kv_shift, kv_bstride), col,
                                      drow);
        }
    };
    auto stage = [&](int P0) {
#pragma unroll
        for (int k = 0; k < NKF; ++k) {
            const int idx = tid + NTH * k, gl = idx / (FB * D / 8), r = idx % (FB * D / 8);
            const int p = r / (D / 8), d = 8 * (r % (D / 8));
            uint4 ku = kv_unpack<KQ>(kraw[k]);
            if (P0 + p > plast) ku = make_uint4(0, 0, 0, 0);
            *(uint4*)&sK[gl][p][d] = ku;
        }
#pragma unroll
        for (int k = 0; k < NVF; ++k) {
            const int idx = tid + NTH * k;
            if (idx >= NG * (FB / 2) * (DH / 8)) break;
            const int pp = idx % (FB / 2), r = idx / (FB / 2);
            const int gl = r / (DH / 8), d = 8 * (r % (DH / 8));
            uint4 v0 = kv_unpack<KQ>(vraw[k][0]), v1 = kv_unpack<KQ>(vraw[k][1]);
            if (P0 + 2 * pp > plast) v0 = make_uint4(0, 0, 0, 0);
            if (P0 + 2 * pp + 1 > plast) v1 = make_uint4(0, 0, 0, 0);
            const uint16_t* a0 = (const uint16_t*)&v0;
            const uint16_t* a1 = (const uint16_t*)&v1;
#pragma unroll
            for (int e = 0; e < 8; ++e) ((uint32_t*)&sVt[gl][d + e][0])[pp] = (uint32_t)a0[e] | ((uint32_t)a1[e] << 16);
        }
    };
    // the first block any token's window reaches (the ones before it contribute nothing)
    const int b0 = lo0 >= c * chunk + FB ? (lo0 - c * chunk - FB) / FB + 1 : 0;
    if (PF && b0 < nb) fetch(c * chunk + b0 * FB);
    // QR: the wave's Q fragments held in registers (SP > 1: its 1 / SP of D, 64 VGPRs for D 256) instead of read from
    // LDS for every block
    constexpr bool QR = SP > 1;
    constexpr int NQ = D / 16 / SP;
    v16h qa[QR ? NQ : 1];
    if constexpr (QR) {
        __syncthreads();  // sQ written
#pragma unroll
        for (int k = 0; k < NQ; ++k) qa[k] = *(const v16h*)&sQ[wt][i][(sp * NQ + k) * 16];
    }

#pragma unroll 1
    for (int b = b0; b < nb; ++b) {
        const int P0 = c * chunk + b * FB;
        if (!PF) fetch(P0);
        __syncthreads();  // the previous block's sK / sVt reads are done (and, the first time, sQ is written)
        stage(P0);
        __syncthreads();
        if (PF && b + 1 < nb) fetch(P0 + FB);  // block b + 1's loads, in flight while block b is computed

        // S = Q K^T for this wave's 16 tokens x 16 positions
        v8f s = {};
        if constexpr (KG) {  // B column i: position P0 + i's K row (zeros past the last position: masked)
            const bool kin = P0 + i <= pos0 + zmax;
            const __half* krow = K + kv_row(kin ? P0 + i : 0, kv_stride, kv_shift, kv_bstride);
#pragma unroll
            for (int kk = sp * (D / 16 / SP); kk < (sp + 1) * (D / 16 / SP); ++kk) {
                const v16h a = QR ? qa[QR ? kk - sp * NQ : 0] : *(const v16h*)&sQ[wt][i][kk * 16];
                v16h bk = {};
                const int e = (g0 + gw) * D + 16 * kk;
                uint4 lo = kv_load8<KQ>(krow, e, drow), hi8 = kv_load8<KQ>(krow, e + 8, drow);  // (unconditional)
                if (!kin) lo = hi8 = make_uint4(0, 0, 0, 0);
                _Float16* bh = (_Float16*)&bk;
                *(uint4*)bh = lo;
                *(uint4*)(bh + 8) = hi8;
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bk, s);
            }
        } else {
#pragma unroll
            for (int kk = sp * (D / 16 / SP); kk < (sp + 1) * (D / 16 / SP); ++kk) {
                const v16h a = QR ? qa[QR ? kk - sp * NQ : 0] : *(const v16h*)&sQ[wt][i][kk * 16];
                const v16h bk = *(const v16h*)&sK[gw][i][kk * 16];
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bk, s);
            }
        }
        if constexpr (SP > 1) {  // the tile's waves add their parts, in the same order in each
#pragma unroll
            for (int r = 0; r < 8; ++r) sS[w][r][lane] = s[r];
            __syncthreads();
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                float t = 0.0f;
#pragma unroll
                for (int j = 0; j < SP; ++j) t += sS[wt * SP + j][r][lane];
                s[r] = t;
            }
        }

        // Online softmax; this lane holds position P0 + i of token rows 2 r + hi (a row's 16 positions are one DPP
        // row of lanes: its max by DPP moves). l[r] is this lane's part of the row's sum, rescaled with the row
        // (alpha is the same in its 16 lanes) and added up across them once, after the last block
        float alpha[8];
        bool rescale = false;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int pz = pos0 + z0 + 2 * r + hi, pp = P0 + i;
            const float sv = pp <= pz && (window <= 0 || pp > pz - window) ? s[r] * scale : -INFINITY;
            const float mn = fmaxf(m[r], row16_max(sv));
            alpha[r] = mn == -INFINITY || mn == m[r] ? 1.0f : __expf(m[r] - mn);
            rescale |= alpha[r] != 1.0f;
            const float p = sv == -INFINITY ? 0.0f : __expf(sv - mn);
            l[r] = l[r] * alpha[r] + p;
            m[r] = mn;
            sP[wt][2 * r + hi][i] = (_Float16)p;  // (the tile's SP waves write the same values)
        }
        // The previous acc rescaled by exp(m_old - m_new): skipped (wave-uniform) when no row of the wave has a
        // new maximum, as in most blocks once the running maxima settle; the same values as multiplying by 1.0
        if (__ballot(rescale) != 0) {
#pragma unroll
            for (int dt = 0; dt < DT; ++dt)
#pragma unroll
                for (int r = 0; r < 8; ++r) acc[dt][r] *= alpha[r];
        }
        __syncthreads();  // sP rows are read by other lanes

        // acc += P V
        const v16h a = *(const v16h*)&sP[wt][i][0];
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            const v16h bv = *(const v16h*)&sVt[gw][(sp * DT + dt) * 16 + i][0];
            acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bv, acc[dt]);
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) l[r] = row16_sum(l[r]);  // (all lanes, before any leaves the loop below)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int z = z0 + 2 * r + hi;
        if (z >= T) continue;
        const size_t pc = ((size_t)z * n_q + h) * n_chunks + c;
        if (i == 0 && ds == 0 && sp == 0) {
            part_ml[2 * pc] = m[r];
            part_ml[2 * pc + 1] = l[r];
        }
#pragma unroll
        for (int dt = 0; dt < DT; ++dt) part_acc[pc * D + ds * DH + (sp * DT + dt) * 16 + i] = acc[dt][r];
    }
}

// attn_bidir  grid (n_q, n), block D: bidirectional attention (embedding models, attention_opts.bidir): token z of a
//             batch at positions 0 .. n - 1 attends to every position p of it, or to |p - z| <= half_window when
//             that is > 0 (llama.cpp's symmetric sliding window, half of n_swa). One workgroup per (head, token):
//             its threads score the positions (kept in LDS, n <= ATTN_BIDIR_MAX), softmax, then thread d sums
//             p's weight times V[p][d] over the positions. F16 cache rows.
#define ATTN_BIDIR_MAX 2048
template <int D>
__device__ __forceinline__ float bidir_reduce(float v, float* red, bool is_max) {
    for (int o = 16; o > 0; o >>= 1) {
        const float w = __shfl_xor(v, o, 32);
        v = is_max ? fmaxf(v, w) : v + w;
    }
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    __syncthreads();  // red may still be read by a previous reduction
    if (lane == 0) red[wave] = v;
    __syncthreads();
    v = red[0];
    for (int i = 1; i < D / 32; ++i) v = is_max ? fmaxf(v, red[i]) : v + red[i];
    return v;
}

template <int D>
__device__ void attn_bidir_body(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift,
                                long long kv_bstride, int n, int n_q, int n_kv, float scale, int half_window,
                                float* out) {
    __shared__ float qs[D];
    __shared__ float sc[ATTN_BIDIR_MAX];
    __shared__ float red[D / 32];
    const int h = blockIdx.x, z = blockIdx.y, t = threadIdx.x;
    const int hk = h / (n_q / n_kv);
    qs[t] = q[((size_t)z * n_q + h) * D + t];
    __syncthreads();
    const int lo = half_window > 0 ? max(0, z - half_window) : 0;
    const int hi = half_window > 0 ? min(n - 1, z + half_window) : n - 1;
    float mx = -INFINITY;
    for (int p = lo + t; p <= hi; p += D) {
        const __half* kr = K + kv_row(p, kv_stride, kv_shift, kv_bstride) + (size_t)hk * D;
        float s = 0.0f;
        for (int i = 0; i < D; i += 8) {
            const uint4 u = *(const uint4*)(kr + i);
            const __half2* k2 = (const __half2*)&u;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 kk = __half22float2(k2[j]);
                s += qs[i + 2 * j] * kk.x + qs[i + 2 * j + 1] * kk.y;
            }
        }
        s *= scale;
        sc[p - lo] = s;
        mx = fmaxf(mx, s);
    }
    mx = bidir_reduce<D>(mx, red, true);
    float sum = 0.0f;
    for (int p = lo + t; p <= hi; p += D) {
        const float e = __expf(sc[p - lo] - mx);
        sc[p - lo] = e;
        sum += e;
    }
    sum = bidir_reduce<D>(sum, red, false);  // (its first barrier also makes the weights visible)
    float acc = 0.0f;
    for (int p = lo; p <= hi; ++p)
        acc += sc[p - lo] * __half2float(V[kv_row(p, kv_stride, kv_shift, kv_bstride) + (size_t)hk * D + t]);
    out[((size_t)z * n_q + h) * D + t] = acc / sum;
}

// attn_partial for grouped-query attention (decode and small batches): one workgroup per KV head g and chunk of PW
// positions instead of per query head and 256 positions, scoring all R = n_q / n_kv of g's query heads from one
// read of each K row and accumulating V once for them (attn_partial reads the chunk's K and V R times). PW 64 (the
// host's choice when the caller's slots allow: 4x the workgroups, which a decode step needs to have enough loads in
// flight) or 256 (CH). The results as attn_partial's, per (query head, chunk), for attn_combine.
// grid (n_kv, n_chunks, T), block 256: CH / PW threads per position score its D dims, each 8-value pieces tp, tp +
// TPP, ..; for V, thread t takes dims 2 (t % (D / 2)) .. + 1 and every NPG-th position (NPG = 512 / D), the position
// groups added in LDS. A decode step is latency-bound (Qwen3-30B at 32k: 2,048 workgroups of 64 positions), so all
// of a thread's K pieces and its first V batch are loaded before anything waits: one memory round trip instead of
// a chain (q, then K, then V in 4 batches). R <= RMAX. KQ: a Q8_0 cache (int8 values times their block's scale).
template <int D, int RMAX, int PW, int KQ = 0>
__device__ __forceinline__ void attn_partial_gqa_body(const float* q, const __half* K, const __half* V, int kv_stride,
                                                      int kv_shift, long long kv_bstride, int pos0, int n_q,
                                                      int n_kv_heads, float scale, float* part_ml, float* part_acc,
                                                      int window) {
    constexpr int TPP = CH / PW, DP = D / TPP;  // threads per position, dims per thread
    constexpr int NK = DP / 8, KP = NK < 8 ? NK : 8;  // K pieces per thread, of them loaded up front
    constexpr int TPR = D / 2, NPG = CH / TPR;  // V: threads per row (2 dims each), position groups
    constexpr int NV = PW / NPG, VU = NV < 16 ? NV : 16;  // V positions per thread, per batch
    __shared__ float sq[RMAX][D];
    __shared__ float sw[RMAX][PW];
    __shared__ float sacc[NPG > 1 ? NPG - 1 : 1][RMAX][D];
    __shared__ float sm[RMAX], sl[RMAX];
    const int R = n_q / n_kv_heads;
    const int g = blockIdx.x, c = blockIdx.y, t = threadIdx.x, n_chunks = gridDim.y, z = blockIdx.z;
    const int n_kv = pos0 + z + 1;
    const int lo = window > 0 ? n_kv - window : 0;  // first position this token sees (sliding window)
    const int h0 = g * R;
    q += (size_t)z * n_q * D;
    part_ml += (size_t)2 * z * n_q * n_chunks;
    part_acc += (size_t)z * n_q * n_chunks * D;
    if (c * PW + PW <= lo) {  // the whole chunk is before the window (uniform): it contributes nothing
        for (int i = t; i < R * D; i += CH) part_acc[((size_t)(h0 + i / D) * n_chunks + c) * D + i % D] = 0.0f;
        if (t < R) {
            part_ml[2 * ((size_t)(h0 + t) * n_chunks + c)] = -INFINITY;
            part_ml[2 * ((size_t)(h0 + t) * n_chunks + c) + 1] = 0.0f;
        }
        return;
    }
    const int drow = KQ ? kvq_drow<KQ>(kv_stride) : 0;
    // 8 K values of piece k (as 8 floats; KQ: times the block's scale)
    auto k_load = [&](const __half* krow, int k, uint4& u, float& sc) {
        const int e = g * D + 8 * k;
        if constexpr (KQ == 2) {
            u.x = *(const uint32_t*)((const uint8_t*)krow + e / 2);
            sc = __half2float(kvq_scales<2>(krow, drow)[e / 32]);
        } else if constexpr (KQ == 1) {
            const uint2 w = *(const uint2*)((const int8_t*)krow + e);
            u.x = w.x, u.y = w.y;
            sc = __half2float(kvq_scales<1>(krow, drow)[e / 32]);
        } else {
            u = *(const uint4*)(krow + e);
        }
    };
    auto k_unpack = [&](const uint4& u, float sc, float* f) {
        if constexpr (KQ == 2) {
#pragma unroll
            for (int j = 0; j < 8; ++j) f[j] = (float)((int)((u.x >> (8 * (j & 3) + 4 * (j >> 2))) & 15) - 8) * sc;
        } else if constexpr (KQ == 1) {
            const int8_t* qv = (const int8_t*)&u;
#pragma unroll
            for (int j = 0; j < 8; ++j) f[j] = (float)qv[j] * sc;
        } else {
            const __half2* h2 = (const __half2*)&u;
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                const float2 x = __half22float2(h2[k2]);
                f[2 * k2] = x.x, f[2 * k2 + 1] = x.y;
            }
        }
    };
    // V: dims e0, e0 + 1 of position p0 + iu (0 past np). Loads are never conditional (a row inside the cache is read
    // and the value dropped): a load in a branch is waited for before the branch ends, which chains them all.
    const int d0 = (t % TPR) * 2, pg = t / TPR, e0 = g * D + d0;
    const int p0 = c * PW, i0 = max(0, lo - p0), np = max(0, min(PW, n_kv - p0));
    const int ib = i0 + (pg - i0 % NPG + NPG) % NPG;  // this thread's first position
    auto v_load = [&](int iu, float2& x) {
        const __half* row = V + kv_row(p0 + max(0, min(iu, np - 1)), kv_stride, kv_shift, kv_bstride);
        if constexpr (KQ == 2) {  // dims e0, e0 + 1: the same nibble of two adjacent bytes
            const float sc = __half2float(kvq_scales<2>(row, drow)[e0 / 32]);
            const int w = *(const unsigned short*)((const uint8_t*)row + (e0 >> 3) * 4 + (e0 & 3)) >> (e0 & 4);
            x = make_float2((float)((w & 15) - 8) * sc, (float)(((w >> 8) & 15) - 8) * sc);
        } else if constexpr (KQ == 1) {
            const int8_t* b = (const int8_t*)row;
            const float sc = __half2float(kvq_scales<1>(row, drow)[e0 / 32]);
            const short w = *(const short*)(b + e0);
            x = make_float2((float)(int8_t)w * sc, (float)(int8_t)(w >> 8) * sc);
        } else {
            x = __half22float2(*(const __half2*)(row + e0));
        }
        if (iu >= np) x = make_float2(0.0f, 0.0f);
    };

    // Issue the loads: K pieces 0 .. KP - 1 of position p, the first V batch, q
    const int pl = t / TPP, tp = t % TPP, p = c * PW + pl;
    const bool valid = p < n_kv && p >= lo;
    const __half* krow = K + kv_row(min(p, n_kv - 1), kv_stride, kv_shift, kv_bstride);  // (a row inside the cache)
    uint4 ku[KP];
    float ks[KP] = {};
#pragma unroll
    for (int i = 0; i < KP; ++i) k_load(krow, i * TPP + tp, ku[i], ks[i]);
    float2 v[VU];
#pragma unroll
    for (int u = 0; u < VU; ++u) v_load(ib + u * NPG, v[u]);
    for (int i = t; i < R * D; i += CH) sq[i / D][i % D] = q[(size_t)h0 * D + i];
    __syncthreads();

    // Scores of position p for the R heads
    float s[RMAX];
#pragma unroll
    for (int r = 0; r < RMAX; ++r) s[r] = 0.0f;
    if (valid) {
#pragma unroll
        for (int i = 0; i < NK; ++i) {
            float f[8];
            if (i < KP) {
                k_unpack(ku[i], ks[i], f);
            } else {
                uint4 u;
                float sc = 0.0f;
                k_load(krow, i * TPP + tp, u, sc);
                k_unpack(u, sc, f);
            }
            const int dk = 8 * (i * TPP + tp);
#pragma unroll
            for (int r = 0; r < RMAX; ++r)
                if (r < R)
#pragma unroll
                    for (int j = 0; j < 8; ++j) s[r] += f[j] * sq[r][dk + j];
        }
    }
#pragma unroll
    for (int r = 0; r < RMAX; ++r)
        if (r < R) {
#pragma unroll
            for (int off = 1; off < TPP; off <<= 1) s[r] += __shfl_xor(s[r], off, 32);  // (TPP adjacent lanes)
            if (tp == 0) sw[r][pl] = valid ? s[r] * scale : -INFINITY;
        }
    __syncthreads();

    // Softmax statistics per head (a wave per head); the scores become their exp(s - m)
    const int w = t >> 5, lane = t & 31;
    for (int r = w; r < R; r += CH / 32) {
        float m = -INFINITY;
        for (int k = lane; k < PW; k += 32) m = fmaxf(m, sw[r][k]);
        m = wave_max(m);
        float l = 0.0f;
        for (int k = lane; k < PW; k += 32) {
            const float e = sw[r][k] > -INFINITY ? __expf(sw[r][k] - m) : 0.0f;
            sw[r][k] = e;
            l += e;
        }
        l = wave_sum(l);
        if (lane == 0) sm[r] = m, sl[r] = l;
    }
    __syncthreads();

    // acc[r][d0, d0 + 1] over this thread's positions ib, ib + NPG, ..; batch after batch of VU, the first loaded
    float2 a[RMAX];
#pragma unroll
    for (int r = 0; r < RMAX; ++r) a[r] = make_float2(0.0f, 0.0f);
    for (int i = ib;; ) {
#pragma unroll
        for (int u = 0; u < VU; ++u) {
            const int iu = min(i + u * NPG, PW - 1);  // (v is 0 past np)
#pragma unroll
            for (int r = 0; r < RMAX; ++r)
                if (r < R) {
                    const float wt = sw[r][iu];
                    a[r].x += wt * v[u].x, a[r].y += wt * v[u].y;
                }
        }
        i += VU * NPG;
        if (VU == NV || i >= np) break;
#pragma unroll
        for (int u = 0; u < VU; ++u) v_load(i + u * NPG, v[u]);
    }
    if (NPG > 1 && pg > 0)
#pragma unroll
        for (int r = 0; r < RMAX; ++r)
            if (r < R) sacc[pg - 1][r][d0] = a[r].x, sacc[pg - 1][r][d0 + 1] = a[r].y;
    __syncthreads();
    if (pg == 0)
#pragma unroll
        for (int r = 0; r < RMAX; ++r)
            if (r < R) {
                float2 x = a[r];
#pragma unroll
                for (int k = 1; k < NPG; ++k) x.x += sacc[k - 1][r][d0], x.y += sacc[k - 1][r][d0 + 1];
                float* o = part_acc + ((size_t)(h0 + r) * n_chunks + c) * D + d0;
                o[0] = x.x, o[1] = x.y;
            }
    if (t < R) {
        part_ml[2 * ((size_t)(h0 + t) * n_chunks + c)] = sm[t];
        part_ml[2 * ((size_t)(h0 + t) * n_chunks + c) + 1] = sl[t];
    }
}

// attn_partial_gqa<RMAX>_<D> (256-position chunks) and attn_partial_gqa<RMAX>s_<D> (64); NAME_q8 for a Q8_0 cache
#define ATTN_GQA_KERNEL_KQ(NAME, D, RMAX, PW, KQ)                                                             \
    extern "C" __global__ void __launch_bounds__(CH)                                                         \
        NAME(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift, long long kv_bstride, \
             int pos0, int n_q, int n_kv_heads, float scale, float* part_ml, float* part_acc, int window) {    \
        attn_partial_gqa_body<D, RMAX, PW, KQ>(q, K, V, kv_stride, kv_shift, kv_bstride, pos0, n_q,           \
                                               n_kv_heads, scale, part_ml, part_acc, window);                \
    }
#define ATTN_GQA_KERNEL(NAME, D, RMAX, PW)                                                                    \
    ATTN_GQA_KERNEL_KQ(NAME, D, RMAX, PW, 0)                                                                  \
    ATTN_GQA_KERNEL_KQ(NAME##_q8, D, RMAX, PW, 1)                                                             \
    ATTN_GQA_KERNEL_KQ(NAME##_q4, D, RMAX, PW, 2)
#define ATTN_GQA_KERNELS(D)                                                                                   \
    ATTN_GQA_KERNEL(attn_partial_gqa8_##D, D, 8, 256)                                                         \
    ATTN_GQA_KERNEL(attn_partial_gqa16_##D, D, 16, 256)                                                       \
    ATTN_GQA_KERNEL(attn_partial_gqa8s_##D, D, 8, 64)                                                         \
    ATTN_GQA_KERNEL(attn_partial_gqa16s_##D, D, 16, 64)
ATTN_GQA_KERNELS(64)
ATTN_GQA_KERNELS(128)
ATTN_GQA_KERNELS(256)

// The kernels for each head size: attn_<kernel>_64 (Qwen2.5-0.5B), attn_<kernel>_128 (Qwen2 / Qwen3), attn_<kernel>_256
// (Qwen3.5 and later, Gemma-4's sliding layers), attn_<kernel>_512 (Gemma-4's global layers; its flash kernel is
// attn_flash_512_h2s2)
#define ATTN_DECODE_KERNELS(D)                                                                                \
    extern "C" __global__ void __launch_bounds__(D)                                                          \
        attn_prep_##D(const float* qfull, const float* k, const float* v, const float* q_norm,                \
                      const float* k_norm, float* q_out, __half* K, __half* V, int kv_stride, int kv_shift,    \
                      long long kv_bstride, int n_q, int n_kv, int n_rot, int pos0, float freq_base, float eps, \
                      int gated, int rot_pairs, int v_norm, int qk_norm, int kv_quant, const float* bq,      \
                      const float* bk, const float* bv, int rope_norm, const float* ff) {                    \
        attn_prep_body<D>(qfull, k, v, q_norm, k_norm, q_out, K, V, kv_stride, kv_shift, kv_bstride, n_q, n_kv, \
                          n_rot, pos0, freq_base, eps, gated, rot_pairs, v_norm, qk_norm, kv_quant, bq, bk, bv, \
                          rope_norm, ff);                                                                    \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(CH)                                                         \
        attn_partial_##D(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift,       \
                         long long kv_bstride, int pos0, int n_q, int n_kv_heads, float scale, float* part_ml, \
                         float* part_acc, int window) {                                                      \
        attn_partial_body<D>(q, K, V, kv_stride, kv_shift, kv_bstride, pos0, n_q, n_kv_heads, scale, part_ml, \
                             part_acc, window);                                                              \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(CH)                                                         \
        attn_partial_##D##_q8(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift,  \
                              long long kv_bstride, int pos0, int n_q, int n_kv_heads, float scale,           \
                              float* part_ml, float* part_acc, int window) {                                 \
        attn_partial_body<D, 1>(q, K, V, kv_stride, kv_shift, kv_bstride, pos0, n_q, n_kv_heads, scale,       \
                                part_ml, part_acc, window);                                                  \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(CH)                                                         \
        attn_partial_##D##_q4(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift,  \
                              long long kv_bstride, int pos0, int n_q, int n_kv_heads, float scale,           \
                              float* part_ml, float* part_acc, int window) {                                 \
        attn_partial_body<D, 2>(q, K, V, kv_stride, kv_shift, kv_bstride, pos0, n_q, n_kv_heads, scale,       \
                                part_ml, part_acc, window);                                                  \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                        \
        attn_combine_##D(const float* part_ml, const float* part_acc, int n_chunks, int n_q, const float* qfull, \
                         float* out, Q8Out o, int gated, int vrot) {                                         \
        attn_combine_body<D>(part_ml, part_acc, n_chunks, n_q, qfull, out, o, gated, vrot);                   \
    }
#define ATTN_BIDIR_KERNEL(D)                                                                                  \
    extern "C" __global__ void __launch_bounds__(D)                                                          \
        attn_bidir_##D(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift,         \
                       long long kv_bstride, int n, int n_q, int n_kv, float scale, int half_window, float* out) { \
        attn_bidir_body<D>(q, K, V, kv_stride, kv_shift, kv_bstride, n, n_q, n_kv, scale, half_window, out);   \
    }
#define ATTN_KERNELS(D)                                                                                       \
    ATTN_DECODE_KERNELS(D)                                                                                    \
    ATTN_BIDIR_KERNEL(D)                                                                                      \
    ATTN_FLASH_KERNEL_NAMED(attn_flash_##D, D, 1, 4, 1, 1)

// attn_flash_<D> (ratio n_q / n_kv a multiple of 4: 4 heads x 16 tokens), attn_flash_<D>_g2 (ratio 2: 4 heads of
// two KV heads), attn_flash_<D>_h2t2 (ratio a multiple of 2: 2 heads x 32 tokens), attn_flash_<D>_h2s2 (2 heads
// x 16 tokens, D split over 2 workgroups)
// (each also as NAME_q8 / NAME_q4 for a Q8_0 / Q4_0 KV cache). ATTN_FLASH_SP: waves per (head, token tile), 2 for
// D 256 (the host's attention.cpp launches as many: attn_flash_sp)
#define ATTN_FLASH_SP(D, DS) ((D) == 256 && (DS) == 1 ? 2 : 1)
#define ATTN_FLASH_KERNEL_KQ(NAME, D, NG, NH, TT, DS, KQ)                                                     \
    extern "C" __global__ void __launch_bounds__(32 * (NH) * (TT) * ATTN_FLASH_SP(D, DS))                    \
        NAME(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift, long long kv_bstride, \
             int pos0, int T, int n_q, int n_kv_heads, int chunk, float scale, float* part_ml, float* part_acc,  \
             int window) {                                                                                   \
        attn_flash_body<D, NG, NH, TT, DS, KQ, ATTN_FLASH_SP(D, DS)>(q, K, V, kv_stride, kv_shift, kv_bstride, \
                                                                     pos0, T, n_q, n_kv_heads, chunk, scale, \
                                                                     part_ml, part_acc, window);             \
    }
#define ATTN_FLASH_KERNEL_NAMED(NAME, D, NG, NH, TT, DS)                                                      \
    ATTN_FLASH_KERNEL_KQ(NAME, D, NG, NH, TT, DS, 0)                                                          \
    ATTN_FLASH_KERNEL_KQ(NAME##_q8, D, NG, NH, TT, DS, 1)                                                     \
    ATTN_FLASH_KERNEL_KQ(NAME##_q4, D, NG, NH, TT, DS, 2)

ATTN_KERNELS(64)
ATTN_KERNELS(128)
ATTN_KERNELS(256)
ATTN_DECODE_KERNELS(512)
ATTN_BIDIR_KERNEL(512)
// Two KV heads per workgroup (Qwen3: ratio 2): head_dim 128 only (256 would need over 64 KB of LDS)
ATTN_FLASH_KERNEL_NAMED(attn_flash_128_g2, 128, 2, 4, 1, 1)
// Gemma-4: sliding layers (D 256, ratio 2), global layers (D 512, ratio 8; 59 KB of LDS)
ATTN_FLASH_KERNEL_NAMED(attn_flash_256_h2t2, 256, 1, 2, 2, 1)
ATTN_FLASH_KERNEL_NAMED(attn_flash_512_h2s2, 512, 1, 2, 1, 2)
// Other ratios (Qwen2 / Qwen2.5: 6 and 7): 2 heads x 32 tokens (ratio a multiple of 2), 1 head x 64 tokens (any)
ATTN_FLASH_KERNEL_NAMED(attn_flash_64_h2t2, 64, 1, 2, 2, 1)
ATTN_FLASH_KERNEL_NAMED(attn_flash_128_h2t2, 128, 1, 2, 2, 1)
ATTN_FLASH_KERNEL_NAMED(attn_flash_64_h1t4, 64, 1, 1, 4, 1)
ATTN_FLASH_KERNEL_NAMED(attn_flash_128_h1t4, 128, 1, 1, 4, 1)
ATTN_FLASH_KERNEL_NAMED(attn_flash_256_h1t4, 256, 1, 1, 4, 1)
