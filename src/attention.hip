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

// Q8_0 KV cache rows (attention_opts.kv_q8): d_row int8 values, then their fp16 scales, one per 32 values (as
// ggml's q8_0: d = amax / 127, q = round(x / d)); kv_stride (in halves) = d_row * 17 / 32. Element e of a row at
// row (as __half*): ((int8*)row)[e] * scale[e / 32].
__device__ __forceinline__ int kvq_drow(int kv_stride) { return kv_stride * 32 / 17; }

// 8 consecutive values e .. e + 7 (e % 8 == 0) of a cache row as fp16, in a uint4 (KQ: dequantized)
template <bool KQ>
__device__ __forceinline__ uint4 kv_load8(const __half* row, int e, int drow) {
    if constexpr (!KQ) {
        return *(const uint4*)(row + e);
    } else {
        const int8_t* b = (const int8_t*)row;
        const uint2 q = *(const uint2*)(b + e);
        const float sc = __half2float(((const __half*)(b + drow))[e / 32]);
        const int8_t* qv = (const int8_t*)&q;
        uint4 r;
        __half* h = (__half*)&r;
#pragma unroll
        for (int i = 0; i < 8; ++i) h[i] = __float2half((float)qv[i] * sc);
        return r;
    }
}

// One value of a cache row as float
template <bool KQ>
__device__ __forceinline__ float kv_value(const __half* row, int e, int drow) {
    if constexpr (!KQ) {
        return __half2float(row[e]);
    } else {
        const int8_t* b = (const int8_t*)row;
        return (float)b[e] * __half2float(((const __half*)(b + drow))[e / 32]);
    }
}

__device__ __forceinline__ float wave_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 32));
    return v;
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

// RMSNorm (gamma w; none if NULL; no norm at all unless norm) then NeoX RoPE of one head held one value per thread
// (D threads); returns this thread's value. Pairs (i, i + n_rot / 2) with i < rot_pairs rotate (theta_i = pos *
// base^(-2 i / n_rot)), the others stay as they are; n_rot 0: no RoPE.
template <int D>
__device__ float norm_rope(float x, const float* w, int n_rot, int rot_pairs, int pos, float freq_base, float eps,
                           bool norm = true) {
    __shared__ float v[D];
    const int d = threadIdx.x;
    if (norm) {  // uniform
        const float ss = block_sum<D>(x * x);
        x = x * rsqrtf(ss / D + eps) * (w ? w[d] : 1.0f);
    }
    v[d] = x;
    __syncthreads();
    const int half = n_rot / 2;
    if (d < n_rot && (d < half ? d : d - half) < rot_pairs) {
        const int i = d < half ? d : d - half;
        const float theta = (float)pos * powf(freq_base, -2.0f * (float)i / (float)n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const float x0 = v[i], x1 = v[i + half];
        x = d < half ? x0 * cs - x1 * sn : x0 * sn + x1 * cs;
    }
    __syncthreads();
    return x;
}

template <int D>
__device__ __forceinline__ void attn_prep_body(const float* qfull, const float* k, const float* v, const float* q_norm,
                                               const float* k_norm, float* q_out, __half* K, __half* V, int kv_stride,
                                               int kv_shift, long long kv_bstride, int n_q, int n_kv, int n_rot,
                                               int pos0, float freq_base, float eps, int gated, int rot_pairs,
                                               int v_norm, int qk_norm, int kv_q8) {
    const int b = blockIdx.x, d = threadIdx.x, z = blockIdx.y, pos = pos0 + z;
    const int qs = gated ? 2 * D : D;  // a head's stride in attn_q's output
    qfull += (size_t)z * n_q * qs;
    q_out += (size_t)z * n_q * D;
    k += (size_t)z * n_kv * D;
    v += (size_t)z * n_kv * D;
    __half* k_row = K + kv_row(pos, kv_stride, kv_shift, kv_bstride);
    __half* v_row = V + kv_row(pos, kv_stride, kv_shift, kv_bstride);
    if (b < n_q) {
        const float x = qfull[(size_t)b * qs + d];
        q_out[b * D + d] = norm_rope<D>(x, q_norm, n_rot, rot_pairs, pos, freq_base, eps, qk_norm);
    } else {
        const int g = b - n_q;
        const float x = norm_rope<D>(k[g * D + d], k_norm, n_rot, rot_pairs, pos, freq_base, eps, qk_norm);
        const float vv = v[g * D + d];
        const float vx = v_norm ? norm_rope<D>(vv, nullptr, 0, 0, pos, freq_base, eps) : vv;
        if (kv_q8) {  // uniform: a wave's 32 dims are one block
            const int drow = kvq_drow(kv_stride), e = g * D + d;
            const float ka = wave_max(fabsf(x)), va = wave_max(fabsf(vx));
            const float kd = ka / 127.0f, vd = va / 127.0f;
            ((int8_t*)k_row)[e] = (int8_t)(int)rintf(kd != 0.0f ? x * (1.0f / kd) : 0.0f);
            ((int8_t*)v_row)[e] = (int8_t)(int)rintf(vd != 0.0f ? vx * (1.0f / vd) : 0.0f);
            if ((d & 31) == 0) {
                ((__half*)((int8_t*)k_row + drow))[e / 32] = __float2half(kd);
                ((__half*)((int8_t*)v_row + drow))[e / 32] = __float2half(vd);
            }
        } else {
            k_row[g * D + d] = __float2half(x);
            v_row[g * D + d] = __float2half(vx);
        }
    }
}

template <int D, bool KQ = false>
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
        if constexpr (KQ) {  // per 32-value block: the int8 dot with q, times the block's scale
            const int8_t* kb = (const int8_t*)(K + kv_row(p, kv_stride, kv_shift, kv_bstride));
            const __half* ks = (const __half*)(kb + kvq_drow(kv_stride)) + g * (D / 32);
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
        } else {
            const uint4* kr = (const uint4*)(K + kv_row(p, kv_stride, kv_shift, kv_bstride) + (size_t)g * D);
#pragma unroll 4
            for (int i = 0; i < D / 8; ++i) {
                const uint4 u = kr[i];
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
            const int drow = kvq_drow(kv_stride);
            for (int i = i0; i < np; ++i)
                acc += sw[i] * kv_value<true>(V + kv_row(p0 + i, kv_stride, kv_shift, kv_bstride), g * D + d, drow);
        } else {
            const __half* vb = V + (size_t)g * D + d;
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
                                                  const float* qfull, float* out, Q8Out o, int gated) {
    __shared__ Q8Shared sh;
    // workgroup: 256 values of the token's n_q * D (256 / D heads, or 256 / D of one head for D 512)
    const int e = blockIdx.x * 256 + threadIdx.x, h = e / D, d = e % D, z = blockIdx.y;
    const int nb = n_q * D / 256;  // 256-value blocks per token
    part_ml += (size_t)2 * z * n_q * n_chunks;
    part_acc += (size_t)z * n_q * n_chunks * D;
    qfull += (size_t)z * (gated ? 2 : 1) * n_q * D;
    out += (size_t)z * n_q * D;
    const float* ml = part_ml + (size_t)2 * h * n_chunks;
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, ml[2 * c]);
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float w = __expf(ml[2 * c] - M);
        num += w * part_acc[((size_t)h * n_chunks + c) * D + d];
        den += w * ml[2 * c + 1];
    }
    float r = num / den;
    if (gated) r /= 1.0f + __expf(-qfull[(size_t)h * 2 * D + D + d]);
    out[h * D + d] = r;
    if (o.qK) q8_block_value(r, true, z * nb + blockIdx.x, threadIdx.x, sh, o);  // uniform: one 256-block
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
template <int D, int NG, int NH, int TT, int DS, bool KQ = false>
__device__ __forceinline__ void attn_flash_body(const float* q, const __half* K, const __half* V, int kv_stride,
                                                int kv_shift, long long kv_bstride, int pos0, int T, int n_q,
                                                int n_kv_heads, int chunk, float scale, float* part_ml,
                                                float* part_acc, int window) {
    constexpr int NW = NH * TT, NTH = 32 * NW, DH = D / DS;  // waves, threads, output dims per workgroup
    // KG: K is not staged; each lane reads its position's K row from global memory (L2) as the WMMA operand,
    // which frees LDS for more workgroups per WGP (the Q rows then padded by 4 halves, not 8). D 512 (2 -> 3
    // workgroups per WGP: Gemma-4 prefill +5%) and D 128 (Qwen3-0.6B +9%); neutral for D 256, which keeps the
    // staging (half the L2 reads of K)
    constexpr bool KG = DS > 1 || D <= 128;
    constexpr int QP = D + (KG ? 4 : 8);  // padded LDS row of Q / K (halves)
    __shared__ __attribute__((aligned(16))) _Float16 sQ[NW][FT][QP];
    __shared__ __attribute__((aligned(16))) _Float16 sK[KG ? 1 : NG][KG ? 1 : FB][KG ? 8 : QP];
    __shared__ __attribute__((aligned(16))) _Float16 sVt[NG][DH][FB];  // V blocks transposed: [dim][position]
    __shared__ __attribute__((aligned(16))) _Float16 sP[NW][FT][FB];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, i = lane & 15, hi = lane >> 4;
    const int hg = blockIdx.x / DS, ds = blockIdx.x % DS, hw = w % NH, tw = w / NH;
    const int h = hg * NH + hw, g0 = hg * NH / (n_q / n_kv_heads), gw = hw * NG / NH;
    const int zt = blockIdx.y * FT * TT, z0 = zt + FT * tw, c = blockIdx.z, n_chunks = gridDim.z;
    const int zmax = min(T, zt + FT * TT) - 1;              // last token of the workgroup's tiles
    const int npos = min(chunk, pos0 + zmax + 1 - c * chunk);  // chunk positions any token of the tiles sees
    const int nb = npos > 0 ? (npos + FB - 1) / FB : 0;
    const int lo0 = window > 0 ? pos0 + zt + 1 - window : 0;  // the first token's first position
    const int drow = KQ ? kvq_drow(kv_stride) : 0;            // Q8_0 rows: values per row

    // This wave's Q rows (tokens z0 .. z0 + 15 of head h) to FP16; rows past T are zero
    for (int idx = lane; idx < FT * D / 4; idx += 32) {
        const int rr = idx / (D / 4), d = 4 * (idx % (D / 4));
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (z0 + rr < T) x = *(const float4*)(q + ((size_t)(z0 + rr) * n_q + h) * D + d);
        sQ[w][rr][d] = (_Float16)x.x, sQ[w][rr][d + 1] = (_Float16)x.y;
        sQ[w][rr][d + 2] = (_Float16)x.z, sQ[w][rr][d + 3] = (_Float16)x.w;
    }

    float m[8], l[8];
    v8f acc[DH / 16];
#pragma unroll
    for (int r = 0; r < 8; ++r) m[r] = -INFINITY, l[r] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < DH / 16; ++dt) acc[dt] = (v8f){};

#pragma unroll 1
    for (int b = 0; b < nb; ++b) {
        const int P0 = c * chunk + b * FB;
        if (P0 + FB <= lo0) continue;  // before every token's window (uniform)
        __syncthreads();  // the previous block's sK / sVt reads are done (and, the first time, sQ is written)
        // Stage the NG KV heads' K blocks (16 positions x D dims; 8 halves per uint4, consecutive lanes on
        // consecutive dims) and V blocks transposed (16 x the DH dims of slice ds): a thread takes positions 2 pp
        // and 2 pp + 1 of 8 dims and writes each dim's pair as one dword, consecutive lanes on consecutive pairs
        // (single halves from lanes 8 dims apart hit one LDS bank 32 times). Positions no token of the tiles
        // sees are zeroed: they are masked, but 0 * (stale V) must stay 0.
#pragma unroll
        for (int k = 0; k < (KG ? 0 : NG * FB * D / 8 / NTH); ++k) {
            const int idx = tid + NTH * k, gl = idx / (FB * D / 8), r = idx % (FB * D / 8);
            const int p = r / (D / 8), d = 8 * (r % (D / 8));
            uint4 ku = make_uint4(0, 0, 0, 0);
            if (P0 + p <= pos0 + zmax)
                ku = kv_load8<KQ>(K + kv_row(P0 + p, kv_stride, kv_shift, kv_bstride), (g0 + gl) * D + d, drow);
            *(uint4*)&sK[gl][p][d] = ku;
        }
#pragma unroll
        for (int k = 0; k < NG * (FB / 2) * (DH / 8) / NTH; ++k) {
            const int idx = tid + NTH * k, pp = idx % (FB / 2), r = idx / (FB / 2);
            const int gl = r / (DH / 8), d = 8 * (r % (DH / 8));
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;
            const int col = (g0 + gl) * D + ds * DH + d;
            if (P0 + 2 * pp <= pos0 + zmax) v0 = kv_load8<KQ>(V + kv_row(P0 + 2 * pp, kv_stride, kv_shift, kv_bstride), col, drow);
            if (P0 + 2 * pp + 1 <= pos0 + zmax)
                v1 = kv_load8<KQ>(V + kv_row(P0 + 2 * pp + 1, kv_stride, kv_shift, kv_bstride), col, drow);
            const uint16_t* a0 = (const uint16_t*)&v0;
            const uint16_t* a1 = (const uint16_t*)&v1;
#pragma unroll
            for (int e = 0; e < 8; ++e) ((uint32_t*)&sVt[gl][d + e][0])[pp] = (uint32_t)a0[e] | ((uint32_t)a1[e] << 16);
        }
        __syncthreads();

        // S = Q K^T for this wave's 16 tokens x 16 positions
        v8f s = {};
        if constexpr (KG) {  // B column i: position P0 + i's K row (zeros past the last position: masked)
            const bool kin = P0 + i <= pos0 + zmax;
            const __half* krow = K + kv_row(kin ? P0 + i : 0, kv_stride, kv_shift, kv_bstride);
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk) {
                const v16h a = *(const v16h*)&sQ[w][i][kk * 16];
                v16h bk = {};
                if (kin) {
                    const int e = (g0 + gw) * D + 16 * kk;
                    const uint4 lo = kv_load8<KQ>(krow, e, drow), hi8 = kv_load8<KQ>(krow, e + 8, drow);
                    _Float16* bh = (_Float16*)&bk;
                    *(uint4*)bh = lo;
                    *(uint4*)(bh + 8) = hi8;
                }
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bk, s);
            }
        } else {
#pragma unroll
            for (int kk = 0; kk < D / 16; ++kk) {
                const v16h a = *(const v16h*)&sQ[w][i][kk * 16];
                const v16h bk = *(const v16h*)&sK[gw][i][kk * 16];
                s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bk, s);
            }
        }

        // Online softmax; this lane holds position P0 + i of token rows 2 r + hi
        float alpha[8];
        bool rescale = false;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int pz = pos0 + z0 + 2 * r + hi, pp = P0 + i;
            const float sv = pp <= pz && (window <= 0 || pp > pz - window) ? s[r] * scale : -INFINITY;
            float mx = sv;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            const float mn = fmaxf(m[r], mx);
            alpha[r] = mn == -INFINITY || mn == m[r] ? 1.0f : __expf(m[r] - mn);
            rescale |= alpha[r] != 1.0f;
            const float p = sv == -INFINITY ? 0.0f : __expf(sv - mn);
            float ps = p;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) ps += __shfl_xor(ps, off, 32);
            l[r] = l[r] * alpha[r] + ps;
            m[r] = mn;
            sP[w][2 * r + hi][i] = (_Float16)p;
        }
        // The previous acc rescaled by exp(m_old - m_new): skipped (wave-uniform) when no row of the wave has a
        // new maximum, as in most blocks once the running maxima settle; the same values as multiplying by 1.0
        if (__ballot(rescale) != 0) {
#pragma unroll
            for (int dt = 0; dt < DH / 16; ++dt)
#pragma unroll
                for (int r = 0; r < 8; ++r) acc[dt][r] *= alpha[r];
        }
        __syncthreads();  // sP rows are read by other lanes

        // acc += P V
        const v16h a = *(const v16h*)&sP[w][i][0];
#pragma unroll
        for (int dt = 0; dt < DH / 16; ++dt) {
            const v16h bv = *(const v16h*)&sVt[gw][dt * 16 + i][0];
            acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bv, acc[dt]);
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int z = z0 + 2 * r + hi;
        if (z >= T) continue;
        const size_t pc = ((size_t)z * n_q + h) * n_chunks + c;
        if (i == 0 && ds == 0) {
            part_ml[2 * pc] = m[r];
            part_ml[2 * pc + 1] = l[r];
        }
#pragma unroll
        for (int dt = 0; dt < DH / 16; ++dt) part_acc[pc * D + ds * DH + dt * 16 + i] = acc[dt][r];
    }
}

// The kernels for each head size: attn_<kernel>_128 (Qwen3), attn_<kernel>_256 (Qwen3.5 and later, Gemma-4's
// sliding layers), attn_<kernel>_512 (Gemma-4's global layers; its flash kernel is attn_flash_512_h2s2)
#define ATTN_DECODE_KERNELS(D)                                                                                \
    extern "C" __global__ void __launch_bounds__(D)                                                          \
        attn_prep_##D(const float* qfull, const float* k, const float* v, const float* q_norm,                \
                      const float* k_norm, float* q_out, __half* K, __half* V, int kv_stride, int kv_shift,    \
                      long long kv_bstride, int n_q, int n_kv, int n_rot, int pos0, float freq_base, float eps, \
                      int gated, int rot_pairs, int v_norm, int qk_norm, int kv_q8) {                        \
        attn_prep_body<D>(qfull, k, v, q_norm, k_norm, q_out, K, V, kv_stride, kv_shift, kv_bstride, n_q, n_kv, \
                          n_rot, pos0, freq_base, eps, gated, rot_pairs, v_norm, qk_norm, kv_q8);            \
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
        attn_partial_body<D, true>(q, K, V, kv_stride, kv_shift, kv_bstride, pos0, n_q, n_kv_heads, scale,    \
                                   part_ml, part_acc, window);                                               \
    }                                                                                                        \
    extern "C" __global__ void __launch_bounds__(256)                                                        \
        attn_combine_##D(const float* part_ml, const float* part_acc, int n_chunks, int n_q, const float* qfull, \
                         float* out, Q8Out o, int gated) {                                                   \
        attn_combine_body<D>(part_ml, part_acc, n_chunks, n_q, qfull, out, o, gated);                         \
    }
#define ATTN_KERNELS(D)                                                                                       \
    ATTN_DECODE_KERNELS(D)                                                                                    \
    ATTN_FLASH_KERNEL_NAMED(attn_flash_##D, D, 1, 4, 1, 1)

// attn_flash_<D> (ratio n_q / n_kv a multiple of 4: 4 heads x 16 tokens), attn_flash_<D>_g2 (ratio 2: 4 heads of
// two KV heads), attn_flash_<D>_h2t2 (ratio a multiple of 2: 2 heads x 32 tokens), attn_flash_<D>_h2s2 (2 heads
// x 16 tokens, D split over 2 workgroups)
// (each also as NAME_q8 for a Q8_0 KV cache)
#define ATTN_FLASH_KERNEL_KQ(NAME, D, NG, NH, TT, DS, KQ)                                                     \
    extern "C" __global__ void __launch_bounds__(32 * (NH) * (TT))                                           \
        NAME(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift, long long kv_bstride, \
             int pos0, int T, int n_q, int n_kv_heads, int chunk, float scale, float* part_ml, float* part_acc,  \
             int window) {                                                                                   \
        attn_flash_body<D, NG, NH, TT, DS, KQ>(q, K, V, kv_stride, kv_shift, kv_bstride, pos0, T, n_q,       \
                                               n_kv_heads, chunk, scale, part_ml, part_acc, window);         \
    }
#define ATTN_FLASH_KERNEL_NAMED(NAME, D, NG, NH, TT, DS)                                                      \
    ATTN_FLASH_KERNEL_KQ(NAME, D, NG, NH, TT, DS, false)                                                      \
    ATTN_FLASH_KERNEL_KQ(NAME##_q8, D, NG, NH, TT, DS, true)

ATTN_KERNELS(128)
ATTN_KERNELS(256)
ATTN_DECODE_KERNELS(512)
// Two KV heads per workgroup (Qwen3: ratio 2): head_dim 128 only (256 would need over 64 KB of LDS)
ATTN_FLASH_KERNEL_NAMED(attn_flash_128_g2, 128, 2, 4, 1, 1)
// Gemma-4: sliding layers (D 256, ratio 2), global layers (D 512, ratio 8; 59 KB of LDS)
ATTN_FLASH_KERNEL_NAMED(attn_flash_256_h2t2, 256, 1, 2, 2, 1)
ATTN_FLASH_KERNEL_NAMED(attn_flash_512_h2s2, 512, 1, 2, 1, 2)
