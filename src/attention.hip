// attention.hip -- gated full-attention token mixer kernels, one token (Phase 20 speed work).
//
// Queued between the q/k/v projections and attn_output, so an attention layer needs one CPU sync. Same math
// as the CPU reference in qwen_ops.cpp (qwen_rmsnorm, qwen_rope_neox, qwen_attention), which follows
// llama.cpp's qwen35 graph. HD = head_dim = 256 (the host checks it).
//
// All three take T tokens at positions pos0 .. pos0 + T - 1 (prefill; T = 1 when generating) through the
// last grid dimension; token z attends to positions 0 .. pos0 + z (causal). Per-token rows: qfull
// [2 * n_q * HD], k / v [n_kv * HD], q / out [n_q * HD].
//
// attn_prep     grid (n_q + n_kv, T), block HD. Workgroup h < n_q: Q head h (first half of attn_q's [Q | gate]
//               head), RMSNorm with q_norm, NeoX RoPE over the first n_rot dims -> q. Workgroup n_q + g:
//               K head g, RMSNorm with k_norm, RoPE -> FP16 into the K cache row of this position, and V
//               head g -> FP16 into the V cache row. The CPU never touches the KV cache.
// attn_partial  grid (n_q, n_chunks, T), block HD (= CH positions per chunk), n_chunks for the last token;
//               chunks past a token's last position contribute nothing (m = -inf, l = 0, acc = 0). Flash-decoding: thread t scores
//               position chunk * CH + t against q (q . K / sqrt(HD)), the workgroup takes the chunk's max m
//               and sum l of exp(score - m), then thread d accumulates sum_t exp(score_t - m) * V[t][d]
//               (coalesced over d). Writes m, l and acc[HD] per (head, chunk).
// attn_combine  grid (n_q, T), block HD. Merges the chunks (rescaling each by exp(m_c - M)), divides by the total
//               weight and applies the output gate sigmoid(gate) -> out. out feeds attn_output, so it is also
//               quantized for the int8 matvec when o.qK != NULL (one head = one 256-value block).
// Query head h reads KV head h / (n_q / n_kv), as ggml's broadcast does.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q8_device.h"

#define HD 256
#define CH HD

// KV cache row of position p, in __half units from the layer's row 0: blocks of 2^kv_shift positions, kv_bstride
// apart, rows kv_stride apart inside a block (kv_cache.h). A contiguous cache is kv_shift = 30.
__device__ __forceinline__ size_t kv_row(int p, int kv_stride, int kv_shift, long long kv_bstride) {
    return (size_t)(p >> kv_shift) * (size_t)kv_bstride + (size_t)(p & ((1 << kv_shift) - 1)) * (size_t)kv_stride;
}

__device__ __forceinline__ float wave_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 32));
    return v;
}

// Sum / max over the HD threads of a workgroup (wave32)
__device__ float block_sum(float v) {
    __shared__ float part[HD / 32];
    v = wave_sum(v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < HD / 32; ++i) s += part[i];
    __syncthreads();
    return s;
}

__device__ float block_max(float v) {
    __shared__ float part[HD / 32];
    v = wave_max(v);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = -INFINITY;
#pragma unroll
    for (int i = 0; i < HD / 32; ++i) m = fmaxf(m, part[i]);
    __syncthreads();
    return m;
}

// RMSNorm (gamma w) then NeoX RoPE of one head held one value per thread; returns this thread's value
__device__ float norm_rope(float x, const float* w, int n_rot, int pos, float freq_base, float eps) {
    __shared__ float v[HD];
    const int d = threadIdx.x;
    const float ss = block_sum(x * x);
    x = x * rsqrtf(ss / HD + eps) * w[d];
    v[d] = x;
    __syncthreads();
    const int half = n_rot / 2;
    if (d < n_rot) {
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

extern "C" __global__ void __launch_bounds__(HD)
attn_prep(const float* qfull, const float* k, const float* v, const float* q_norm, const float* k_norm, float* q_out,
          __half* K, __half* V, int kv_stride, int kv_shift, long long kv_bstride, int n_q, int n_kv, int n_rot, int pos0,
          float freq_base, float eps) {
    const int b = blockIdx.x, d = threadIdx.x, z = blockIdx.y, pos = pos0 + z;
    qfull += (size_t)z * 2 * n_q * HD;
    q_out += (size_t)z * n_q * HD;
    k += (size_t)z * n_kv * HD;
    v += (size_t)z * n_kv * HD;
    __half* k_row = K + kv_row(pos, kv_stride, kv_shift, kv_bstride);
    __half* v_row = V + kv_row(pos, kv_stride, kv_shift, kv_bstride);
    if (b < n_q) {
        const float x = qfull[(size_t)b * 2 * HD + d];
        q_out[b * HD + d] = norm_rope(x, q_norm, n_rot, pos, freq_base, eps);
    } else {
        const int g = b - n_q;
        const float x = norm_rope(k[g * HD + d], k_norm, n_rot, pos, freq_base, eps);
        k_row[g * HD + d] = __float2half(x);
        v_row[g * HD + d] = __float2half(v[g * HD + d]);
    }
}

extern "C" __global__ void __launch_bounds__(HD)
attn_partial(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift, long long kv_bstride, int pos0,
             int n_q, int n_kv_heads,
             float scale, float* part_ml, float* part_acc) {
    __shared__ float sq[HD], sw[CH];
    const int h = blockIdx.x, c = blockIdx.y, t = threadIdx.x, n_chunks = gridDim.y, z = blockIdx.z;
    const int n_kv = pos0 + z + 1;
    q += (size_t)z * n_q * HD;
    part_ml += (size_t)2 * z * n_q * n_chunks;
    part_acc += (size_t)z * n_q * n_chunks * HD;
    const int g = h / (n_q / n_kv_heads);
    sq[t] = q[h * HD + t];
    __syncthreads();

    // Score of position p = c * CH + t (thread per position); K row read as 8 halves at a time
    const int p = c * CH + t;
    float score = -INFINITY;
    if (p < n_kv) {
        const uint4* kr = (const uint4*)(K + kv_row(p, kv_stride, kv_shift, kv_bstride) + (size_t)g * HD);
        float s = 0.0f;
#pragma unroll 4
        for (int i = 0; i < HD / 8; ++i) {
            const uint4 u = kr[i];
            const __half2* h2 = (const __half2*)&u;
#pragma unroll
            for (int k2 = 0; k2 < 4; ++k2) {
                const float2 f = __half22float2(h2[k2]);
                s += f.x * sq[8 * i + 2 * k2] + f.y * sq[8 * i + 2 * k2 + 1];
            }
        }
        score = s * scale;
    }
    const float m = block_max(score);
    const float e = p < n_kv ? __expf(score - m) : 0.0f;
    const float l = block_sum(e);
    sw[t] = e;
    __syncthreads();

    // acc[d] over this chunk's positions (thread per dim)
    const int p0 = c * CH, np = max(0, min(CH, n_kv - p0));
    const __half* vb = V + (size_t)g * HD + t;
    float acc = 0.0f;
    for (int i = 0; i < np; ++i) acc += sw[i] * __half2float(vb[kv_row(p0 + i, kv_stride, kv_shift, kv_bstride)]);

    const size_t pc = (size_t)h * n_chunks + c;
    part_acc[pc * HD + t] = acc;
    if (t == 0) {
        part_ml[2 * pc] = m;
        part_ml[2 * pc + 1] = l;
    }
}

extern "C" __global__ void __launch_bounds__(HD)
attn_combine(const float* part_ml, const float* part_acc, int n_chunks, int n_q, const float* qfull, float* out,
             Q8Out o) {
    __shared__ Q8Shared sh;
    const int h = blockIdx.x, d = threadIdx.x, z = blockIdx.y;
    part_ml += (size_t)2 * z * n_q * n_chunks;
    part_acc += (size_t)z * n_q * n_chunks * HD;
    qfull += (size_t)z * 2 * n_q * HD;
    out += (size_t)z * n_q * HD;
    const float* ml = part_ml + (size_t)2 * h * n_chunks;
    float M = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, ml[2 * c]);
    float num = 0.0f, den = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const float w = __expf(ml[2 * c] - M);
        num += w * part_acc[((size_t)h * n_chunks + c) * HD + d];
        den += w * ml[2 * c + 1];
    }
    const float gate = qfull[(size_t)h * 2 * HD + HD + d];
    const float r = num / den / (1.0f + __expf(-gate));
    out[h * HD + d] = r;
    if (o.qK) q8_block_value(r, true, z * n_q + h, d, sh, o);  // uniform; HD == 256: one head = one block
}

// ---------------------------------------------------------------- prefill: query-tiled flash attention
//
// attn_flash  grid (n_q / 4, ceil(T / FT), n_chunks), block 128 (4 waves). The same output as attn_partial
//             (m, l, acc per (token, head, chunk), merged by attn_combine), for FT = 16 tokens and 4 query
//             heads of one KV head at once, so each K/V chunk is read once per 64 query rows instead of once per
//             row. A chunk is `chunk` positions (a multiple of FB, chosen by the host so that a token has at
//             most ATTN_MAX_SPLITS chunks). Wave w takes head 4 * blockIdx.x + w; n_q / n_kv must be a multiple of 4 (the host checks).
//             Per block of FB = 16 positions: the K block and the transposed V block are staged in LDS (FP16,
//             as cached), S = Q K^T (16 tokens x 16 positions, FP16 WMMA, FP32 accumulation; Q rounded to FP16
//             as ggml's flash attention does), causal mask, online softmax per token row (running max m and sum
//             l; the row's previous acc is rescaled by exp(m_old - m_new)), then acc += P V with P in FP16.
//             Operand layouts as tests/wmma_layout_verify: A row i / B column i in lanes i and i + 16; C
//             column lane % 16, element r = row 2 r + lane / 16.

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

#define FT 16          // query tokens per tile
#define FB 16          // positions per block
#define QP (HD + 8)    // padded LDS row of Q / K (halves)

extern "C" __global__ void __launch_bounds__(128)
attn_flash(const float* q, const __half* K, const __half* V, int kv_stride, int kv_shift, long long kv_bstride, int pos0,
           int T, int n_q, int n_kv_heads,
           int chunk, float scale, float* part_ml, float* part_acc) {
    __shared__ __attribute__((aligned(16))) _Float16 sQ[4][FT][QP];
    __shared__ __attribute__((aligned(16))) _Float16 sK[FB][QP];
    __shared__ __attribute__((aligned(16))) _Float16 sVt[HD][FB];  // V block transposed: [dim][position]
    __shared__ __attribute__((aligned(16))) _Float16 sP[4][FT][FB];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, i = lane & 15, hi = lane >> 4;
    const int h = blockIdx.x * 4 + w, g = blockIdx.x * 4 / (n_q / n_kv_heads);
    const int z0 = blockIdx.y * FT, c = blockIdx.z, n_chunks = gridDim.z;
    const int zmax = min(T, z0 + FT) - 1;                   // last token of the tile
    const int npos = min(chunk, pos0 + zmax + 1 - c * chunk);  // chunk positions any token of the tile sees
    const int nb = npos > 0 ? (npos + FB - 1) / FB : 0;

    // This wave's Q rows (tokens z0 .. z0 + 15 of head h) to FP16; rows past T are zero
    for (int idx = lane; idx < FT * HD / 4; idx += 32) {
        const int rr = idx / (HD / 4), d = 4 * (idx % (HD / 4));
        float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (z0 + rr < T) x = *(const float4*)(q + ((size_t)(z0 + rr) * n_q + h) * HD + d);
        sQ[w][rr][d] = (_Float16)x.x, sQ[w][rr][d + 1] = (_Float16)x.y;
        sQ[w][rr][d + 2] = (_Float16)x.z, sQ[w][rr][d + 3] = (_Float16)x.w;
    }

    float m[8], l[8];
    v8f acc[HD / 16];
#pragma unroll
    for (int r = 0; r < 8; ++r) m[r] = -INFINITY, l[r] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < HD / 16; ++dt) acc[dt] = (v8f){};

#pragma unroll 1
    for (int b = 0; b < nb; ++b) {
        const int P0 = c * chunk + b * FB;
        __syncthreads();  // the previous block's sK / sVt reads are done (and, the first time, sQ is written)
        // Stage the K and V blocks: 16 positions x 256 dims, 8 halves per uint4. Positions no token of the
        // tile sees are zeroed: they are masked, but 0 * (stale V) must stay 0.
#pragma unroll
        for (int k = 0; k < FB * HD / 8 / 128; ++k) {
            const int idx = tid + 128 * k, p = idx / (HD / 8), d = 8 * (idx % (HD / 8));
            uint4 ku = make_uint4(0, 0, 0, 0), vu = ku;
            if (P0 + p <= pos0 + zmax) {
                const size_t off = kv_row(P0 + p, kv_stride, kv_shift, kv_bstride) + (size_t)g * HD + d;
                ku = *(const uint4*)(K + off);
                vu = *(const uint4*)(V + off);
            }
            *(uint4*)&sK[p][d] = ku;
            const _Float16* vh = (const _Float16*)&vu;
#pragma unroll
            for (int e = 0; e < 8; ++e) sVt[d + e][p] = vh[e];
        }
        __syncthreads();

        // S = Q K^T for this wave's 16 tokens x 16 positions
        v8f s = {};
#pragma unroll
        for (int kk = 0; kk < HD / 16; ++kk) {
            const v16h a = *(const v16h*)&sQ[w][i][kk * 16];
            const v16h bk = *(const v16h*)&sK[i][kk * 16];
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bk, s);
        }

        // Online softmax; this lane holds position P0 + i of token rows 2 r + hi
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int z = z0 + 2 * r + hi;
            const float sv = P0 + i <= pos0 + z ? s[r] * scale : -INFINITY;
            float mx = sv;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            const float mn = fmaxf(m[r], mx);
            const float alpha = mn == -INFINITY ? 1.0f : __expf(m[r] - mn);
            const float p = sv == -INFINITY ? 0.0f : __expf(sv - mn);
            float ps = p;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) ps += __shfl_xor(ps, off, 32);
            l[r] = l[r] * alpha + ps;
            m[r] = mn;
#pragma unroll
            for (int dt = 0; dt < HD / 16; ++dt) acc[dt][r] *= alpha;
            sP[w][2 * r + hi][i] = (_Float16)p;
        }
        __syncthreads();  // sP rows are read by other lanes

        // acc += P V
        const v16h a = *(const v16h*)&sP[w][i][0];
#pragma unroll
        for (int dt = 0; dt < HD / 16; ++dt) {
            const v16h bv = *(const v16h*)&sVt[dt * 16 + i][0];
            acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bv, acc[dt]);
        }
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int z = z0 + 2 * r + hi;
        if (z >= T) continue;
        const size_t pc = ((size_t)z * n_q + h) * n_chunks + c;
        if (i == 0) {
            part_ml[2 * pc] = m[r];
            part_ml[2 * pc + 1] = l[r];
        }
#pragma unroll
        for (int dt = 0; dt < HD / 16; ++dt) part_acc[pc * HD + dt * 16 + i] = acc[dt][r];
    }
}
