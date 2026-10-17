// attention.hip -- gated full-attention token mixer kernels, one token (Phase 20 speed work).
//
// Queued between the q/k/v projections and attn_output, so an attention layer needs one CPU sync. Same math
// as the CPU reference in qwen_ops.cpp (qwen_rmsnorm, qwen_rope_neox, qwen_attention), which follows
// llama.cpp's qwen35 graph. HD = head_dim = 256 (the host checks it).
//
// attn_prep     grid n_q + n_kv, block HD. Workgroup h < n_q: Q head h (first half of attn_q's [Q | gate]
//               head), RMSNorm with q_norm, NeoX RoPE over the first n_rot dims -> q. Workgroup n_q + g:
//               K head g, RMSNorm with k_norm, RoPE -> FP16 into the K cache row of this position, and V
//               head g -> FP16 into the V cache row. The CPU never touches the KV cache.
// attn_partial  grid (n_q, n_chunks), block HD (= CH positions per chunk). Flash-decoding: thread t scores
//               position chunk * CH + t against q (q . K / sqrt(HD)), the workgroup takes the chunk's max m
//               and sum l of exp(score - m), then thread d accumulates sum_t exp(score_t - m) * V[t][d]
//               (coalesced over d). Writes m, l and acc[HD] per (head, chunk).
// attn_combine  grid n_q, block HD. Merges the chunks (rescaling each by exp(m_c - M)), divides by the total
//               weight and applies the output gate sigmoid(gate) -> out. out feeds attn_output, so it is also
//               quantized for the int8 matvec when o.qK != NULL (one head = one 256-value block).
// Query head h reads KV head h / (n_q / n_kv), as ggml's broadcast does.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "q8_device.h"

#define HD 256
#define CH HD

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
          __half* k_row, __half* v_row, int n_q, int n_rot, int pos, float freq_base, float eps) {
    const int b = blockIdx.x, d = threadIdx.x;
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
attn_partial(const float* q, const __half* K, const __half* V, int kv_stride, int n_kv, int n_q, int n_kv_heads,
             float scale, float* part_ml, float* part_acc) {
    __shared__ float sq[HD], sw[CH];
    const int h = blockIdx.x, c = blockIdx.y, t = threadIdx.x, n_chunks = gridDim.y;
    const int g = h / (n_q / n_kv_heads);
    sq[t] = q[h * HD + t];
    __syncthreads();

    // Score of position p = c * CH + t (thread per position); K row read as 8 halves at a time
    const int p = c * CH + t;
    float score = -INFINITY;
    if (p < n_kv) {
        const uint4* kr = (const uint4*)(K + (size_t)p * kv_stride + (size_t)g * HD);
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
    const int p0 = c * CH, np = min(CH, n_kv - p0);
    const __half* vb = V + (size_t)p0 * kv_stride + (size_t)g * HD + t;
    float acc = 0.0f;
    for (int i = 0; i < np; ++i) acc += sw[i] * __half2float(vb[(size_t)i * kv_stride]);

    const size_t pc = (size_t)h * n_chunks + c;
    part_acc[pc * HD + t] = acc;
    if (t == 0) {
        part_ml[2 * pc] = m;
        part_ml[2 * pc + 1] = l;
    }
}

extern "C" __global__ void __launch_bounds__(HD)
attn_combine(const float* part_ml, const float* part_acc, int n_chunks, const float* qfull, float* out, Q8Out o) {
    __shared__ Q8Shared sh;
    const int h = blockIdx.x, d = threadIdx.x;
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
    if (o.qK) q8_block_value(r, true, h, d, sh, o);  // uniform; HD == 256
}
