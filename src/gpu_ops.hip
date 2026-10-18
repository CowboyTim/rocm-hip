// gpu_ops.hip -- small per-token vector kernels of the Qwen3.5 forward pass (Phase 20 speed work).
//
// add_rmsnorm  x_out = x_in + add (if add != NULL), then y = x_out / sqrt(mean(x_out^2) + eps) * w. Every
//              residual add of the forward pass is followed by an RMSNorm (the next FFN's or mixer's, or
//              output_norm), so one fused kernel covers each layer boundary.
// swiglu       out = SiLU(g) * u, element-wise; one workgroup per 256 values.
// Both are the producers of matvec inputs, so they also quantize their output for the int8 matvecs
// (q8_device.h) when o.qK != NULL, which saves a launch each; n must then be a multiple of 256.
// Same math as qwen_rmsnorm / the CPU SwiGLU in qwen_ops (the reference); the sum of squares is a float
// tree reduction here, a double loop there.

#include <hip/hip_runtime.h>

#include "q8_device.h"

// add_rmsnorm: grid (ceil(n / 256), T), block 256; blockIdx.y is the token (rows of n values, quantized as
// one flattened [T * n] vector). Every workgroup computes the full sum of squares itself (n is
// small, ~10 KB, and the redundant reads hit the cache), then normalizes and quantizes only its own 256-value
// block: the whole GPU works on it instead of one workgroup (that took ~9.5 us, 65 times per token). The
// residual is written to x_out, not back to x_in, so workgroups never read a value another one has already
// updated; the caller ping-pongs the two buffers. Every workgroup sums in the same order, so they all get
// the same scale.
extern "C" __global__ void __launch_bounds__(256)
add_rmsnorm(const float* x_in, const float* add, float* x_out, const float* w, float* y, int n, float eps, Q8Out o) {
    __shared__ float part[8];
    __shared__ Q8Shared sh;
    const long long row = (long long)blockIdx.y * n;
    x_in += row;
    x_out += row;
    y += row;
    if (add) add += row;
    const int t = threadIdx.x, i = blockIdx.x * 256 + t;
    float ss = 0.0f;
    for (int k = t; k < n; k += 256) {
        const float v = add ? x_in[k] + add[k] : x_in[k];
        ss += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((t & 31) == 0) part[t >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) tot += part[k];
    const float inv = rsqrtf(tot / n + eps);

    const bool active = i < n;
    float v = 0.0f;
    if (active) {
        v = add ? x_in[i] + add[i] : x_in[i];
        if (add) x_out[i] = v;
        v = v * inv * (w ? w[i] : 1.0f);
        y[i] = v;
    }
    if (o.qK) q8_block_value(v, active, blockIdx.y * (n / 256) + blockIdx.x, t, sh, o);  // uniform; n % 256 == 0
}

// add_rmsnorm_rows: the same for many rows (prefill): grid T, block 256, one workgroup per row, which it reads
// once (add_rmsnorm's workgroups each re-read the whole row for the sum of squares). n % 256 == 0 and
// n <= 256 * NORM_MAXV. Thread t holds values t + 256 j and sums them in add_rmsnorm's order, so the
// results are bit-identical.
#define NORM_MAXV 16
extern "C" __global__ void __launch_bounds__(256)
add_rmsnorm_rows(const float* x_in, const float* add, float* x_out, const float* w, float* y, int n, float eps,
                 Q8Out o) {
    __shared__ float part[8];
    __shared__ Q8Shared sh;
    const long long row = (long long)blockIdx.x * n;
    x_in += row;
    x_out += row;
    y += row;
    if (add) add += row;
    const int t = threadIdx.x, nb = n / 256;
    float x[NORM_MAXV];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            const int k = t + 256 * j;
            x[j] = add ? x_in[k] + add[k] : x_in[k];
            ss += x[j] * x[j];
        }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((t & 31) == 0) part[t >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) tot += part[k];
    const float inv = rsqrtf(tot / n + eps);

#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {  // uniform
            const int k = t + 256 * j;
            if (add) x_out[k] = x[j];
            const float v = x[j] * inv * (w ? w[k] : 1.0f);
            y[k] = v;
            if (o.qK) q8_block_value(v, true, blockIdx.x * nb + j, t, sh, o);
        }
}

extern "C" __global__ void __launch_bounds__(256)
swiglu(const float* g, const float* u, float* out, int n, Q8Out o) {
    __shared__ Q8Shared sh;
    const int i = blockIdx.x * 256 + threadIdx.x;
    float v = 0.0f;
    if (i < n) {
        const float a = g[i];
        v = a / (1.0f + __expf(-a)) * u[i];
        out[i] = v;
    }
    if (o.qK) q8_block_value(v, i < n, blockIdx.x, threadIdx.x, sh, o);  // uniform; n % 256 == 0 when set
}

// ---------------------------------------------------------------- mixture of experts (qwen35moe)
//
// moe_router   grid T, block 256: per token, softmax over the n_expert router logits, the n_used largest
//              probabilities in descending order (each by its rank in one pass; ties go to the lowest expert index),
//              their weights renormalized by max(sum, 6.103515625e-5), and the shared-expert gate
//              sigmoid(x . w_shexp). As llama.cpp's build_moe_ffn (softmax gating, norm_w, no scale).
// moe_combine  grid (ceil(n / 256), T), block 256: out = sum_j w_j * down_j (in slot order, as ggml's add chain)
//              + sg * shared.

#define MOE_MAX_EXPERTS 1024

__device__ float block256_sum(float v, float* part) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off, 32);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) s += part[k];
    __syncthreads();
    return s;
}

__device__ float block256_max(float v, float* part) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, 32));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    float m = part[0];
#pragma unroll
    for (int k = 1; k < 8; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

extern "C" __global__ void __launch_bounds__(256)
moe_router(const float* logits, const float* x, const float* w_shexp, int* ids, float* wts, float* sg, int n_expert,
           int n_used, int n) {
    __shared__ __attribute__((aligned(16))) float p[MOE_MAX_EXPERTS];
    __shared__ float part[8];
    __shared__ float sel_w[64];
    const int t = blockIdx.x, tid = threadIdx.x;
    logits += (size_t)t * n_expert;
    x += (size_t)t * n;

    // Shared-expert gate first: its loads do not depend on the softmax, so they overlap the logits' loads
    float d = 0.0f;
    if (n % 1024 == 0 && (((uintptr_t)x | (uintptr_t)w_shexp) & 15) == 0) {  // uniform; 16-byte loads, unrolled
#pragma unroll 4
        for (int i = 4 * tid; i < n; i += 1024) {
            const float4 a = *(const float4*)(x + i), b = *(const float4*)(w_shexp + i);
            d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else {
        for (int i = tid; i < n; i += 256) d += x[i] * w_shexp[i];
    }
    d = block256_sum(d, part);
    if (tid == 0) sg[t] = 1.0f / (1.0f + expf(-d));

    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += 256) mx = fmaxf(mx, logits[e]);
    mx = block256_max(mx, part);
    float sum = 0.0f;
    for (int e = tid; e < n_expert; e += 256) {
        p[e] = expf(logits[e] - mx);
        sum += p[e];
    }
    sum = block256_sum(sum, part);
    const float inv = 1.0f / sum;
    for (int e = tid; e < n_expert; e += 256) p[e] *= inv;
    __syncthreads();

    // Top n_used by rank: expert e's rank is the number of experts that beat it (larger p, or equal p and a
    // lower index), so ranks 0 .. n_used - 1 are the selection in descending order, ties to the lower index
    for (int e = tid; e < n_expert; e += 256) {
        const float pe = p[e];
        int rank = 0, j = 0;
#pragma unroll 4
        for (; j + 4 <= n_expert; j += 4) {  // 16-byte LDS reads, several in flight
            const float4 q = *(const float4*)&p[j];
            rank += (q.x > pe || (q.x == pe && j < e)) + (q.y > pe || (q.y == pe && j + 1 < e)) +
                    (q.z > pe || (q.z == pe && j + 2 < e)) + (q.w > pe || (q.w == pe && j + 3 < e));
        }
        for (; j < n_expert; ++j) rank += p[j] > pe || (p[j] == pe && j < e);
        if (rank < n_used) {
            ids[t * n_used + rank] = e;
            sel_w[rank] = pe;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float ws = 0.0f;
        for (int r = 0; r < n_used; ++r) ws += sel_w[r];
        ws = fmaxf(ws, 6.103515625e-5f);
        for (int r = 0; r < n_used; ++r) wts[t * n_used + r] = sel_w[r] / ws;
    }

}

extern "C" __global__ void __launch_bounds__(256)
moe_combine(const float* down, const float* wts, const float* shared, const float* sg, float* out, int n_used, int n) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float* d = down + (size_t)t * n_used * n + i;
    float acc = wts[t * n_used] * d[0];
    for (int j = 1; j < n_used; ++j) acc += wts[t * n_used + j] * d[(size_t)j * n];
    out[(size_t)t * n + i] = acc + shared[(size_t)t * n + i] * sg[t];
}

// moe_group  grid 1, block 256: group a prefill batch's (token, slot) entries by expert, for the expert-tiled
//            matmuls (matmul_q8_id_* in matvec_q.hip). Entries sorted by expert (stable): tok_rows[i] = token,
//            pair_rows[i] = token * n_used + slot. Each expert's run is cut into tiles of up to `tile` entries:
//            tile_e / tile_start / tile_n, *n_tiles of them. Serial in one thread: <= T * n_used + n_expert steps,
//            deterministic.
extern "C" __global__ void __launch_bounds__(256)
moe_group(const int* ids, int T, int n_used, int n_expert, int tile, int* tok_rows, int* pair_rows, int* n_tiles,
          int* tile_e, int* tile_start, int* tile_n) {
    __shared__ int cnt[MOE_MAX_EXPERTS], off[MOE_MAX_EXPERTS];
    for (int e = threadIdx.x; e < n_expert; e += 256) cnt[e] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    const int n = T * n_used;
    for (int i = 0; i < n; ++i) cnt[ids[i]]++;
    int acc = 0;
    for (int e = 0; e < n_expert; ++e) {
        off[e] = acc;
        acc += cnt[e];
    }
    int nt = 0;
    for (int e = 0; e < n_expert; ++e)
        for (int st = off[e]; st < off[e] + cnt[e]; st += tile) {
            tile_e[nt] = e;
            tile_start[nt] = st;
            tile_n[nt] = min(tile, off[e] + cnt[e] - st);
            ++nt;
        }
    *n_tiles = nt;
    for (int i = 0; i < n; ++i) {
        const int p = off[ids[i]]++;
        tok_rows[p] = i / n_used;
        pair_rows[p] = i;
    }
}
