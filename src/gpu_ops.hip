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

#include <hip/hip_fp16.h>
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
    if (o.qK) q8_block_value(v, i < n, blockIdx.x, threadIdx.x, sh, o);  // uniform; n % 32 == 0 when set
}

// ---------------------------------------------------------------- mixture of experts (qwen35moe)
//
// moe_router   grid T, block 256: per token, softmax over the n_expert router logits, the n_used largest
//              probabilities in descending order (wave argmax rounds; ties go to the lowest expert index),
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

// moe_router's selection on one wave: lane l holds experts l + 32 j, j < PL, of p[n_expert] in registers; each
// of the n_used rounds takes the largest remaining (the lowest index among equals) into ids / sel_w
template <int PL>
__device__ __forceinline__ void router_select(const float* p, int n_expert, int n_used, int* ids, float* sel_w,
                                              int lane) {
    float v[PL];
#pragma unroll
    for (int j = 0; j < PL; ++j) {
        const int e = lane + 32 * j;
        v[j] = e < n_expert ? p[e] : -1.0f;
    }
    for (int r = 0; r < n_used; ++r) {
        float bv = v[0];
        int bj = 0;
#pragma unroll
        for (int j = 1; j < PL; ++j)
            if (v[j] > bv) bv = v[j], bj = j;  // strict: the lowest index among equals
        int be = lane + 32 * bj;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 32);
            const int oe = __shfl_xor(be, off, 32);
            if (ov > bv || (ov == bv && oe < be)) bv = ov, be = oe;
        }
        if (lane == 0) {
            ids[r] = be;
            sel_w[r] = bv;
        }
        if ((be & 31) == lane) {
#pragma unroll
            for (int j = 0; j < PL; ++j)
                if (j == be / 32) v[j] = -INFINITY;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256)
moe_router(const float* logits, const float* x, const float* w_shexp, int* ids, float* wts, float* sg, int n_expert,
           int n_used, int n, int ids_stride) {
    __shared__ __attribute__((aligned(16))) float p[MOE_MAX_EXPERTS];
    __shared__ float part[8];
    __shared__ float sel_w[64];
    const int t = blockIdx.x, tid = threadIdx.x;
    logits += (size_t)t * n_expert;
    x += (size_t)t * n;

    // Shared-expert gate first (none without w_shexp: Gemma-4): its loads do not depend on the softmax, so they
    // overlap the logits' loads
    if (w_shexp) {  // uniform
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
    }

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

    // Top n_used: n_used rounds of a wave argmax over the remaining experts, ties to the lower index (the order of
    // the ranks the reference's argsort gives). Lane l holds experts l + 32 j in registers (static indices); the
    // winner's lane drops it. A bitonic sort of all 512 (45 steps, each with a workgroup barrier) took ~10 us.
    if (tid < 32) {
        if (n_expert <= 512) router_select<16>(p, n_expert, n_used, ids + t * ids_stride, sel_w, tid);
        else router_select<MOE_MAX_EXPERTS / 32>(p, n_expert, n_used, ids + t * ids_stride, sel_w, tid);
    }
    if (tid == n_used && ids_stride > n_used) ids[t * ids_stride + n_used] = -1;  // the shared expert's slot
    __syncthreads();
    if (tid == 0) {
        float ws = 0.0f;
        for (int r = 0; r < n_used; ++r) ws += sel_w[r];
        ws = fmaxf(ws, 6.103515625e-5f);
        for (int r = 0; r < n_used; ++r) wts[t * n_used + r] = sel_w[r] / ws;
    }

}

// moe_combine with per-expert output scales exp_scale[n_expert] (Gemma-4's ffn_down_exps.scale, applied to the
// expert's output before its weight, as llama.cpp's build_moe_ffn) when set, and no shared expert when shared is NULL
extern "C" __global__ void __launch_bounds__(256)
moe_combine(const float* down, const float* wts, const float* shared, const float* sg, float* out, int n_used, int n,
            const int* ids, const float* exp_scale) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float* d = down + (size_t)t * n_used * n + i;
    const float* w = wts + t * n_used;
    const int* e = ids + t * n_used;
    float acc = exp_scale ? (d[0] * exp_scale[e[0]]) * w[0] : w[0] * d[0];
    for (int j = 1; j < n_used; ++j)
        acc += exp_scale ? (d[(size_t)j * n] * exp_scale[e[j]]) * w[j] : w[j] * d[(size_t)j * n];
    out[(size_t)t * n + i] = shared ? acc + shared[(size_t)t * n + i] * sg[t] : acc;
}

// moe_group  grid 1, block 1024: group a prefill batch's (token, slot) entries by expert, for the expert-tiled
//            matmuls (matmul_q8_id_* / matmul_wmma_id_* in matvec_q.hip). Entries sorted by expert:
//            tok_rows[i] = token, pair_rows[i] = token * n_used + slot. Each expert's run is cut into tiles of up
//            to `tile` entries: tile_e / tile_start / tile_n, *n_tiles of them, in expert order. Counts and
//            tiles are deterministic; the order of the entries within an expert's run is not (atomic slots).
//            Nothing depends on it: the kernels compute each entry's value with the same sum wherever it sits
//            in a tile.
extern "C" __global__ void __launch_bounds__(1024)
moe_group(const int* ids, int T, int n_used, int n_expert, int tile, int* tok_rows, int* pair_rows, int* n_tiles,
          int* tile_e, int* tile_start, int* tile_n) {
    __shared__ int cnt[MOE_MAX_EXPERTS], slot[MOE_MAX_EXPERTS], wsum[2][32];
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int n = T * n_used;
    cnt[tid] = 0;  // n_expert <= MOE_MAX_EXPERTS == blockDim.x
    __syncthreads();
    for (int i = tid; i < n; i += 1024) atomicAdd(&cnt[ids[i]], 1);
    __syncthreads();
    // exclusive scans over the experts (thread e): entry offsets and tile offsets
    const int c = tid < n_expert ? cnt[tid] : 0, nt = (c + tile - 1) / tile;
    int a = c, b = nt;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const int ua = __shfl_up(a, d), ub = __shfl_up(b, d);
        if (lane >= d) a += ua, b += ub;
    }
    if (lane == 31) wsum[0][w] = a, wsum[1][w] = b;
    __syncthreads();
    if (w == 0) {
        int x = wsum[0][lane], y = wsum[1][lane];
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int ux = __shfl_up(x, d), uy = __shfl_up(y, d);
            if (lane >= d) x += ux, y += uy;
        }
        wsum[0][lane] = x - wsum[0][lane];  // exclusive per-wave bases
        wsum[1][lane] = y - wsum[1][lane];
    }
    __syncthreads();
    const int eo = wsum[0][w] + a - c, to = wsum[1][w] + b - nt;
    slot[tid] = eo;  // the next free entry of expert tid
    if (tid < n_expert) {
        for (int k = 0; k < nt; ++k) {
            tile_e[to + k] = tid;
            tile_start[to + k] = eo + k * tile;
            tile_n[to + k] = min(tile, c - k * tile);
        }
        if (tid == n_expert - 1) *n_tiles = to + nt;
    }
    __syncthreads();
    for (int i = tid; i < n; i += 1024) {
        const int p = atomicAdd(&slot[ids[i]], 1);
        tok_rows[p] = i / n_used;
        pair_rows[p] = i;
    }
}

// ---------------------------------------------------------------- hyper-connections + PLE (qwen4exp)
//
// The residual is hc streams of n values per token, rows r = t * hc + c of res. As llama.cpp's qwen4exp graph
// (build_hc_mix / build_hc_combine / build_ple), each op in the reference's order.
//
// hc_norm     grid (S, T * hc), block 256: with cur, first the combine res_out[r] = res_in[r] + cur[t] * 2 *
//             sigmoid(inject[t][c] / hc); then y[r] = RMSNorm(res[r]) * w[c * n ..] (per stream, gamma over all
//             hc * n values), quantized as one flattened vector. n % 256 == 0, n <= 256 * NORM_MAXV. S = 1: a
//             workgroup per row (prefill; res_out may be res_in). S = n / 256: the row is spread over S
//             workgroups (one token: 4 rows would leave the GPU idle), each summing the whole row in the same
//             order and writing its own 256 values; res_out must not be res_in (the caller ping-pongs).
// hc_lowrank  out[t] = SiLU(lo[t] / hc) for rows of n values, zero-padded to rows of np (the up projection's
//             padded K), quantized as one flattened vector (np % 32 == 0 when set)
// hc_mix      grid (ceil(n / 256), T): out[t] = (sum over c of xn[r] * sigmoid(gate[r])) / hc, streams added in
//             order; quantized (n % 256 == 0 when set)

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

extern "C" __global__ void __launch_bounds__(256)
hc_norm(const float* res_in, float* res_out, const float* cur, const float* inject, const float* w, float* y, int n,
        int hc, float eps, Q8Out o) {
    __shared__ float part[8];
    __shared__ Q8Shared sh;
    const int r = blockIdx.y, t = r / hc, c = r % hc, tid = threadIdx.x, nb = n / 256;
    const bool split = gridDim.x > 1;
    const float* x = res_in + (size_t)r * n;
    float* xo = res_out + (size_t)r * n;
    const float inv_hc = 1.0f / (float)hc;
    // every load first (the residual, the block output, the stream weight), then the arithmetic: a token's
    // rows are spread over many small workgroups, whose time is the latency of these loads
    float v[NORM_MAXV], u[NORM_MAXV];
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            const int k = tid + 256 * j;
            v[j] = x[k];
            u[j] = cur ? cur[(size_t)t * n + k] : 0.0f;
        }
    const float wt = cur ? 2.0f * sigmoidf_(inject[t * hc + c] * inv_hc) : 0.0f;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            if (cur) v[j] = __fadd_rn(v[j], __fmul_rn(u[j], wt));
            ss += v[j] * v[j];
        }
    const float inv = rsqrtf(block256_sum(ss, part) / n + eps);
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb && (!split || j == (int)blockIdx.x)) {  // uniform
            const int k = tid + 256 * j;
            if (cur) xo[k] = v[j];
            const float out = v[j] * inv * w[c * n + k];
            y[(size_t)r * n + k] = out;
            if (o.qK) q8_block_value(out, true, r * nb + j, tid, sh, o);
        }
}

extern "C" __global__ void __launch_bounds__(256)
hc_lowrank(const float* lo, float* out, int n, int np, int T, float scale, Q8Out o) {
    __shared__ Q8Shared sh;
    const int i = blockIdx.x * 256 + threadIdx.x, t = i / np, k = i - t * np;
    const bool active = i < np * T;
    float v = 0.0f;
    if (active) {
        if (k < n) {
            const float a = lo[(size_t)t * n + k] * scale;
            v = a / (1.0f + __expf(-a));
        }
        out[i] = v;
    }
    if (o.qK) q8_block_value(v, active, blockIdx.x, threadIdx.x, sh, o);  // uniform
}

extern "C" __global__ void __launch_bounds__(256)
hc_mix(const float* xn, const float* gate, float* out, int n, int hc, Q8Out o) {
    __shared__ Q8Shared sh;
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    float v = 0.0f;
    if (i < n) {
        const size_t base = (size_t)t * hc * n + i;
        v = __fmul_rn(xn[base], sigmoidf_(gate[base]));
        for (int c = 1; c < hc; ++c) v = __fadd_rn(v, __fmul_rn(xn[base + (size_t)c * n], sigmoidf_(gate[base + (size_t)c * n])));
        v *= 1.0f / (float)hc;
        out[(size_t)t * n + i] = v;
    }
    if (o.qK) q8_block_value(v, i < n, t * (n / 256) + blockIdx.x, threadIdx.x, sh, o);  // uniform
}

// mtp_concat  grid (ceil(2n / 256), T * hc): out[t][c] = [e[t] | h[t][c]] (n values each), the MTP head's
//             eh_proj input per stream
extern "C" __global__ void __launch_bounds__(256)
mtp_concat(const float* e, const float* h, float* out, int n, int hc) {
    const int r = blockIdx.y, t = r / hc, i = blockIdx.x * 256 + threadIdx.x;
    if (i < 2 * n) out[(size_t)r * 2 * n + i] = i < n ? e[(size_t)t * n + i] : h[(size_t)r * n + i - n];
}

// ple_gate  grid T, block 256: per stream c, s = (RMSNorm(key[r]) * wk) . (RMSNorm(res[r]) * wq) / sqrt(n),
//           gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))); gated[r] = value[t] * gate,
//           normed[r] = RMSNorm(gated[r]) * wc. key, res, gated, normed: [T][hc][n]; value [T][n].
extern "C" __global__ void __launch_bounds__(256)
ple_gate(const float* key, const float* value, const float* res, const float* wk, const float* wq, const float* wc,
         float* gated, float* normed, int n, int hc, float eps) {
    __shared__ float part[8];
    const int t = blockIdx.x, tid = threadIdx.x, nb = n / 256;
    float val[NORM_MAXV];
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) val[j] = value[(size_t)t * n + tid + 256 * j];
    for (int c = 0; c < hc; ++c) {
        const size_t r = ((size_t)t * hc + c) * n;
        float kx[NORM_MAXV], qx[NORM_MAXV];
        float sk = 0.0f, sq = 0.0f;
#pragma unroll
        for (int j = 0; j < NORM_MAXV; ++j)
            if (j < nb) {
                kx[j] = key[r + tid + 256 * j];
                qx[j] = res[r + tid + 256 * j];
                sk += kx[j] * kx[j];
                sq += qx[j] * qx[j];
            }
        const float ik = rsqrtf(block256_sum(sk, part) / n + eps), iq = rsqrtf(block256_sum(sq, part) / n + eps);
        float d = 0.0f;
#pragma unroll
        for (int j = 0; j < NORM_MAXV; ++j)
            if (j < nb) {
                const int k = c * n + tid + 256 * j;
                d += (kx[j] * ik * wk[k]) * (qx[j] * iq * wq[k]);
            }
        const float s = block256_sum(d, part) * (1.0f / sqrtf((float)n));
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float g = sigmoidf_((s > 0.0f ? 1.0f : s < 0.0f ? -1.0f : 0.0f) * mag);
        float sg = 0.0f;
#pragma unroll
        for (int j = 0; j < NORM_MAXV; ++j)
            if (j < nb) {
                kx[j] = val[j] * g;
                sg += kx[j] * kx[j];
            }
        const float ig = rsqrtf(block256_sum(sg, part) / n + eps);
#pragma unroll
        for (int j = 0; j < NORM_MAXV; ++j)
            if (j < nb) {
                const int k = tid + 256 * j;
                gated[r + k] = kx[j];
                normed[r + k] = kx[j] * ig * wc[c * n + k];
            }
    }
}

// ple_conv  grid ceil(m / 256), block 256, one thread per channel ch of m = hc * n: the depthwise causal conv of
//           normed over the tokens, K taps dilated by dil (tap k reads (K - 1 - k) * dil positions back; earlier
//           positions come from state [(K - 1) * dil][m], oldest first), then res += gated + SiLU(conv). The last
//           (K - 1) * dil inputs become the new state. w: F16 [m][K].
struct MidStates {  // deltanet_mid_states (deltanet.h): p[j], when set, receives the state after token j
    float* p[3];
};

extern "C" __global__ void __launch_bounds__(256)
ple_conv(const float* normed, const float* gated, float* res, float* state, const __half* w, int m, int T, int K,
         int dil, MidStates mid) {
    const int ch = blockIdx.x * 256 + threadIdx.x;
    if (ch >= m) return;
    const int H = (K - 1) * dil;
    for (int t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int j = t - (K - 1 - k) * dil;
            const float x = j >= 0 ? normed[(size_t)j * m + ch] : state[(size_t)(H + j) * m + ch];
            const float term = x * __half2float(w[(size_t)ch * K + k]);
            acc = k == 0 ? term : acc + term;
        }
        const size_t i = (size_t)t * m + ch;
        res[i] += gated[i] + acc / (1.0f + __expf(-acc));
    }
    // mid: the states after the first tokens, the one after token j = combined[j + 1 + i]
    for (int j = 0; j < 3 && j < T - 1; ++j)
        if (mid.p[j])
            for (int i = 0; i < H; ++i) {
                const int src = j + 1 + i;
                mid.p[j][(size_t)i * m + ch] =
                    src < H ? state[(size_t)src * m + ch] : normed[(size_t)(src - H) * m + ch];
            }
    // new state row i = combined[T + i], combined = state ++ normed; ascending i only reads rows > i of state
    for (int i = 0; i < H; ++i) {
        const int src = T + i;
        state[(size_t)i * m + ch] = src < H ? state[(size_t)src * m + ch] : normed[(size_t)(src - H) * m + ch];
    }
}

// ---------------------------------------------------------------- Gemma-4
//
// add_normed  grid T, block 256: out = (base + RMSNorm(a) * w) * (scale ? *scale : 1) per row of n values
//             (n % 256 == 0, n <= 256 * NORM_MAXV): Gemma-4's norms after the attention and the FFN, added to the
//             residual, and the layer's output scale. out may be base (each thread reads its values first).
// geglu       out = GELU_tanh(g) * u (ggml_gelu's tanh form) on rows of h values: g / u row r at gate + r * gs /
//             up + r * us, so the experts' fused gate|up rows ([gate h | up h]) need no copy; quantized into q8
//             as swiglu when set.
// softcap     x = cap * tanh(x / cap) (the final logits).
extern "C" __global__ void __launch_bounds__(256)
add_normed(const float* base, const float* a, const float* w, const float* scale, float* out, int n, float eps) {
    __shared__ float part[8];
    const long long row = (long long)blockIdx.x * n;
    base += row;
    a += row;
    out += row;
    const int t = threadIdx.x, nb = n / 256;
    float x[NORM_MAXV];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            x[j] = a[t + 256 * j];
            ss += x[j] * x[j];
        }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((t & 31) == 0) part[t >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) tot += part[k];
    const float inv = rsqrtf(tot / n + eps), s = scale ? *scale : 1.0f;
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            const int k = t + 256 * j;
            out[k] = (base[k] + x[j] * inv * w[k]) * s;
        }
}

// Gemma-4's norm chains around the FFN, one workgroup per row (grid T, block 256), each step summed as the
// separate kernels sum it (add_rmsnorm_rows / add_normed / moe_combine: the same bits, fewer passes over the rows):
// gemma_attn_post  x_out = x + RMSNorm(a) * w_a (the attention output, post_attention_norm); y1 = RMSNorm(x_out) *
//                  w1 quantized into o1 (the dense MLP's input); y2 = RMSNorm(x_out) * w2 if w2 (the router's).
// gemma_ffn_post   c = the experts' outputs combined (moe_combine: exp_scale, wts); h = RMSNorm(mlp) * w1 +
//                  RMSNorm(c) * w2; x_out = (x + RMSNorm(h) * w_post) * *scale; y = RMSNorm(x_out) * w_next quantized
//                  into o (the next layer's attention input, or the output norm's).
// The sum of squares of the thread's values x[0 .. nb) (values t + 256 j of the row), as add_rmsnorm_rows takes
// it; part is reused, so the call ends with a barrier
__device__ __forceinline__ float row_rms_inv(const float x[NORM_MAXV], int nb, int n, float eps, float* part) {
    const int t = threadIdx.x;
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) ss += x[j] * x[j];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor(ss, off, 32);
    if ((t & 31) == 0) part[t >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) tot += part[k];
    __syncthreads();
    return rsqrtf(tot / n + eps);
}

extern "C" __global__ void __launch_bounds__(256)
gemma_attn_post(const float* x, const float* a, const float* w_a, float* x_out, const float* w1, float* y1, Q8Out o1,
                const float* w2, float* y2, int n, float eps) {
    __shared__ float part[8];
    __shared__ Q8Shared sh;
    const long long row = (long long)blockIdx.x * n;
    const int t = threadIdx.x, nb = n / 256;
    float v[NORM_MAXV];
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) v[j] = a[row + t + 256 * j];
    const float ia = row_rms_inv(v, nb, n, eps, part);
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            const int k = t + 256 * j;
            v[j] = (x[row + k] + v[j] * ia * w_a[k]) * 1.0f;  // add_normed without a scale
            x_out[row + k] = v[j];
        }
    const float inv = row_rms_inv(v, nb, n, eps, part);
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {  // uniform
            const int k = t + 256 * j;
            const float y = v[j] * inv * w1[k];
            y1[row + k] = y;
            if (w2) y2[row + k] = v[j] * inv * w2[k];
            if (o1.qK) q8_block_value(y, true, blockIdx.x * nb + j, t, sh, o1);
        }
}

extern "C" __global__ void __launch_bounds__(256)
gemma_ffn_post(const float* x, const float* mlp, const float* down, const float* wts, const int* ids,
               const float* exp_scale, int n_used, const float* w1, const float* w2, const float* w_post,
               const float* scale, float* x_out, const float* w_next, float* y, Q8Out o, int n, float eps) {
    __shared__ float part[8];
    __shared__ Q8Shared sh;
    const int tk = blockIdx.x;
    const long long row = (long long)tk * n;
    const int t = threadIdx.x, nb = n / 256;
    float h[NORM_MAXV], c[NORM_MAXV];
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) h[j] = mlp[row + t + 256 * j];
    const float im = row_rms_inv(h, nb, n, eps, part);
    const float* wt = wts + tk * n_used;
    const int* e = ids + tk * n_used;
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            const int k = t + 256 * j;
            h[j] = h[j] * im * w1[k];
            const float* d = down + (size_t)tk * n_used * n + k;  // moe_combine's sum
            float acc = exp_scale ? (d[0] * exp_scale[e[0]]) * wt[0] : wt[0] * d[0];
            for (int u = 1; u < n_used; ++u)
                acc += exp_scale ? (d[(size_t)u * n] * exp_scale[e[u]]) * wt[u] : wt[u] * d[(size_t)u * n];
            c[j] = acc;
        }
    const float ic = row_rms_inv(c, nb, n, eps, part);
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) h[j] = (h[j] + c[j] * ic * w2[t + 256 * j]) * 1.0f;
    const float ih = row_rms_inv(h, nb, n, eps, part), sc = scale ? *scale : 1.0f;
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {
            const int k = t + 256 * j;
            h[j] = (x[row + k] + h[j] * ih * w_post[k]) * sc;
            x_out[row + k] = h[j];
        }
    const float inv = row_rms_inv(h, nb, n, eps, part);
#pragma unroll
    for (int j = 0; j < NORM_MAXV; ++j)
        if (j < nb) {  // uniform
            const int k = t + 256 * j;
            const float v = h[j] * inv * w_next[k];
            y[row + k] = v;
            if (o.qK) q8_block_value(v, true, tk * nb + j, t, sh, o);
        }
}

extern "C" __global__ void __launch_bounds__(256)
geglu(const float* gate, const float* up, float* out, int n, int h, int gs, int us, Q8Out o) {
    __shared__ Q8Shared sh;
    const int i = blockIdx.x * 256 + threadIdx.x;
    float v = 0.0f;
    if (i < n) {
        const int r = i / h, c = i % h;
        const float x = gate[(size_t)r * gs + c];
        const float g = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
        v = g * up[(size_t)r * us + c];
        out[i] = v;
    }
    if (o.qK) q8_block_value(v, i < n, blockIdx.x, threadIdx.x, sh, o);  // uniform; n % 32 == 0 when set
}

extern "C" __global__ void __launch_bounds__(256)
softcap(float* x, int n, float cap) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}
