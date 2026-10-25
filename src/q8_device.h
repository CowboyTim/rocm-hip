// q8_device.h -- device-side activation quantization for the int8 matvecs (qmatvec.h), shared by every kernel
// that produces a matvec input, so they can quantize their output in the same launch.
//
// One 256-value block, one value per thread of a 256-thread group, into both of ggml's activation formats:
//   Q8_0 (ROCmFP4 weights): 32-value blocks, d = amax / 127 stored as FP16, q = round-half-even(x * 127 / amax)
//                           (the x86 SIMD quantize_row_q8_0 the reference runs, not the _ref variant), plus
//                           the sum of each block's q (for the offsets of Q5_1 weights: ggml's Q8_1 s / d).
//   Q8_K (K-quant weights): d = 1 / (-127 / max) with max the signed value of largest |x| (first on ties),
//                           q = min(127, round-half-even(x * -127 / max)), plus the sum of every 16 q values.
// Bit-identical to ggml's CPU quantizers, so each block's integer dot product matches the reference exactly.

#ifndef Q8_DEVICE_H
#define Q8_DEVICE_H

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Output pointers; the same layout as the first members of qmatvec_q8 (qmatvec.h). qK == nullptr: no output.
struct Q8Out {
    int8_t* q0;
    float* d0;
    int8_t* qK;
    float* dK;
    int* bsK;
    int* s0;  // sums of each Q8_0 block's 32 q values
};

// Shared scratch for one 256-thread group
struct Q8Shared {
    float a[8];
    int i[8];
    float v[8];
};

// Quantize value v, element t (0..255) of 256-value block b. Every thread of the workgroup must call this
// the same number of times (it contains a __syncthreads); threads of an inactive group pass active = false.
__device__ __forceinline__ void q8_block_value(float v, bool active, int b, int t, Q8Shared& sh, const Q8Out& o) {
    const int lane = t & 31, w = t >> 5, i = b * 256 + t;
    const float a = active ? fabsf(v) : 0.0f;

    // Q8_0 over the wave's 32 values
    float am = a;
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor(am, off, 32));
    const float id0 = am != 0.0f ? 127.0f / am : 0.0f;
    const int q0 = active ? (int)rintf(v * id0) : 0;
    int s0 = q0;
    for (int off = 16; off > 0; off >>= 1) s0 += __shfl_xor(s0, off, 32);
    if (active) {
        o.q0[i] = (int8_t)q0;
        if (lane == 0) {
            o.d0[b * 8 + w] = __half2float(__float2half(am / 127.0f));
            o.s0[b * 8 + w] = s0;
        }
    }

    // Q8_K over the block's 256 values: largest |x|, lowest index on ties
    float ba = a, bv = v;
    int bi = t;
    for (int off = 16; off > 0; off >>= 1) {
        const float oa = __shfl_xor(ba, off, 32), ov = __shfl_xor(bv, off, 32);
        const int oi = __shfl_xor(bi, off, 32);
        if (oa > ba || (oa == ba && oi < bi)) { ba = oa; bi = oi; bv = ov; }
    }
    if (lane == 0) { sh.a[w] = ba; sh.i[w] = bi; sh.v[w] = bv; }
    __syncthreads();
    float fa = sh.a[0], fv = sh.v[0];
    int fi = sh.i[0];
#pragma unroll
    for (int k = 1; k < 8; ++k)
        if (sh.a[k] > fa || (sh.a[k] == fa && sh.i[k] < fi)) { fa = sh.a[k]; fi = sh.i[k]; fv = sh.v[k]; }
    __syncthreads();  // sh is reused by the next call
    int q = 0;
    float iscale = 0.0f;
    if (fa != 0.0f) {
        iscale = -127.0f / fv;
        q = min(127, (int)rintf(iscale * v));
    }
    int s16 = q;
    for (int off = 8; off > 0; off >>= 1) s16 += __shfl_xor(s16, off, 32);
    if (active) {
        o.qK[i] = (int8_t)q;
        if (t == 0) o.dK[b] = fa != 0.0f ? 1.0f / iscale : 0.0f;
        if ((lane & 15) == 0) o.bsK[b * 16 + w * 2 + (lane >> 4)] = s16;
    }
}

#endif // Q8_DEVICE_H
