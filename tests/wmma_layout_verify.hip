// wmma_layout_verify -- pins down the gfx11 wave32 operand layout of v_wmma_i32_16x16x16_iu8 (Phase 20 W1).
//
// The WMMA prefill kernels depend on it, so it is checked on the hardware rather than assumed:
//   A (16 x 16 int8, M x K): lane i (and i + 16) holds row i, K values 0..15 packed in 4 dwords
//   B (16 x 16 int8, K x N): lane i (and i + 16) holds column i, K values 0..15
//   C (16 x 16 int32):       lane l holds column l % 16; element r of the 8-vector is row 2 r + l / 16
// v_wmma_f32_16x16x16_f16 (flash attention, W5.2) uses the same layout with FP16 operands and FP32 C.
// Self-contained (hipcc, no model).

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__global__ void wmma_once(const int8_t* A, const int8_t* B, int* C) {
    const int l = threadIdx.x, i = l & 15;
    v4i a, b;
    const int8_t* ar = A + i * 16;  // row i of A
    int8_t bc[16];
    for (int k = 0; k < 16; ++k) bc[k] = B[k * 16 + i];  // column i of B
    __builtin_memcpy(&a, ar, 16);
    __builtin_memcpy(&b, bc, 16);
    v8i c = {};
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false);
    for (int r = 0; r < 8; ++r) C[(2 * r + l / 16) * 16 + i] = c[r];
}

__global__ void wmma_f16_once(const _Float16* A, const _Float16* B, float* C) {
    const int l = threadIdx.x, i = l & 15;
    v16h a, b;
    for (int k = 0; k < 16; ++k) a[k] = A[i * 16 + k], b[k] = B[k * 16 + i];
    v8f c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    for (int r = 0; r < 8; ++r) C[(2 * r + l / 16) * 16 + i] = c[r];
}

static int check_f16() {
    _Float16 hA[256], hB[256];
    for (int m = 0; m < 16; ++m)
        for (int k = 0; k < 16; ++k) hA[m * 16 + k] = (_Float16)((m * 7 + k * 3) % 17 - 8);
    for (int k = 0; k < 16; ++k)
        for (int n = 0; n < 16; ++n) hB[k * 16 + n] = (_Float16)(((k * 5 + n * 11) % 23 - 11) * 0.25f);
    _Float16 *dA, *dB;
    float* dC;
    (void)hipMalloc(&dA, 512);
    (void)hipMalloc(&dB, 512);
    (void)hipMalloc(&dC, 256 * 4);
    (void)hipMemcpy(dA, hA, 512, hipMemcpyHostToDevice);
    (void)hipMemcpy(dB, hB, 512, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(wmma_f16_once, dim3(1), dim3(32), 0, 0, dA, dB, dC);
    float hC[256];
    (void)hipMemcpy(hC, dC, 256 * 4, hipMemcpyDeviceToHost);
    int bad = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            // Exact products and sums, but the instruction's internal FP32 sum is off by up to an ulp or so; a
            // layout error would be off by whole units
            float ref = 0;
            for (int k = 0; k < 16; ++k) ref += (float)hA[m * 16 + k] * (float)hB[k * 16 + n];
            if (fabsf(hC[m * 16 + n] - ref) > 1e-5f * (fabsf(ref) + 1.0f)) {
                if (bad < 5) printf("  C[%d][%d] = %.9g, expected %.9g\n", m, n, hC[m * 16 + n], ref);
                ++bad;
            }
        }
    printf("wmma_layout_verify: v_wmma_f32_16x16x16_f16 wave32 layout %s (%d of 256 wrong)\n", bad ? "FAIL" : "PASS", bad);
    return bad;
}

int main() {
    int8_t hA[256], hB[256];
    for (int m = 0; m < 16; ++m)
        for (int k = 0; k < 16; ++k) hA[m * 16 + k] = (int8_t)((m * 7 + k * 3) % 17 - 8);
    for (int k = 0; k < 16; ++k)
        for (int n = 0; n < 16; ++n) hB[k * 16 + n] = (int8_t)((k * 5 + n * 11) % 23 - 11);
    int8_t *dA, *dB;
    int* dC;
    (void)hipMalloc(&dA, 256);
    (void)hipMalloc(&dB, 256);
    (void)hipMalloc(&dC, 256 * 4);
    (void)hipMemcpy(dA, hA, 256, hipMemcpyHostToDevice);
    (void)hipMemcpy(dB, hB, 256, hipMemcpyHostToDevice);
    (void)hipMemset(dC, 0x7f, 256 * 4);
    hipLaunchKernelGGL(wmma_once, dim3(1), dim3(32), 0, 0, dA, dB, dC);
    int hC[256];
    (void)hipMemcpy(hC, dC, 256 * 4, hipMemcpyDeviceToHost);
    int bad = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            int ref = 0;
            for (int k = 0; k < 16; ++k) ref += hA[m * 16 + k] * hB[k * 16 + n];
            if (hC[m * 16 + n] != ref) {
                if (bad < 5) printf("  C[%d][%d] = %d, expected %d\n", m, n, hC[m * 16 + n], ref);
                ++bad;
            }
        }
    printf("wmma_layout_verify: v_wmma_i32_16x16x16_iu8 wave32 layout %s (%d of 256 wrong)\n", bad ? "FAIL" : "PASS", bad);
    bad += check_f16();
    return bad ? 1 : 0;
}
