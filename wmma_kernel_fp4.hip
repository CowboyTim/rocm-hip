#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// FP4 ROCmFP4 Matmul using WMMA intrinsics on gfx1151
//
// ROCmFP4: E2M1-derived with Codebook10
// Values: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
// Packed: 1 nibble per byte (single nibble per entry)
// Scale: UE4M3 encoded as uint8_t, per block
//
// Dequantization via LDS LUT (fast on-chip SRAM, ~2-4 cycle latency):
//   - 16-entry LUT in LDS: entry[nibble] = FP16 dequant value
//   - 16 × 2 bytes = 32 bytes total
//   - Replaces FP32 multiply with single LDS load
//
// Then feed FP16 into native WMMA 16x16x16 matrix units

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
static const int8_t fp4_codebook_int8[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

extern "C" __global__ void wmma_matmul_fp4(const unsigned char* weights_fp4,
                                              const __half* activations,
                                              __half* result,
                                              float scale)
{
    const int tid = threadIdx.x;
    const int lane = tid % 16;

    // 16-entry LDS LUT: entry[nibble] = FP16 dequant value
    // LDS latency ~2-4 cycles vs ~100+ cycles for global memory
    // Total size: 16 × 2 bytes = 32 bytes
    __shared__ half lds_lut[16];

    // Build LUT: only threads 0-15 participate, each builds one entry
    if (tid < 16) {
        float val = (float)fp4_codebook_int8[tid] * scale;
        lds_lut[tid] = __half(val);
    }
    __syncthreads();

    // FP4 fragment: 16 fp16 elements for WMMA input (dequantized from FP4)
    half16 w_frag = {};
    half16 a_frag;
    half16 c_frag = {};

    // Dequantize FP4 nibbles via LDS LUT lookup
    // Row-major layout: weights_fp4[lane + ele*16]
    // Each byte contains one nibble (4 bits), LDS load replaces FP32 multiply
    // Nibbles are indices into codebook: 0-7 -> positive values (0,1,2,3,4,6,8,10)
    for (int ele = 0; ele < 16; ++ele) {
        uint8_t nibble = weights_fp4[lane + ele * 16];
        int idx = (nibble < 8) ? nibble : 7;
        w_frag[ele] = lds_lut[idx];  // Fast LDS indexed load
    }

    // Load FP16 activations row-major: B[ele][lane] = ele*16 + lane
    for (int ele = 0; ele < 16; ++ele) {
        a_frag[ele] = activations[ele * 16 + lane];
    }

    // WMMA: FP16 x FP16 -> FP16, 16x16x16
    // OPSEL=false: output at even indices [0,2,...,30] of c_frag
    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(w_frag, a_frag, c_frag, false);

    // Store FP16 result
    // Lanes 0-15 -> rows 0,2,...,14; lanes 16-31 -> rows 1,3,...,15
    for (int ele = 0; ele < 8; ++ele) {
        const int r = ele * 2 + (tid / 16);
        result[16 * r + lane] = c_frag[ele * 2];
    }
}
