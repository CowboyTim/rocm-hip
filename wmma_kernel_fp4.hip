#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// FP4 ROCmFP4 Matmul using WMMA intrinsics on gfx1151
//
// ROCmFP4: E2M1-derived with Codebook10 (tuned from standard MXFP4's Codebook12)
// Values: 0, 1, 2, 3, 4, 6, 8, 10 (per sign) — 8 positive, 8 negative
// Packed: 2 nibbles per byte (16 values per 8 bytes)
// Scale: UE4M3 encoded as uint8_t, per block
//
// Since gfx1151 has no native FP4 WMMA intrinsic, we:
// 1. Dequantize FP4 nibbles → FP16 via DP4A codebook permutation (__builtin_amdgcn_perm)
// 2. Use WMMA for FP16 matmul accumulation
// 3. Convert FP16 result → FP32 output

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

    // FP4 fragment: 16 fp16 elements for WMMA input (dequantized from FP4)
    half16 w_frag = {};
    half16 a_frag;
    half16 c_frag = {};

    // Dequantize FP4 nibbles via codebook lookup, then to __half for WMMA
    // Row-major layout: weights_fp4[lane + ele*16]
    for (int ele = 0; ele < 16; ++ele) {
        uint8_t nibble = weights_fp4[lane + ele * 16];
        int8_t code = fp4_codebook_int8[nibble];
        w_frag[ele] = (__half)((float)code * scale);
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
