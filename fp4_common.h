// fp4_common.h -- ROCmFP4 Codebook10 canonical definitions
// Shared by host code (.cpp, plain clang++) and device kernels (.hip).
// No HIP includes here so it works in both contexts.
//
// Contract (matches FP4_FP6_FP8_Analysis.md section 1 and llama.cpp block layout):
//   - Block of 32 values: 16 packed data bytes (2 nibbles/byte, low nibble first)
//     followed by 2 UE4M3 scale bytes, one per 16-element half-block.
//   - A_scales[r][h] scales A row r, K-half-block h (K values = K/16 scale bytes).
//   - B_scales[c][h] scales B column c, K-half-block h.

#ifndef FP4_COMMON_H
#define FP4_COMMON_H

#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define FP4_HD __host__ __device__
#else
#define FP4_HD
#endif

namespace rocmfp4 {

constexpr int BLOCK_SIZE    = 32;   // values per block
constexpr int HALF_BLOCK    = 16;   // values per scale byte
constexpr int NIBBLES_BYTE  = 2;    // values per packed data byte

// Canonical Codebook10 (E2M1-derived, tuned on Qwen3 tensors).
// Nibbles 0-7 -> +{0,1,2,3,4,6,8,10}; nibbles 8-F -> their negatives. Sign is bit 3.
static const int8_t CODEBOOK[16] = {
     0,  1,  2,  3,  4,  6,  8, 10,
     0, -1, -2, -3, -4, -6, -8,-10 };

// DP4A codebook expansion tables as uint32 register constants (little-endian byte order):
//   P_LO/P_HI hold the positive magnitudes split [0..3] / [4,6,8,10]; N_LO/N_HI their negations.
// Verified on gfx1151: __builtin_amdgcn_perm(X,Y,m) per mask byte selects Y[m&3] when bit2 clear
// and X[m&3] when set; mask bytes >= 8 are undefined -- keep masks <= 7 (see fp4_dequant in kernels).
static const uint32_t CB_P_LO = 0x03020100u;   // bytes [+0,+1,+2,+3]
static const uint32_t CB_P_HI = 0x0a080604u;   // bytes [+4,+6,+8,+10]
static const uint32_t CB_N_LO = 0xfdfeff00u;   // bytes [-0,-1,-2,-3]
static const uint32_t CB_N_HI = 0xf6f8fafcu;   // bytes [-4,-6,-8,-10]

FP4_HD inline float dequant_code(int nibble) { return (float)CODEBOOK[nibble & 0x0F]; }

// UE4M3 byte -> float. Exact port of ggml_cuda_ue4m3_to_fp32 (Analysis doc section 1.4):
// top nibble (bits [7:4]) is the exponent field incl. sign-bit position, low 3 bits mantissa.
FP4_HD inline float ue4m3_to_float(uint8_t x) {
    int exp = (x >> 3) & 0xF;
    int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = (float)man * (1.0f / 512.0f);   // ldexp(man, -9), exact for man < 8
    } else {
        uint32_t bits = (((uint32_t)exp + 119u) << 23) | ((uint32_t)man << 20);
        memcpy(&raw, &bits, sizeof(float));
    }
    return raw * 0.5f;
}

// Packed data access: value k of a row lives in byte k/2, low nibble first.
FP4_HD inline uint8_t packed_nibble(const uint8_t* base, int k) {
    return (base[k >> 1] >> ((k & 1) << 2)) & 0x0F;
}

// Buffer sizing helpers (K must be a multiple of BLOCK_SIZE).
constexpr size_t data_bytes_per_row(int K) { return (size_t)(K / NIBBLES_BYTE); }
constexpr size_t scale_bytes_per_line(int K) { return (size_t)(K / HALF_BLOCK); }

} // namespace rocmfp4

#endif // FP4_COMMON_H
