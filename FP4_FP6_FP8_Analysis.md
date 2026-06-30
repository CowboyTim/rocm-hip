# ROCmFP4 and FP6 Implementation Analysis

## 1. ROCmGPU FP4 Implementation (Current State)

### 1.1 FP4 Data Format

```
ROCmFP4: E2M1-derived with Codebook10 (tuned from standard MXFP4's Codebook12)
```

| Aspect | Value |
|--------|-------|
| Exponent bits | 2 |
| Mantissa bits | 1 |
| Values per sign | 8 (0, 1, 2, 3, 4, 6, 8, 10) |
| Block size | 32 values |
| Packed bytes | 16 (2 nibbles per byte) |
| Scale bytes | 2 (UE4M3, per 16-element half-block) |
| Block total | 18 bytes → 0.5625 BPW |

**Codebook10 (ROCmFP4):**
```c
static const int8_t rocmfp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10,
};
```

**Why 10 instead of 12?**
- Tuned after sampling Qwen3 dense tensors
- Reduces outlier pull without changing the packed 4-bit layout
- Same integer dot-product path

### 1.2 FP4 GPU Dequant Pipeline (gfx1151 - RDNA3.5)

```
Step 1: FP4 nibbles (4-bit) → int8 codes
        __builtin_amdgcn_perm (DP4A)    ← 1 instruction, hardware ✅

Step 2: int8 × int8 → int32 (DP4A dot product)
        __builtin_amdgcn_sudot4          ← 8-bit SIMD, hardware ✅

Step 3: int32 × UE4M3 scale → FP32
        Standard FP32 multiply           ← no tensor core needed

Step 4: Final result
        sum = d0 * sumi0 + d1 * sumi1    ← FP32
```

**Key functions:**

| File | Function | Role |
|------|----------|------|
| `rocmfp4_hip_codebook.cuh:33-57` | `rocmfp4_get_int_from_codebook_16()` | Codebook expansion via `__builtin_amdgcn_perm` |
| `common.cuh:688-693` | `ggml_cuda_dp4a()` | 8-bit DP4A dot product (SDOT4 on RDNA3) |
| `common.cuh:824-856` | `ggml_cuda_ue4m3_to_fp32()` | UE4M3 scale decode (bit manipulation) |

**Codebook expansion (DP4A-friendly):**
```cpp
constexpr uint32_t values0 = 0x03020100u;  // [0, 1, 2, 3]
constexpr uint32_t values1 = 0x0a080604u;  // [4, 6, 8, 10]
constexpr uint32_t values2 = 0xfdfeff00u;  // [0, -1, -2, -3]
constexpr uint32_t values3 = 0xf6f8fafcu;  // [-4, -6, -8, -10]

// __builtin_amdgcn_perm selects from these based on 4-bit index
```

### 1.3 FP4 Quantization (Float → FP4)

```
Step 1: Find amax in block
Step 2: Binary search for best UE4M3 scale around amax / 10.0f
Step 3: For each candidate scale, compute MSE
Step 4: Pack nibbles with chosen scale
```

**Key functions:**

| File | Function | Role |
|------|----------|------|
| `rocmfp4.c:287-329` | `rocmfp4_choose_scale_ue4m3_exhaustive_unweighted()` | Exhaustive scale search |
| `rocmfp4.c:106-137` | `rocmfp4_best_index_scaled()` | Best codebook index for given value |
| `rocmfp4.c:259-285` | `rocmfp4_nearest_scale_ue4m3()` | Binary search for scale byte |

### 1.4 FP4 Scale Decode (UE4M3 → FP32)

Two implementations exist:

**CPU path** (`rocmfp4.c:50-104`): 127-entry LUT
```c
static const float rocmfp4_scale_ue4m3_half[127] = {
    ROCMFP4_SCALE_SUB(0), ROCMFP4_SCALE_E1(0), ROCMFP4_SCALE_E2(0), ...
};
```

**GPU path** (`common.cuh:824-856`): bit manipulation (faster on GPU)
```cpp
static __device__ __forceinline__ float ggml_cuda_ue4m3_to_fp32(uint8_t x) {
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        const uint32_t bits = ((uint32_t) exp + 119u) << 23 | ((uint32_t) man << 20);
        memcpy(&raw, &bits, sizeof(float));
    }
    return raw * 0.5f;
}
```

### 1.5 FP4 GPU Kernel Integration

| Component | File | Integration Point |
|-----------|------|-------------------|
| Type registration | `ggml.c:676-691` | GGML_TYPE_INFO array |
| CPU dispatch | `ggml-cpu.c:232-243` | from_float, vec_dot, vec_dot_type |
| MMQ tile loading | `mmq.cuh:911-954` | `load_tiles_rocmfp4()` |
| Vec dot | `vecdotq.cuh:353-373` | `vec_dot_rocmfp4_q8_1()` |
| FlashAttention | `fattn-common.cuh:318-319` | K/V dequant for attention |
| Dequant | `dequantize.cuh:54-73` | Device dequantize functions |
| Convert | `convert.cu:491-523` | CUDA dequant kernels |
| Cpy | `cpy.cu:130-186` | F32<->FP4 copy kernels |
| Build | `CMakeLists.txt` | Source file registration, template instances |
| Python | `gguf-py/gguf/quants.py` | Quantize/dequantize classes |

---

## 2. FP8 on ROCm (Scale Format Reference)

### 2.1 FP8 Hardware Support

| Architecture | FP8 Intrinsics? | FP8 Format |
|-------------|-----------------|------------|
| gfx942 (CDNA2) | Yes | `__hip_fp8_e4m3_fnuz` (FNUZ only) |
| gfx1200/1201 (RDNA4) | Yes | `__nv_fp8_e4m3` (OCP) |
| gfx950 (CDNA4) | Yes | Both E4M3 and E5M2 |
| **gfx1151 (RDNA3.5)** | **No** | Software fallback |

```cpp
// From amd_hip_fp8.h
#if (defined(__gfx942__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx950__))
    #define HIP_FP8_CVT_FAST_PATH 1
#else
    #define HIP_FP8_CVT_FAST_PATH 0  // gfx1151: software
#endif
```

### 2.2 FP8 Scale Use in llama.cpp

FP8 is used **only for UE4M3 scales**, not as a weight format:
- ROCmFP4: UE4M3 scale → FP8 e4m3 bits → FP32 multiply
- NVFP4: UE4M3 scale → FP8 e4m3 bits → FP32 multiply

FP8 is **not** a weight quantization type in llama.cpp. No `GGML_TYPE_FP8_*` exists.

### 2.3 FP8 Decode on gfx1151 (Software)

```cpp
// amd_hip_fp8.h:526-570 - cast_to_f8_from_f32, cast_to_f32_from_f8
// Software: bit manipulation (~25-40 instructions/val)
```

**Why not LUT?**
- 127 finite values × 2 sign = 254 entries
- Constant memory overhead
- Bit manipulation compiles to ~6-8 instructions on GPU

---

## 3. FP6 Implementation Analysis

### 3.1 FP6 Format Definitions

**FP6 E2M3:** 1 sign, 2 exponent (bias=1), 3 mantissa bits

| Value | Code (hex) | Value |
|-------|-----------|-------|
| 0.0 | 0x00 | -0.0 | 0x01 |
| +1.0 | 0x02 | -1.0 | 0x03 |
| +2.0 | 0x04 | -2.0 | 0x05 |
| +4.0 | 0x08 | -4.0 | 0x09 |
| Subnormals | 0x08-0x38 | Various | 0x09-0x39 |

**Max magnitude:** 3.75 (E=2, M=7: 1.111 × 2^1)
**Subnormals:** 7 (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
**Total finite codes:** 56 (28 positive, 28 negative)
**Reserved:** 16 (8 pos inf, 8 neg inf)

**FP6 E3M2:** 1 sign, 3 exponent (bias=3), 2 mantissa bits

| Value | Code (hex) | Value |
|-------|-----------|-------|
| 0.0 | 0x00 | -0.0 | 0x01 |
| +0.25 | 0x02 | -0.25 | 0x03 |
| +0.5 | 0x04 | -0.5 | 0x05 |
| +1.0 | 0x06 | -1.0 | 0x07 |
| +2.0 | 0x08 | -2.0 | 0x09 |
| +4.0 | 0x0A | -4.0 | 0x0B |
| +8.0 | 0x0C | -8.0 | 0x0D |

**Max magnitude:** 14.0 (E=6, M=3: 1.75 × 2^3)
**Subnormals:** 3 (0.0625, 0.125, 0.1875)
**Total finite codes:** 64 (32 positive, 32 negative)
**Reserved:** 16 (8 pos inf, 8 neg inf)

### 3.2 ROCm Native FP6 Support

From `amd_hip_fp6.h`:
```cpp
// Hardware intrinsics ONLY on gfx950 (CDNA4)
#if __gfx950__
    // __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16
    // __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16
#else
    // Software: fcbx::from_float<Encoding::E2M3, true>(x, 0)
#endif
```

gfx1151 (RDNA3.5) → **software fcbx fallback**. No hardware acceleration.

### 3.3 FP6 vs FP4 Comparison

| Aspect | FP4 (ROCmFP4) | FP6 (proposed) |
|--------|--------------|----------------|
| Values per byte | 2 nibbles | **1.33 values** (4 per 3 bytes) |
| Codebook size | 16 (4 bits) | **64-128** (6 bits) |
| `__builtin_amdgcn_perm` | ✅ Works (4-bit index) | ❌ Won't work (6-bit index) |
| GPU dequant | DP4A table lookup | Software table lookup |
| gfx1151 hardware | ✅ `__builtin_amdgcn_perm` | ❌ fcbx software |
| BFW | 0.5625 BPW | **~1.0 BPW** |
| Max magnitude (Codebook10) | 10.0 | 3.5 (E2M3) / 8.0 (E3M2) |

### 3.4 Why FP6 Can't Use DP4A

`__builtin_amdgcn_perm` works on **4-bit indices** (`q & 0x07070707`). FP6 needs **6-bit indices** (`q & 0x3F3F3F3F`), which is incompatible with the DP4A permutation pattern.

Alternative approaches for FP6 GPU dequant:
1. **LUT in constant memory** - single memory load per scale value
2. **Bit manipulation** - same as FP8, ~6-8 instructions
3. **fcbx library** - ROCm's software implementation

### 3.5 FP6 Codebook Options

**Option A: Full E2M3 codebook (16 values per sign)**
```
0, ±0.25, ±0.5, ±0.75, ±1.0, ±1.25, ±1.5, ±1.75, ±2.0, ±2.5, ±3.0, ±3.5, ±3.75
```
Max magnitude: 3.75
Bits: 6

**Option B: Subsampled E2M3 (9 values per sign, like FP4 Codebook10)**
```
0, ±1, ±2, ±3, ±4, ±6, ±8, ±10, ±12
```
This would match FP4's pattern but doesn't correspond to actual E2M3 encoding.

**Option C: E3M2 codebook (16 values per sign)**
```
0, ±0.25, ±0.5, ±1.0, ±2.0, ±4.0, ±8.0,
±0.5, ±0.75, ±1.5, ±1.75, ±3.0, ±3.5, ±7.0, ±14.0
```
Max magnitude: 14.0
Bits: 6

### 3.6 FP6 Block Structure Proposal

```c
typedef struct {
    uint8_t qs[24];  // 32 FP6 values packed: 4 values per 3 bytes = 24 bytes
    uint8_t e[2];    // 2 UE4M3 scale bytes (per 16-element half-block)
} block_fp6;         // 26 bytes total → 0.8125 BPW (E2M3)
```

---

## 4. FP4 vs FP6 vs FP8 Performance on gfx1151

### 4.1 Instruction Count Comparison

| Operation | FP4 | FP6 | FP8 |
|-----------|-----|-----|-----|
| Codebook expand | `__builtin_amdgcn_perm` (~1 instr) | Software LUT (~3-5 instr) | N/A |
| DP4A dot product | `__builtin_amdgcn_sudot4` (1 instr) | N/A | N/A |
| Scale decode | Bit manip (~6 instr) | Bit manip (~6 instr) | Software (~25-40 instr) |
| Total dequant | ~15 cycles | ~25 cycles | N/A |

### 4.2 Hardware Support Matrix

| Format | Hardware on gfx1151? | Path | Instructions/val |
|--------|---------------------|------|-----------------|
| FP4 (ROCmFP4) | ✅ | DP4A | ~15 |
| FP4 (native) | ❌ | fcbx software | ~25 |
| FP8 scales | ❌ | Software bit ops | ~30 |
| FP6 (E2M3/E3M2) | ❌ | fcbx software | ~25 |
| FP16/BF16 | ✅ | Native half/bf16 | ~8 |
| FP32 | ✅ | Native | ~8 |

### 4.3 Why ROCmFP4 > ROCm Native FP4 on gfx1151

ROCm native `__hip_fp4_e2m1::operator float()` uses:
```cpp
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(...)  // hardware
#else
    fcbx::to_float<E2M1, true>(...)  // software - slower on gfx1151
#endif
```

ROCmFP4 uses `__builtin_amdgcn_perm` which:
- Works on ALL RDNA/GCN architectures
- Single hardware instruction
- Not gfx950-only

### 4.4 Matmul Pipeline

```
FP4 weights × Q8_0 activations → FP32 output
```

| Step | ROCmFP4 | cuBLAS |
|------|---------|--------|
| Dequant | DP4A + LUT | N/A |
| Multiply | DP4A int8×int8 | FP32 GEMM |
| Scale | FP32 × UE4M3 | N/A |
| Result | FP32 sum | FP32 GEMM |

**Why not cuBLAS?**
- cuBLAS has no FP4 or custom quantization format
- Custom kernel uses DP4A which is hardware on RDNA3.5
- Memory bandwidth savings (8x less for FP4) outweigh extra dequant cost

**Why not FP4 compute?**
- No FP4 × FP4 hardware instruction anywhere
- Only Blackwell has native FP4 MMQ
- FP4 exists purely for storage (0.5 bytes/val vs 4 bytes/val FP32)

---

## 5. Summary of Key Findings

1. **ROCmFP4 is better than native ROCm FP4 on gfx1151** because `__builtin_amdgcn_perm` is hardware-accelerated on all RDNA, while native FP4 only has hardware on gfx950.

2. **FP6 cannot use DP4A** because 6-bit codes don't fit the 4-bit-perm pattern. Would need software table lookup or fcbx.

3. **FP8 is not a weight quantization** in llama.cpp. It's used solely for UE4M3 scale encoding. FP8 hardware on gfx1151 is limited to CDNA/RDNA4.

4. **FP4 dequant to FP32 is fast** (~15 cycles on gfx1151) via DP4A. The matmul itself is FP32, which is fully hardware-accelerated.

5. **cuBLAS is used for dense FP32/FP16/BF16** only. All quantized matmul is handwritten HIP kernels using DP4A/MFMA.

6. **FP4 memory bandwidth savings (8x)** are the primary motivation, not compute speed. FP4 enables larger models in the same VRAM.

7. **FP6 would be ~1.0 BPW** (vs FP4 at 0.56 BPW) and would have worse accuracy per bit due to wider codebook. Not a clear improvement over existing formats.

---

## 6. Relevant File Paths

### Core Implementation
- `ggml/rocmfp4/rocmfp4.h` - Block struct definitions, function prototypes
- `ggml/rocmfp4/rocmfp4.c` - CPU quantize, dequantize, vec_dot
- `ggml/rocmfp4/rocmfp4_hip.cu` - HIP dequant kernels
- `ggml/rocmfp4/rocmfp4_hip_codebook.cuh` - DP4A codebook expansion
- `ggml/rocmfp4/rocmfp4_hip_scale.cuh` - GPU scale LUT

### CUDA Backend
- `ggml/src/ggml-cuda/common.cuh` - UE4M3 decode, DP4A intrinsic
- `ggml/src/ggml-cuda/vecdotq.cuh` - FP4 vec_dot functions
- `ggml/src/ggml-cuda/mmq.cuh` - MMQ tile loading
- `ggml/src/ggml-cuda/dequantize.cuh` - Device dequant functions
- `ggml/src/ggml-cuda/convert.cu` - CUDA dequant kernels
- `ggml/src/ggml-cuda/cpy.cu` - F32<->FP4 copy
- `ggml/src/ggml-cuda/mmvq.cu` - MVVQ kernels
- `ggml/src/ggml-cuda/fattn-common.cuh` - FlashAttention FP4 dequant

### Type Registration
- `ggml/include/ggml.h` - GGML_TYPE enum
- `ggml/src/ggml.c` - Type info registration
- `ggml/src/ggml-cpu/ggml-cpu.c` - CPU dispatch
- `include/llama.h` - File type enum

### Build & Tools
- `ggml/src/CMakeLists.txt` - Source registration
- `ggml/src/ggml-cuda/CMakeLists.txt` - Template instances
- `gguf-py/gguf/quants.py` - Python quantization
- `src/llama-quant.cpp` - Quantization tool

### ROCm Headers (System)
- `/opt/rocm/include/hip/amd_detail/amd_hip_fp4.h` - FP4 types
- `/opt/rocm/include/hip/amd_detail/amd_hip_fp6.h` - FP6 types
- `/opt/rocm/include/hip/amd_detail/amd_hip_fp8.h` - FP8 types
