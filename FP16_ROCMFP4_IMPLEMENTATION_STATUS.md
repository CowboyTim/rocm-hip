# FP16 ROCmFP4 Activation Path - Implementation Status

## Problem Statement

The FP16 ROCmFP4 quantized model (`Qwen3.5-4B-Q4_0_ROCMFP4_FP16.gguf`) produces garbled output (garbage characters) during text generation, while the BF16 and Q8_1 quantized variants produce coherent text.

## Root Cause (Identified)

**The MMQ kernel accesses F16 activation elements OUT OF BOUNDS.**

### Technical Details

The `vec_dot_rocmfp4_fp16_dp4a` function accesses FP16 data as:
```cpp
y_fp16[j * MMQ_TILE_NE_K + k01 + l]
```
- `j` = 0..7 (output columns)
- `MMQ_TILE_NE_K =`
- `k01` = 0, 64, 128, 192 (K offsets)
- `l` = 0..7 (VDR elements)

For the last kb0, max accessed = `(bpe-1) * 128 + 199`
Actual buffer size = `ne0 * ne1 / 2`
**Out of bounds = 128 F16 int* elements (always)**

This causes garbage data to be read from shared memory, corrupting FP4 weight dequantization.

## Changes Made

### Commits
1. `701c8aa` - Initial FP16 activation loading and dispatch fixes
2. `4451cc5` - Correct fp16_ncols_y formula (ne1 * 64)
3. `2a36f55` - Debug analysis identifying root cause

### Key Code Changes
- **mmvq.cu**: Added FP16-specific dispatch, fixed activation loading formula
- **mmq.cuh**: Added FP16 row stride constexpr, fixed activation offsets
- **ggml-cuda.cu**: Fixed dispatch to call `ggml_cuda_mul_mat_vec_q_fp16_f32`
- Removed broken `mul_mat_rocmfp4_fp16.cu/.cuh` (unused custom kernel)

## Build Status
✅ Build succeeds with no errors

## Runtime Status
- ✅ FP16 ROCmFP4 model runs without crashing
- ❌ Output is garbage characters (root cause identified)
- Model uses ~50 tokens/s (slower than BF16 ~375 tokens/s for prompt eval)

## Remaining Issues

### 1. OUT-OF-BOUNDS Access (CRITICAL)
The kernel accesses 128 F16 int* elements out of bounds for all K dimensions. This is the root cause of the garbage output.

### 2. Proposed Fixes (in order of priority)

**Option 1: Pad the F16 Buffer (Quick Fix)**
```cpp
const int64_t ne_src1_padded = (ne_src1 + 128) / 2 * 2;
CUDA_CHECK(cudaMemsetAsync(src1_f16.ptr + actual_size, 0, 128 * sizeof(int), stream));
```

**Option 2: Fix Activation Loading (Correct Fix)**
Adjust the formula to account for the extra elements accessed by vec_dot.

**Option 3: Specialized FP16 Kernel (Long-term Fix)**
Create a kernel that properly handles FP16 data layout.

## Comparison with Working Paths

| Path | Quantization | Activation | Status |
|------|-------------|------------|--------|
| BF16 | None (BF16) | cuBLAS GEMM | ✅ Coherent text |
| Q4_0_ROCMFP4 | FP4 | F32 → Q8_1 quantized | ✅ Coherent text |
| Q4_0_ROCMFP4_FP16 | FP4 | F32 → F16 | ❌ Garbage output |

## Key Files

- `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` - FP16 activation loading and dispatch
- `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` - FP16 row stride and activation offset
- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` - Dispatch path
- `llama.cpp/docs/debugging/FP16_ROCMFP4_DEBUG_ANALYSIS.md` - Detailed analysis

## Next Steps

1. Apply Option 1 (padding) to verify this fixes the issue
2. If padding works, commit the fix and test with full model
3. Consider Option 2 or 3 for a more robust long-term fix
