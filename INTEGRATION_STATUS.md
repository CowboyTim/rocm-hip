# ROCmFP4 + gfx1151 Strix Halo — Project Status & Plan

---

## Table of Contents

1. [Architecture Analysis](#architecture-analysis)
2. [Submodule & Branch Status](#submodule--branch-status)
3. [ROCmFP4 Integration](#rocmfp4-integration)
4. [Code Migration Verification](#code-migration-verification)
5. [FP16 ROCmFP4 Activation Path — Implementation Status](#fp16-rocmfp4-activation-path---implementation-status)
6. [HIP Dequantization Implementation Plan](#hip-dequantization-implementation-plan)
7. [13-Week Roadmap: MFMA/WMMA/FP4 + KV-Cache Fast-Path](#13-week-roadmap-mfmawmmafp4--kv-cache-fast-path)
8. [Quick-Start Checklist](#quick-start-checklist)

---

## Architecture Analysis

### Local Build vs Official Qwen3.8-Flash-DGX

This local build (`Qwen3.8-27B`) is **NOT** the official Qwen3.8-Flash-Next model. It appears to be a smaller, distilled variant from the Qwen3.5/Qwen3 series.

| | **Local Model** | **Official Qwen3.8-Flash-Next (HF/DGX)** |
|---|---|---|
| Param Count | ~27B total | ~180B total (~6B active) |
| Type | **Dense** (single FFN per layer) | **MoE** (512 experts, top-10 routed) |
| Layers | 64 decoder layers | 48 decoder layers |
| Attention | Alternating linear_attention / full_attention (every 4th) | Hybrid GDN + QSA sparse attention + multi-hyperconnections |
| N-gram (PLE) | **No** n-gram embedding table or caching | **Yes** — ~51B PLE n-gram embedding tables injected into tokens |
| Mamba SSM | Yes (GDN uses mamba_ssm_dtype float32) | Yes (same lineage) |
| Multimodal | Has vision_config but no image/video tokens in local files | Text + Image + Video (vision_start/end, audio tokens) |
| Tokenizer | BPE (`Qwen2Tokenizer`), vocab 248,320 | Same base tokenizer |

**Local Model:** Dense transformer with alternating linear and full attention layers. No expert routing, no MoE parameters, no n-gram caching infrastructure. bf16/fp16 precision (no NVFP4 quantization). Uses `Qwen3_5ForConditionalGeneration` architecture.

**Official Qwen3.8-Flash-Next:** Mixture-of-Experts: 512 experts per layer, top-10 active (~6B params activated). N-gram embedding tables (~51B params for PLE lookup). NVFP4 W4A4 quantization via NVIDIA Model Optimizer. Hybrid GDN + QSA sparse attention + multi-hyperconnection streams. Full multimodal support (text, image, video, audio).

**Conclusion:** The local build is a dense variant from the Qwen3.5/Qwen3 series, not the Flash-Next MoE+n-gram model described on GitHub or HuggingFace. It lacks the key innovations of the official Flash-Next model:
1. **No MoE routing** — single feed-forward network per layer
2. **No n-gram caching** — no PLE (Paged LLM Embedding) infrastructure
3. **Smaller scale** — ~27B vs ~180B total parameters (~6B active in Flash-Next)

---

## Submodule & Branch Status

### Current State

```
Branch: rocmfp4-strix-upstream-merged
Base: upstream/master (commit 5735e10c4)
Ahead of upstream/master: 53 commits
Working tree: clean
```

### Key ROCmFP4 Commits on `rocmfp4-strix-upstream-merged`

| Commit | Description |
|--------|-------------|
| `73ab415ca` | llama: disable graph reuse when contexts share memory under SPLIT_MODE_TENSOR |
| `3761dd1b5` | vulkan: enable CM2 decode-vector extension |
| `4758cb05e` | common: default GPU_MAX_HW_QUEUES=1 on HIP to fix idle GPU load |
| `c91c22a1a` | adjust ncols_picker for routed MoE in mul_mat_q_case function |
| `bb251a74f` | fix: VMM pool cuMemSetAccess for ROCm gfx1151 APU |
| `69b586372` | Remove broken diffusion-gemma.cpp model to unblock build |
| `b9cea4d4c` | Add GGML_USE_HIP_MAPPED_COPY flag for zero‑copy view optimization on UMA (Strix Halo) |
| `9f22190a5` | **Merge ROCm FP4 (gfx1151) support**: add GGML_USE_ROCMFP4 option, HIP dequant kernels, copy kernels, and build script integration |

### Submodules

- **llama.cpp/** — Upstream master with ROCmFP4 integration (branch: `rocmfp4-strix-upstream-merged`)
- **rocmfp4-llama/** — ROCmFP4 fork with MTP support (branch: `nemotron-mtp-rocmfp4-strix`)

---

## ROCmFP4 Integration

### Overview

This project integrates ROCmFP4 FP16 activation support into the upstream `llama.cpp` master branch. The integration is based on ROCmFP4 implementation by Tim Aerts (aardbeiplantje).

### Quick Reference

```bash
# Build with ROCmFP4 support
bash build_llama.cpp.sh

# Test ROCmFP4 with FP16 activations
LLAMA_CPP_DIR=$(pwd)/llama.cpp/build bash llama.sh completion \
  --model Qwen3.5-4B-ROCMFP4.gguf -p "Hi" -n 500

# Quantize BF16 to ROCmFP4 FP16 variants
./llama.cpp/build/bin/llama-quantize Qwen3.5-4B-BF16.gguf out.gguf Q4_0_ROCMFP4_FP16
./llama.cpp/build/bin/llama-quantize Qwen3.5-4B-BF16.gguf out.gguf Q4_0_ROCMFP4_FAST_FP16
```

### Repo Structure

Docker-based llama.cpp deployment for AMD Strix Halo (gfx1151) with ROCm FP4 support. Three submodules: `llama.cpp` (fork on `rocmfp4-strix` branch), `minijinja` (upstream), `rocmfp4-llama` (source branch with ROCmFP4 patches).

### Building the Docker Image

```bash
docker buildx bake -f docker-bake.hcl _local
```

Produces `local/ai/llama.cpp-gfx1151:latest`. The ROCm tarball is fetched from an S3 bucket and cached under `/var/cache/downloads` during build.

### Running

```bash
./run.sh server          # detached (default)
./run.sh cli -m /models/model.gguf   # interactive TTY
./run.sh bench           # benchmark suite
./run.sh quantize in.gguf out.gguf   # BF16/FP16 -> ROCmFP4
./run.sh tail            # follow logs
./run.sh bash            # interactive shell in container
```

Mount models via `MODELS_DIR=/path/to/models ./run.sh`. HF cache via `HF_HOME`.

### llama.sh Subcommands (inside container)

- **server** (default) — starts llama.cpp server + nginx on unix socket `/tmp/llama.sock`
- **cli** — interactive CLI chat
- **quantize** — quantize GGUF (default output type: `Q4_0_ROCMFP4_STRIX_LEAN`)
- **bench** — comprehensive benchmark suite

### Presets

`/llamacpp_presets.ini` — `[<ModelName>]` sections override defaults. Model presets are matched by filename prefix. Add new model presets by copying the `[Qwen3.6-35B-A3B-ROCMFP4]` block and adjusting `ctx-size`/other params.

### Key Env Vars (host or container)

`HSA_OVERRIDE_GFX_VERSION=11.5.1`, `GGML_HIP_FORCE_KV_GPU=1`, `GGML_HIP_ALLOC_GRAPH_RESERVE=2048`, `HSA_ENABLE_SDMA=0`.

### Perl XS Module (nginx integration)

`Llama/` — Perl XS bindings to llama.cpp, built into the Docker image. Tests run during build (`make test`). The module is loaded by nginx via `perl_require llama.pm` in `nginx.conf`.

### Build Script Flags

`build_llama.cpp.sh` — key CMake flags: `-DGGML_USE_ROCMFP4=ON`, `-DAMDGPU_TARGETS=gfx1151`, `-DGGML_HIP_UMA=ON`, `-DGGML_HIP_FORCE_KV_GPU=ON`, `-DGGML_HIP_USE_MMAP_ALIASED=ON`.

### ROCmFP4 Fixes Applied to Upstream

The following fixes were applied to make ROCmFP4 work like in `rocmfp4-llama`:

#### 1. HIP Dequantization Kernels (`ggml/src/ggml-hip/dequantize_hip.cuh`)

- **Problem**: ROCmFP4 dequant functions (`dequantize_rocmfp4`, `dequantize_rocmfp4_fast`) were stubs returning zeros
- **Fix**: Added proper implementation using `rocmfp4_ue4m3_to_fp32_half_finite()` and `rocmfp4_decode_i8()` from `rocmfp4_hip_scale.cuh`
- **Added**: `#include "../../rocmfp4/rocmfp4_hip_scale.cuh"`

#### 2. MTP Tensor Loading for qwen35 (`src/models/qwen35.cpp`)

- **Problem**: MTP (NextN) tensors skipped when `load_mtp=false` (default), causing "unused tensor" warnings
- **Fix**: Changed `mtp_flags` from `!ml.load_mtp ? TENSOR_SKIP : 0` to always `0` (required for qwen35 models)

#### 3. Gated Delta Net Kernel (`ggml/src/ggml-cuda/gated_delta_net.cu`)

- **Problem**: llama.cpp had a refactored kernel with different signature (`state_slot_stride`, separate `state` pointer) that lacked HIP backend implementation
- **Fix**: Replaced with working rocmfp4-llama version (simpler signature, working HIP build)
- **Files**: `ggml/src/ggml-cuda/gated_delta_net.cu`, `ggml/src/ggml-cuda/gated_delta_net.cuh`
- **Added**: `ggml_cuda_op_gated_delta_net_fused_cache` fallback implementation

### Current Model Status (as of 2026-08-28)

| Model | Status |
|-------|--------|
| Qwen3-0.6B-Q8_0.gguf | ✅ Works correctly |
| Qwen3.5-4B-ROCMFP4.gguf | ❌ Garbled output (qwen35 arch with recurrent/MTP layers) |

**Known limitation:** Qwen3.5-4B-ROCMFP4.gguf (qwen35 arch with recurrent/MTP layers) still produces garbled output in llama.cpp due to API divergence from rocmfp4-llama's working qwen35.cpp. The `rocmfp4-llama` submodule (branch `nemotron-mtp-rocmfp4-strix`) has the working implementation.

**Investigation done:**
- Recurrent layer detection confirmed correct (layers 0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30 recurrent; 3,7,11,15,19,23,27,31,32 attention)
- Tensor shapes in `build_layer_attn_linear` and `build_recurrent_attn` look correct
- Removed `res->t_layer_inp[il] = inpL;` (differed from rocmfp4-llama) — no change
- Replaced gated_delta_net kernel with rocmfp4-llama working version — no change
- Using `build_delta_net_fused (AR)` path with `fused_gdn_ar=1, fused_gdn_ch=1`

**Remaining investigation needed:**
- Check if fused vs autoregressive path differs (rocmfp4-llama uses simpler fused path)
- Compare `build_delta_net_autoregressive` implementation
- Verify recurrent state initialization in `build_rs`
- Check recurrent cache (`ssm_states_all`, `conv_states_all`) zeroing at context creation
- Debug `ggml_gated_delta_net` kernel output for NaN/Inf

---

## Code Migration Verification

### Report Summary

| Check | Status | Notes |
|-------|--------|-------|
| Upstream Master Merge | ✅ PASS | All commits from origin/master included |
| ROCmFP4 Source Migration | ⚠️ IN PROGRESS | Some items completed, others pending |

### Part 1: Files Copied ✅ COMPLETED

**Vulkan Shaders:**
- ✓ dequant_rocmfp4.comp ✓ DONE
- ✓ dequant_rocmfp4_fast.comp ✓ DONE

Build system compiles `.comp` → embedded binary during build.

### Part 2: Quantization Types ✅ VERIFIED PRESENT

In `include/llama.h` lines 164-165:

```c
LLAMA_FTYPE_MOSTLY_Q4_0_ROCMFP4_STRIX    = 105, // Strix Halo quality/speed recipe
LLAMA_FTYPE_MOSTLY_Q4_0_ROCMFP4_STRIX_LEAN = 106, // Strix Halo size-biased K/V recipe
```

✅ Already merged into target branch — no action needed.

### Part 3: Implementation Code Gaps ⚠️ IN PROGRESS

**Completed (Aug 14):**
- set-rows.cu copied completely
- ggml-cuda.cu dispatcher updated for COPY/GET_ROWS/SET_ROWS operations
- fattn.cu attention dispatch + both #ifdef paths updated

**Remaining Work:**
- Review vulkan.cpp ~96 reference differences
- Build test to verify compilation succeeds
- Runtime testing on gfx1151 device

---

## FP16 ROCmFP4 Activation Path — Implementation Status

### Problem Statement

The FP16 ROCmFP4 quantized model (`Qwen3.5-4B-Q4_0_ROCMFP4_FP16.gguf`) produces garbled output (garbage characters) during text generation, while the BF16 and Q8_1 quantized variants produce coherent text.

### Root Cause (Identified)

**The MMQ kernel accesses F16 activation elements OUT OF BOUNDS.**

#### Technical Details

The `vec_dot_rocmfp4_fp16_dp4a` function accesses FP16 data as:

```cpp
y_fp16[j * MMQ_TILE_NE_K + k01 + l]
```

- `j` = 0..7 (output columns)
- `MMQ_TILE_NE_K =`
- `k01` = 0, 64, 128, 192 (K offsets)
- `l` = 0..7 (VDR elements)

For the last kb0, max accessed = `(bpe-1) * 128 + 199`. Actual buffer size = `ne0 * ne1 / 2`. **Out of bounds = 128 F16 int* elements (always).**

This causes garbage data to be read from shared memory, corrupting FP4 weight dequantization.

### Changes Made

**Commits:**
1. `701c8aa` — Initial FP16 activation loading and dispatch fixes
2. `4451cc5` — Correct fp16_ncols_y formula (ne1 * 64)
3. `2a36f55` — Debug analysis identifying root cause

**Key Code Changes:**
- **mmvq.cu**: Added FP16-specific dispatch, fixed activation loading formula
- **mmq.cuh**: Added FP16 row stride constexpr, fixed activation offsets
- **ggml-cuda.cu**: Fixed dispatch to call `ggml_cuda_mul_mat_vec_q_fp16_f32`
- Removed broken `mul_mat_rocmfp4_fp16.cu/.cuh` (unused custom kernel)

### Build & Runtime Status

- ✅ Build succeeds with no errors
- ✅ FP16 ROCmFP4 model runs without crashing
- ❌ Output is garbage characters (root cause identified)
- Model uses ~50 tokens/s (slower than BF16 ~375 tokens/s for prompt eval)

### Remaining Issues

**1. OUT-OF-BOUNDS Access (CRITICAL)**

The kernel accesses 128 F16 int* elements out of bounds for all K dimensions. This is the root cause of the garbage output.

**2. Proposed Fixes (in order of priority):**

**Option 1: Pad the F16 Buffer (Quick Fix)**

```cpp
const int64_t ne_src1_padded = (ne_src1 + 128) / 2 * 2;
CUDA_CHECK(cudaMemsetAsync(src1_f16.ptr + actual_size, 0, 128 * sizeof(int), stream));
```

**Option 2: Fix Activation Loading (Correct Fix)**

Adjust the formula to account for the extra elements accessed by vec_dot.

**Option 3: Specialized FP16 Kernel (Long-term Fix)**

Create a kernel that properly handles FP16 data layout.

### Comparison with Working Paths

| Path | Quantization | Activation | Status |
|------|-------------|------------|--------|
| BF16 | None (BF16) | cuBLAS GEMM | ✅ Coherent text |
| Q4_0_ROCMFP4 | FP4 | F32 → Q8_1 quantized | ✅ Coherent text |
| Q4_0_ROCMFP4_FP16 | FP4 | F32 → F16 | ❌ Garbage output |

### Key Files

- `llama.cpp/ggml/src/ggml-cuda/mmvq.cu` — FP16 activation loading and dispatch
- `llama.cpp/ggml/src/ggml-cuda/mmq.cuh` — FP16 row stride and activation offset
- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` — Dispatch path
- `llama.cpp/docs/debugging/FP16_ROCMFP4_DEBUG_ANALYSIS.md` — Detailed analysis

---

## HIP Dequantization Implementation Plan (Port from SYCL)

The HIP backend is missing dequantize implementations for several quantization types that exist in the SYCL backend (`ggml/src/ggml-sycl/dequantize.hpp`). This causes build failures or runtime issues when using IQ/MXFP4/NVFP4 quantized models.

### Missing Quantization Types (SYCL → HIP)

| Type | SYCL Function | HIP Status |
|------|---------------|------------|
| IQ2_XXS | `dequantize_iq2_xxs` | Missing |
| IQ2_XS | `dequantize_iq2_xs` | Missing |
| IQ2_S | `dequantize_iq2_s` | Missing |
| IQ3_XXS | `dequantize_iq3_xxs` | Missing |
| IQ3_S | `dequantize_iq3_s` | Missing |
| IQ1_S | `dequantize_iq1_s` | Missing |
| IQ1_M | `dequantize_iq1_m` | Missing |
| IQ4_NL | `dequantize_iq4_nl` | Missing |
| IQ4_XS | `dequantize_iq4_xs` | Missing |
| MXFP4 | `dequantize_mxfp4` | Missing |
| NVFP4 | `dequantize_nvfp4` | Missing |

### Phased Implementation

**Phase 1: IQ2 Family** (IQ2_XXS, IQ2_XS, IQ2_S)
- Port from SYCL `dequantize_iq2_xxs`, `dequantize_iq2_xs`, `dequantize_iq2_s`
- Add constant lookup tables: `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `ksigns_iq2xs`, `kmask_iq2xs` as `__constant__` memory

**Phase 2: IQ3 Family** (IQ3_XXS, IQ3_S)
- Port from SYCL `dequantize_iq3_xxs`, `dequantize_iq3_s`
- Add lookup tables: `iq3xxs_grid`, `iq3s_grid`, reuse `ksigns_iq2xs`, `kmask_iq2xs`

**Phase 3: IQ1 Family** (IQ1_S, IQ1_M)
- Port from SYCL `dequantize_iq1_s`, `dequantize_iq1_m`
- Add lookup table: `iq1s_grid_gpu`

**Phase 4: IQ4 Family** (IQ4_NL, IQ4_XS)
- Port from SYCL `dequantize_iq4_nl`, `dequantize_iq4_xs`
- Add lookup table: `kvalues_iq4nl`

**Phase 5: MXFP4/NVFP4**
- Port from SYCL `dequantize_mxfp4`, `dequantize_nvfp4`
- Add lookup tables: `kvalues_mxfp4`, scale conversion functions

**Phase 6: Block Dequantize Kernels**
- Add `__global__` kernels for each new type (e.g., `dequantize_block_iq2_xxs`, etc.)
- Required by `get_rows_cuda_kq` and `convert.cu`

### Key Files

- `/llama.cpp/ggml/src/ggml-hip/dequantize_hip.cuh` — Main dequantize functions
- Source: `/llama.cpp/ggml/src/ggml-sycl/dequantize.hpp` (SYCL reference)

### Implementation Notes

- Adapt SYCL `dfloat2` → HIP `float2`, `__dpct_inline__` → `__device__ __forceinline__`
- SYCL `sycl::fma` → `fmaf`, `sycl::vec` → manual operations
- Add `__constant__` lookup tables at file scope
- Include necessary headers: `ggml.h`, `rocmfp4_hip_scale.cuh`

---

## 13-Week Roadmap: MFMA/WMMA/FP4 + KV-Cache Fast-Path

All items keep the original 13-week calendar; the KV-cache / prefill work is parallelised with the kernel development because it touches the same files and dispatch points.

### 0️⃣ Quick-Look — Where KV-Cache / Prefill Lives Today

| Component | File(s) | Current Path |
|-----------|---------|--------------|
| KV-cache allocation / layout | `src/llama-kv-cache.cpp`, `src/llama-kv-cache-iswa.cpp`, `src/llama-kv-cache-dsv4.cpp` | Linear buffer, 2-D (layers × heads) with stride-based indexing |
| Prefill (prompt processing) | `llama.cpp → llama_encode → llama_eval → ggml_graph_compute → ggml_cuda_mul_mat_q / ggml_cuda_flash_attn` | Calls the same GEMM / Flash-Attention kernels used for decode |
| Cache-update (decode step) | `llama_decode → ggml_cuda_mul_mat_q` (K-proj, V-proj) + `ggml_cuda_flash_attn` | Same kernels as prefill, but one token at a time |

Because **prefill = one large GEMM + Flash-Attn** and **decode = many tiny GEMM + Flash-Attn**, any kernel-level speed-up automatically benefits both. The extra work is to *avoid redundant memory movement / layout changes* when the prompt is long and the cache is filled in one shot.

### 1️⃣ Extended Discovery (Weeks 1–2) — KV-Cache / Prefill Audit

| New Task | Owner | Deliverable |
|----------|-------|-------------|
| Map current KV-cache layout — draw exact `ggml_tensor` shapes for `K` and `V` per layer/head, note strides, padding, contiguity | KV-cache engineer | Diagram + table (layer, head, K shape, V shape, stride) |
| Profile prefill vs decode — run a 4k-token prompt on gfx1151, capture `rocprof` traces for prefill GEMM, first decode step, steady-state decode | Perf engineer | CSV + rocpdf screenshots |
| Identify "copy-on-write" — does the current code copy the prompt's K/V into the cache, or write directly into the pre-allocated cache buffer? | KV-cache engineer | Boolean + code pointers |
| Add a "prefill-only" benchmark — `llama-cli -p "<4k-token prompt>" -n 0` (no decode) to isolate prefill cost | Perf engineer | Baseline numbers (tokens/s, latency, LDS/reg usage) |

*All of this lives in the same Week 1–2 window as the original discovery (baseline ROCm-FP4 kernel audit, ISA gap analysis, baseline token-gen numbers).*

### 2️⃣ Build-System Flag — Add a **KV-Cache Fast-Path Switch** (Weeks 2–3)

| New CMake Option | Location | Reason |
|------------------|----------|--------|
| `GGML_USE_GFX1151_KV_FAST` (default OFF) | `ggml/CMakeLists.txt` → propagated to `ggml/src/ggml-hip/CMakeLists.txt` | Turns on the *in-place KV-cache write* + *prefill-only Flash-Attn* kernels without affecting the generic path |
| Propagate to `build_llama.cpp.sh` (`-DGGML_USE_GFX1151_KV_FAST=ON`) | Build script | One-line enable for Strix-Halo |

*No code changes yet — just the switch.*

### 3️⃣ Core Kernel Work — **MFMA / WMMA / VOP3P** (Weeks 3–8)

Exactly the same kernels as the original plan. **Additional requirement:** make the new MFMA/WMMA kernels **KV-cache aware** (see §4).

| Phase | Kernel | KV-Cache / Prefill Hook |
|-------|--------|--------------------------|
| 3.1 MFMA GEMM primitive | `ggml/src/ggml-hip/mfma_gfx1151.cu` (new) | Accept an optional `dst_base + dst_stride` that points directly into the pre-allocated KV buffer; if `dst_stride == 0` fall back to temporary buffer (current behavior) |
| 3.2 WMMA fallback | `wmma_gfx1151.cu` (new) | Same stride-aware signature |
| 3.3 VOP3P dot-product | `dot_vop3p_gfx1151.cu` (new) | No change — used by both prefill & decode |
| 3.4 Flash-Attn gfx1151 | `fattn_gfx1151.cu` (new) | Two entry points: `flash_attn_prefill` — takes the whole prompt K/V tensors, writes **directly into the KV cache** (no intermediate copy); `flash_attn_decode` — unchanged (single-token) |
| 3.5 mmq / mmvq dispatch | `mmq.cu`, `mmvq.cu` | When `src0->type == GGML_TYPE_Q4_0_ROCMFP4` **and** `GGML_USE_GFX1151_KV_FAST` → call the stride-aware MFMA GEMM that writes straight into the KV cache |
| 3.6 Copy / pack | unchanged | Verify `cpy_q_q_block` still optimal for the new layout (it already copies packed FP4 bytes) |

**Acceptance criteria** (same as original + KV-cache specific):
- Prefill of 4k tokens writes K/V **directly** into the cache (zero extra `hipMemcpyAsync`)
- Decode step sees the same layout — no extra transpose
- End-to-end token/s ≥ 1.8× **and** prefill latency ↓ ≥ 30% (because the copy is gone)

### 4️⃣ KV-Cache / Prefill Integration (Weeks 4–6)

| Week | Activity | Owner | Exit Criteria |
|------|----------|-------|---------------|
| 4 | **In-place KV write path** — modify `llama_kv_cache_update` to call the new stride-aware MFMA GEMM instead of `ggml_cuda_mul_mat_q` + `hipMemcpyAsync` | KV-cache engineer | Unit test: `llama_kv_cache_update` writes exactly the same bytes as before (memcmp) |
| 5 | **Prefill-only Flash-Attn entry** — add `flash_attn_prefill` that receives the whole prompt K/V and writes into the KV cache in a single kernel launch | Flash-Attn engineer | Prefill latency ↓ ≥ 30% on 4k-token prompt (measured with `rocprof`) |
| 5–6 | **Decode path unchanged** — ensure existing decode path still works (no regression) | QA engineer | All existing decode tests pass |
| 6 | **Benchmark suite** — run `llama-bench -m model.fp4.gguf -p <4k> -n 0` (prefill only) and `llama-bench -m model.fp4.gguf -p <4k> -n 128` (prefill + decode) | Perf engineer | Prefill latency ↓ ≥ 30%; overall tokens/s ≥ 1.8× baseline |

*All of these weeks overlap with the original kernel development (Weeks 3–8) — you are just adding a few extra hooks.*

### 5️⃣ Integration & Dispatch (Weeks 8–9) — Same as Original + KV Flag

| Task | Owner | Note |
|------|-------|------|
| Add `if (GGML_USE_GFX1151_KV_FAST && ggml_hip_same_mapped_region(...))` in `ggml_cuda_cpy` to **skip the copy** when source & destination are the same mapped region (already done for zero-copy) | Copy engineer | Re-use existing helper |
| Extend `ggml_cuda_mul_mat_q` / `ggml_cuda_flash_attn` with `#if defined(GGML_USE_GFX1151_KV_FAST)` guards that call the new *prefill* / *KV-aware* kernels | Dispatch engineer | Same guard pattern as `GGML_USE_GFX1151_MFMA` |
| Expose both flags in `build_llama.cpp.sh` (`-DGGML_USE_GFX1151_MFMA=ON -DGGML_USE_GFX1151_KV_FAST=ON`) | Build engineer | One-line enable |

### 🔟 Validation, Testing & Performance (Weeks 10–12) — **Add KV-Cache Specific Tests**

| Test | Metric | Target |
|------|--------|--------|
| Unit test — KV-cache in-place write | `memcmp(cache_before, cache_after) == 0` after prefill | ✅ |
| Prefill-only latency (4k tokens) | `rocprof` wall-time | ≤ 0.7× original |
| Decode steady-state throughput | tokens/s | ≥ 1.8× baseline |
| End-to-end token generation (prefill + decode) | tokens/s, latency | ≥ 1.8× baseline, latency ↓ ≥ 20% |
| Memory traffic (`rocprof --stats`) | LDS / VRAM bytes moved | ↓ ≥ 30% for prefill |
| Numerical parity (logits vs FP32 CPU) | max-abs-diff < 1e-4 | ✅ |
| Long-run stress (1h) | No crashes, no drift | ✅ |

### 🔟 Documentation, Release & Handoff (Week 13) — Add KV-Cache Sections

| Doc Update | Owner |
|------------|-------|
| `docs/backend/ROCm.md` — new section "gfx1151 KV-cache fast-path" (flags, expected speed-up) | Docs owner |
| `docs/build.md` — list both flags (`GGML_USE_GFX1151_MFMA`, `GGML_USE_GFX1151_KV_FAST`) | Docs owner |
| Release notes — "gfx1151 native MFMA/WMMA + zero-copy KV-cache prefill" | PM |
| Upstream PR — single PR containing both flag sets | Repo maintainer |

### 📋 Updated 13-Week Gantt (Condensed)

| Week | Main Focus (Original) | **Extra KV / Prefill Work** |
|------|----------------------|----------------------------|
| 1–2 | Discovery, baseline | **KV-cache audit, prefill profiling** |
| 2–3 | Build-system flags (`GGML_USE_GFX1151_MFMA`) | **Add `GGML_USE_GFX1151_KV_FAST`** |
| 3–5 | MFMA GEMM primitive | **Stride-aware MFMA API** |
| 5–6 | WMMA + VOP3P | **No extra** |
| 6–7 | Flash-Attn gfx1151 | **Add `flash_attn_prefill`** |
| 8–9 | Dispatch + build flag | **Add KV-fast dispatch** |
| 4–6 (parallel) | — | **In-place KV write, prefill kernel** |
| 9–11 | Validation | **Add KV-cache benchmarks** |
| 12 | Power / stress | **Same** |
| 13 | Docs / release | **Add KV-cache docs** |

*Total calendar stays **13 weeks**; the KV-cache work is **parallelised** with the kernel development because it touches the same files and dispatch points.*

### 🎯 Bottom Line

| What You Gain | How |
|---------------|-----|
| **2×-wide MFMA / WMMA** for FP4 GEMM | MFMA kernels (Weeks 3–5) |
| **Native 4-bit matrix-multiply** (no de-quant) | Fused de-quant inside MFMA (Week 3) |
| **VOP3P dot-products** for attention scores | Dot kernels (Week 5) |
| **Zero-copy KV-cache prefill** (no extra `hipMemcpy`) | In-place MFMA write + `flash_attn_prefill` (Weeks 4–6) |
| **Overall token-generation ↑ ≈ 2×**, prefill latency ↓ ≥ 30% | End-to-end bench (Weeks 10–12) |

All of this lives under **two orthogonal CMake flags** (`GGML_USE_GFX1151_MFMA` and `GGML_USE_GFX1151_KV_FAST`) so you can enable them independently on any gfx1151 system. The plan re-uses the same code paths you are already building for the MFMA path, keeping the engineering effort modest while delivering a **large, measurable win for both prompt processing and steady-state decoding** on Strix Halo (gfx1151) APUs.

---

## Quick-Start Checklist for the First Engineer

```text
[ ] Clone the rocmfp4-strix-upstream-merged branch
[ ] Add GGML_USE_GFX1151_MFMA and GGML_USE_GFX1151_KV_FAST to ggml/CMakeLists.txt
[ ] Create ggml/src/ggml-hip/mfma_gfx1151.cu/.cuh
[ ] Implement a 64×64×64 FP4-aware MFMA GEMM (inline asm)
[ ] Add unit test (GoogleTest) — compare to reference GEMM
[ ] Hook the kernel into mmq.cu via GGML_USE_GFX1151_MFMA guard
[ ] Add stride-aware signature for KV-cache write
[ ] Implement flash_attn_prefill in fattn_gfx1151.cu
[ ] Wire GGML_USE_GFX1151_KV_FAST dispatch in ggml_cuda_mul_mat_q / ggml_cuda_flash_attn
[ ] Run ./build_llama.cpp.sh -DGGML_USE_GFX1151_MFMA=ON -DGGML_USE_GFX1151_KV_FAST=ON
[ ] Verify token/s on a 7B FP4 model (≥1.8× speed-up)
[ ] Run prefill-only benchmark (llama-bench -p <4k> -n 0) — latency ↓ ≥ 30%
```

---

## 📂 How to Persist This Document

Save this file as `INTEGRATION_STATUS.md` in the repository root:

```bash
git add INTEGRATION_STATUS.md && git commit -m "docs: merge all ROCmFP4 status documents into single comprehensive guide"
```
