# Project Plan & Analysis: rocm-hip (Strix Halo HIP POC)

## What It Is

A HIP/ROCm proof-of-concept for **AMD Strix Halo (gfx1151, RDNA3.5 APU)** proving out two ideas:

1. `mmap()` file-backed memory as zero-copy GPU data via `hipHostRegister`
2. Low-precision matmul (FP16 WMMA + ROCmFP4 FP4) using RDNA3.5 hardware intrinsics

~35 commits; a second branch (`q4_0_rocmfp4_q8_1`) extends this into full llama.cpp integration work.

## Core Demo Pattern (repeated in every binary)

1. Create buffer in `/dev/shm/*.bin` -> `ftruncate` -> `mmap(MAP_SHARED)` (file-backed "unified memory")
2. `hipHostRegister(ptr, size, hipHostRegisterMapped/Default)` to pin pages so the GPU can walk them; `hipHostGetDevicePointer` for the device address (identical on UMA APU -- printed for verification)
3. Launch kernel with **no `hipMemcpy` anywhere** -- GPU reads/writes CPU-visible file-backed RAM directly
4. Kernels are pre-compiled to `.co` objects (`hipcc --genco`, `-mno-wavefrontsize64`) and loaded at runtime by `mmap`ing the `.co` file itself + `hipModuleLoadData`. Host `.cpp` is compiled with plain clang++ (no offload); split build rule in Makefile.

## Components on `main`

| Target | Files | What it does |
|---|---|---|
| `mandelbrot` | main.cpp, kernels.hip | Flagship mmap demo: 800x600 escape-time renderer writes RGBA pixels straight into a `/dev/shm` mapping; verifies known set points (cardioid cusp, seahorse valley, filaments); dumps PPM |
| `(halo)` | mmap.cpp | Original minimal POC: `+1.0f` increment kernel over 100MB of mapped memory; carries the verbose "Strix Halo halo" design comments |
| `wmma_matmul` | wmma_kernel.hip, wmma_main.cpp | 16x16 FP16 matmul via `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`; documents RDNA3 OPSEL=false unpacked-fragment lane/row mapping in detail; CPU-verifies C=(i+1)(j+1)*16 |
| `wmma_matmul_fp4` | wmma_kernel_fp4.hip, wmma_main_fp4.cpp | **ROCmFP4 Codebook10** format (E2M1-derived: 0, +/-1..+/-4,+/-6,+/-8,+/-10, tuned on Qwen3 tensors): dequant nibble->FP16 through a 32-byte LDS LUT, then into the FP16 WMMA tensor core -- the best gfx1151 path |
| `wmma_matmul_fp4_native` | *_fp4_native.* | FP4xFP4->FP32 32x16 tile; references `__builtin_amdgcn_wmma_f32_32x16x128_f4` but that's gfx1250-only, so ships a scalar fallback for gfx1151 |
| `wmma_bench` + run_bench.py | 5 bench kernels + host + Python orchestrator | Head-to-head FP4 matmul at configurable M/N/K (default 1K^3): scalar-LUT vs DP4A-intrinsic vs native-fallback vs FP4->FP16-WMMA+LDS-LUT. CPU reference in-mapping, mismatch%/GFLOPS table, JSON export. DP4A variant expands codebook with `__builtin_amdgcn_perm` (4 const tables) and dot-products with `__builtin_amdgcn_sudot4` -- pure integer hardware path |

## Analysis Doc & Second Branch

- **FP4_FP6_FP8_Analysis.md** -- deep llama.cpp source study: why ROCmFP4 beats native FP4 on gfx1151 (`perm` is HW on all RDNA; native FP4 cvt needs gfx950), why FP6 can't use DP4A (6-bit codes > 4-bit perm mask), FP8 exists only as UE4M3 scale encoding, ~cycle-count estimates per dequant path.

- **Branch `q4_0_rocmfp4_q8_1`** -- the "real" project: llama.cpp submodule (pinned to a fork carrying ROCmFP4 ftype 107/108 integration, 53 commits ahead of upstream at the time), `INTEGRATION_STATUS.md` (incl. an analysis concluding the local Qwen3.8-27B build is a dense Qwen3.5-lineage model, not the official Flash-DGX MoE+n-gram one), `build_llama.cpp.sh` for Strix Halo + rocWMMA headers, mmap zero-copy GGUF loading, 13-week roadmap toward MFMA/WMMA/FP4 + KV-cache fast path.

## Git State & Observations

- Remote `origin/main`; working tree clean. History arc: trivial test -> error-handling fixes -> Mandelbrot-over-UMa -> WMMA intrinsics exploration -> FP4 benchmark suite -> llama.cpp submodule added Jul 1 then **removed** from main ("no llama.cpp"), leaving `.gitmodules` as an empty file and the gitlink deleted -- that work lives only on the q4 branch.
- A prior commit also deleted `.COMMIT_NOTES.md`, `PLAN.md`/`TODO.md`, `rocmfp4-llama/`, old docs (consolidated into INTEGRATION_STATUS.md on q4).

### Issues Worth Fixing

1. README still says "Initial Commit"; AGENTS.md on main is a stub (the useful version is on q4)
2. `wmma_main_fp4_native.cpp:47` has a stale prototype `wmma_matmul_fp4_builtin` that matches no symbol -- dead code (dispatch uses string names), but misleading
3. The 5th bench kernel (`wmma_bench_fp4_dp4a_int8`) is compiled by Makefile/run_bench.py but **never executed** by `wmma_bench_main.cpp` (only 4 methods run)
4. DP4A dequant in `wmma_bench_fp4_dp4a_hardware.hip:37-61`: the second `perm` selects between negative tables based on bit 2, but Codebook10's sign is bit 3 of the nibble -- positive codes come back as negatives; comments in-file even flag the confusion ("we need two separate perms..."). Scalar/negative data works around it via clamping; this method likely fails verification for real signed data
5. Codebooks + `HIP_CHECK` macros are duplicated across ~9 files instead of shared headers

## Build

ROCm at `/opt/rocm` (hipcc available). Makefile targets: `mandelbrot wmma_matmul wmma_matmul_fp4 wmma_matmul_fp4_native wmma_bench`, all pinned to `--offload-arch=gfx1151`. Nothing built yet (no `.co` artifacts in tree).


# Inference Engine Roadmap -- New Phase

This section outlines the forward-looking plan: building a minimalist inference engine that leverages ROCmFP4 FP4 quantization and zero-copy mmap() GGUF loading on AMD Strix Halo (gfx1151) with 128GB UMA.

## Design Principles

- **Zero-copy everywhere**: GGUF tensors live in file-backed mapped memory (either `/dev/shm` or directly from the .gguf file), GPU reads/writes them through `hipHostRegister`. No `hipMemcpy` ever.
- **No host-to-device conversion**: weights stay in their quantized format on the GPU. Dequant happens inline via hardware-accelerated DP4A (`__builtin_amdgcn_perm` + `__builtin_amdgcn_sudot4`). FP4->WMMA path uses LDS LUT for nibble-to-FP16 conversion before tensor core compute.
- **Qwen 3 model family**: dense transformer architecture, not MoE. Assumes Qwen3 lineage tokenizer + RoPE. Quantization targets Qwen3's known sensitivity to weight precision.
- **128GB UMA advantage**: no need to partition between "VRAM" and system RAM. Everything is shared physical memory accessible by both CPU and GPU at identical addresses. The constraint becomes memory bandwidth, not capacity -- hence why FP4 matters more than FP16.
- **C API surface**: all public interfaces exported as `extern "C"` functions. C++ HIP implementation hidden behind opaque handles. Designed for eventual Perl XS CPAN module consumption.

## Quantization Strategy Decision Matrix

| Format | BPW | HW dequant on gfx1151? | Path | FLOPs/val | Verdict |
|--------|-----|----------------------|------|-----------|---------|
| ROCmFP4 (Codebook10) | 0.5625 | Yes -- DP4A perm | `perm` -> `sudot4` -> UE4M3 scale mul | ~15 cycles | **Primary: fastest path** |
| NVFP4 | 0.5625 | No -- fcbx software | Software fallback | ~25 cycles | Keep available but slower |
| MXFP4 | 0.5625 | No -- fcbx software | Software fallback | ~25 cycles | Same as NVFP4 |
| Q8_0 | 1.0 | Yes -- int8 SDOT4 | DP4A dot product only, no dequant needed | ~8 cycles | **Fallback for head/output layers** |
| Q4_K / Q5_K | ~0.6-0.78 | Yes -- DP4A pattern | Similar to FP4 but different block layout | ~12-15 cycles | Viable alternative, more complex blocks |
| FP16/BF16 WMMA | 2.0/2.0 | Native tensor core | Direct WMMA intrinsic call | ~8 cycles (no dequant) | For KV-cache, RoPE positions, bias terms |

Key conclusions from FP4_FP6_FP8_Analysis.md applied to inference engine:

1. **ROCmFP4 is the weight format of choice**. `__builtin_amdgcn_perm` works on all RDNA architectures with a single hardware instruction. It expands nibbles into int8 codes via a 32-byte constant table lookup in one instruction, then feeds directly into `__builtin_amdgcn_sudot4` for the dot-product accumulation. This is the fastest known dequant+matmul path on gfx1151 at 0.56 BPW.

2. **Qwen3 benefits most from ROCmFP4 over native FP4**. The analysis doc shows native ROCm FP4 requires gfx950-only intrinsics (Blackwell). On Strix Halo it falls back to fcbx software which compiles to ~25 instructions per value vs DP4A's ~15.

3. **FP8 has no role as a weight format**. In llama.cpp it exists only as UE4M3 scale encoding for FP4 blocks. FP8 hardware intrinsics are only available on CDNA/RDNA4 -- not on RDNA3.5 Strix Halo.

4. **FP6 is not viable on this GPU**. Six-bit codes don't fit `__builtin_amdgcn_perm`'s 4-bit-per-index pattern. Software dequant would be slower than even native FP4. Skip entirely for now.

5. **Mixed precision layout**: Use ROCmFP4 for all attention and MLP weight matrices (the bulk of params). Use Q8_0 for LM-head output projection (needs full dynamic range for vocabulary sampling). Use FP16/BF16 only for KV-cache storage (no quantization needed, fast WMMA access) and RoPE position tensors (small, computed on host).

## Target Architecture

```
CPU (x86):                          GPU (gfx1151 RDNA3.5):
+---------------------+            +--------------------+
| GGUF .file         |  mmap()    | hipHostRegister    |
|  [Qwen3 model]     |+---------->|  (zero-copy pin)   |
|                    |  +share    |                    |
| tokenizer          |  memory    | DP4A dequant:      |
| (int32 IDs)        |<----------+  __builtin_perm     |
|                    |  mmap()    |  __builtin_sudot4   |
| host dispatch      |  reads CPU | FP4->FP16 LUT       |
| (C API calls)      |  visible data | WMMA matmul       |
|                    |  in shared RAM | LDS LUT expand    |
+---------------------+            +--------------------+
        ^                                    |
        |                                      v
+-------------------------------+      +--------------------+
| KV Cache (GPU-side)           |<-----| attention kernel   |
| mapped from /dev/shm          |   register pointers    | FP4 weights +            |
| ~shared with GPU via          |                   FP16 KV values   |
| hipHostRegisterMapped         |             RoPE + softmax   |
+-------------------------------+            +--------------------+
```

The critical insight: on Strix Halo's UMA architecture, a single `mmap()` of the GGUF file or `/dev/shm` buffer gives both CPU and GPU access to the same physical pages. No copy path exists because there is no separate VRAM -- the GPU's memory controller accesses unified memory directly.

## API Design Overview

### C API Header (`inference.h`)

All functions exported as `extern "C"`. Opaque handles for all state.

```c
// Model loading
inference_model_t* inference_load_model(const char* gguf_path);       // mmap() loads GGUF header + tensors index
void               inference_free_model(inference_model_t* model);      // unregisters and unmmaps

// Tokenization (CPU side)
int*              inference_tokenize(inference_model_t* model, const char* text, int* out_len);
void              inference_free_tokens(int* tokens, int len);

// Inference
int    inference_kv_cache_init(inference_model_t* model, int n_ctx);  // allocates KV cache in /dev/shm
void   inference_kv_cache_free(inference_model_t* model);
void   inference_set_seed(inference_model_t* model, unsigned int seed);
int    inference_decode_token(inference_model_t* model, int token_id, int* next_token_out);
float  inference_sample_logits(inference_model_t* model, float temperature, int* vocab_ids, float* logits);

// Control
int    inference_n_ctx(const inference_model_t* model);
int    inference_batch_size(const inference_model_t* model);
const char* inference_model_name(const inference_model_t* model);

// Direct weight access for debugging/analysis
const void* inference_get_tensor_ptr(const inference_model_t* model, const char* tensor_name, size_t* out_bytes);
```

### C++ HIP Implementation (`inference.cpp`, `kernels/*.hip`)

All implementation lives in C++ with HIP device code. The C API is a thin FFI wrapper.

## Phased Approach

### Phase 0: Foundation (already done or trivial)

- [x] FP4_FP6_FP8_Analysis.md written and committed to main branch
- [ ] Update README.md from "Initial Commit" to proper project description
- [ ] Create AGENTS.md on main with useful content from q4 branch
- [ ] Update PLAN.md's build section with new target structure

### Phase 1: GGUF Loading Infrastructure (~2 weeks)

**Goal**: mmap() a quantized GGUF file (including split files), load metadata into CPU-visible structures, register GPU-accessible mappings.

Deliverables:
1. `gguf_mmap.h` / `gguf_mmap.cpp` -- custom GGUF reader that uses `mmap()` instead of `fread()`. Parses GGUF magic, version, tensor count, key/value pairs, then maps each tensor's data region directly at its file offset.
2. Supports multi-file splits: detects `splits_count` and `split_no` keys from GGUF header, opens additional `.gguf` files and indexes their tensors alongside the primary file.
3. Registers all mapped regions with `hipHostRegister(mapped_ptr, size, hipHostRegisterMapped)` so the GPU can access them without copies. Prints device pointer for verification against the original mapping address.
4. Tensor type detection: reads GGUF type field (GGML_TYPE_MXFP4=39, etc.) and stores block_size/row_size info from ggml.h.

Key design decisions:
- Use existing llama.cpp GGUF parsing infrastructure as reference but strip out the `fread()` path entirely; replace with memory-mapped access via `::mmap(MAP_SHARED, ...)` on the .gguf file.
- For split models, each file gets its own `MAP_SHARED` mapping. The KV cache goes into `/dev/shm/*.kvcache.bin` (shared-memory anonymous mmap).
- Weights stay in their quantized format -- no dequantization happens during loading. Dequant happens inside kernels at compute time.

### Phase 2: ROCmFP4 Matmul Kernel (~2 weeks)

**Goal**: Implement the core FP4 matmul kernel that loads ROCmFP4 Codebook10 weights from GPU-accessible memory, dequantizes inline via DP4A + LDS LUT, and feeds into FP16 WMMA tensor cores.

Deliverables:
1. `fp4_matmul_kernel.hip` -- the primary compute kernel. Each work-group handles one output tile (e.g., 256x32 or similar). Inside the kernel:
   - Load FP4 nibbles from weight matrix (GPU-visible mmap'd memory)
   - Expand nibbles to int8 codes via `__builtin_amdgcn_perm` against a 32-byte constant codebook table
   - Accumulate dot products via `__builtin_amdgcn_sudot4` (int8xint8->int32 reduction over 8 values)
   - Scale accumulated sums by UE4M3 scale factors loaded from per-block scale bytes
   - Output is FP32 sum ready for add-on bias/activation

2. `fp4_matmul_wmma.hip` -- bridges the FP4 dequant path into WMMA tensor cores. Dequantized FP4 values go through an LDS LUT (nibble -> FP16 lookup), then feed into `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`. This leverages the hardware tensor core while keeping dequant overhead low via LDS caching of the LUT.

3. Verification harness: CPU-side reference implementation that runs on the same mapped buffer, produces expected FP32 results, and compares against GPU output with tolerance threshold.

Key design decisions:
- Weights in ROCmFP4 format stay in their native packed layout on the GPU. No conversion to FP16 happens before the kernel; dequant is inline at compute time.
- The DP4A path (`perm` + `sudot4`) is preferred over LDS-LUT-for-dequant because it's a single hardware instruction vs ~6 instructions for bit-manipulation scale decode.
- Block size: 32 values per block (matching llama.cpp convention). Scale per 16-element half-block (matching ROCmFP4 Codebook10 spec).
- No shared memory allocation needed -- all data lives in the mmap'd GGUF file regions already registered with `hipHostRegister`.

### Phase 3: KV Cache & Attention (~3 weeks)

**Goal**: Implement self-attention with FP4-computed Q/K/V projections, FP16-stored KV cache, and RoPE position encoding -- all zero-copy through unified memory mappings.

Deliverables:
1. `kv_cache.hip` / `kv_cache.cpp` -- manages the KV cache buffer allocated in `/dev/shm`. Provides token-level indexing into cache slots. Each layer has its own KV cache region within the shared file. The cache uses FP16 (BF16 possible) for Q/K/V values since these are small tensors that benefit from fast WMMA access but don't need quantization (they're transient, recomputed each step anyway).
2. `rope_embedding.hip` -- applies rotary positional embeddings to Q/K vectors. Computed on GPU side using FP16 math via WMMA intrinsics. Position IDs loaded from CPU-mapped buffer.
3. `attention_kernel.hip` -- full self-attention: matmul(Q, K^T) -> softmax -> matmul(softmax_output, V). All weight matrices in ROCmFP4, all activations in FP16/BF16. The softmax operates on a small matrix (seq_len x seq_len, typically <= 4096) so it fits in LDS without bank conflicts.
4. Layer normalization kernel (`layernorm.hip`) -- standard RMSNorm or LayerNorm computation on GPU, operating on mapped activation buffers.

Key design decisions:
- KV cache uses FP16 storage because it's small relative to weights and benefits from direct WMMA hardware acceleration. Quantizing KV cache would introduce error accumulation over the sequence length.
- RoPE is computed in-kernel during attention rather than pre-computed on CPU, saving a memory pass.
- All intermediate activations stay in unified mapped memory -- no staging buffers needed.

### Phase 4: Model Architecture Integration (~3 weeks)

**Goal**: Wire up the kernels into a complete forward pass for Qwen 3 dense transformer architecture (MLP layers with gated linear units, multi-head/multi-query attention with SwiGLU gating).

Deliverables:
1. `qwen_model.hip` / `qwen_model.cpp` -- models the Qwen 3 architecture: embedding layer, stack of decoder layers (each with attn + MLP), final LN, and LM head. Each layer calls the appropriate kernel chain.
2. Gated Linear Unit (GLU) implementation: SwiGLU is the activation pattern used by Qwen series models. The kernel computes SiLU(gate * W_up * x) element-wise after matrix multiply.
3. Embedding lookup (`embedding_lookup.hip`) -- indexed load from embedding table (FP16), scattered write back if needed for KV updates.
4. LM head projection uses Q8_0 quantization instead of FP4 because vocabulary sampling requires full dynamic range (values can span many orders of magnitude near the softmax boundary).

Key design decisions:
- Model parallelism is not needed on 128GB UMA -- the entire model fits in unified memory even at FP4 quantization. A single GPU thread-block grid handles everything.
- For a ~7B parameter model at ROCmFP4 (~0.56 BPW), weights consume approximately 4GB. At Q8_0 it would be ~7GB. With KV cache at 128K context length, total memory footprint stays well under 128GB.

### Phase 5: Tokenizer & C API Surface (~2 weeks)

**Goal**: Complete the user-facing API with tokenizer integration and all inference control functions declared in the design section above.

Deliverables:
1. `tokenizer.cpp` -- loads SentencePiece/BPE tokenizer from GGUF metadata, converts text to token IDs and vice versa. Integrates with llama.cpp's tokenizer infrastructure as reference but exposes only the minimal subset needed.
2. The complete C API as documented in "API Design Overview" above, implemented as thin wrappers around the C++ HIP implementation.
3. Error handling: every function returns error codes via out-parameters or integer return values (0 = success). CUDA/HIP errors propagated through wrapper functions that check `hipGetLastError()` after each device call.

### Phase 6: Benchmarking & Optimization (~2 weeks)

**Goal**: Validate performance against baselines, identify bottlenecks, optimize kernel launch parameters for Strix Halo's specific architecture.

Deliverables:
1. `bench_inference.py` -- Python orchestrator that measures tokens/sec at varying sequence lengths, batch sizes, and model sizes. Outputs JSON reports with per-layer timing breakdowns.
2. Profiling using ROCm SMI/hcc_profiler tools to verify kernel occupancy and memory bandwidth utilization on gfx1151.
3. Kernel tuning: experiment with different tile sizes (32x32 vs 64x32 vs 128x32), wavefront counts, and LDS LUT configurations to maximize throughput.
4. Comparison against baseline implementations: FP16-only matmul (no quantization), Q8_0 matmul, and reference llama.cpp CPU inference for accuracy validation.

### Phase 7: Integration Testing & Validation (~2 weeks)

**Goal**: End-to-end testing with real Qwen 3 model weights, correctness verification against reference llama.cpp outputs, stability testing under extended generation sequences.

Deliverables:
1. Test suite that loads a real Qwen3 GGUF model (starting with the smallest available size), runs token-by-token comparison against llama.cpp reference output, verifies attention weights match within floating-point tolerance.
2. Memory leak detection: valgrind / ROCm memory tracking over thousands of decode steps.
3. Extended generation tests: 4K and 8K context sequences to validate KV cache integrity and RoPE position handling across full range.
4. Multi-quantization path testing: verify both ROCmFP4 and Q8_0 paths produce correct results on models with mixed precision layouts.

## Timeline Summary

| Phase | Weeks | Key Deliverable |
|-------|-------|-----------------|
| 0 | 1 | Foundation docs committed |
| 1 | 2 | GGUF mmap loader working |
| 2 | 2 | ROCmFP4 matmul kernel verified |
| 3 | 3 | Attention + KV cache functional |
| 4 | 3 | Complete forward pass for Qwen3 |
| 5 | 2 | C API surface complete |
| 6 | 2 | Benchmarks and tuning done |
| 7 | 2 | End-to-end validation passing |

Total: ~17 weeks from start to production-ready inference engine.

## Makefile Target Structure (Future)

```makefile
# Existing targets (unchanged)
all: mandelbrot wmma_matmul wmma_matmul_fp4 wmma_matmul_fp4_native wmma_bench

# New inference engine targets
inference: libinference.a inference_demo              # static library + demo binary
libinference.so:                                     # shared library for XS binding

# Kernel compilation rules (same pattern as existing)
%.co: %.hip     $(HIPCC) $(HIPFLAGS) -c $< -o $@

# Host compilation for inference engine
CXXFLAGS_INF := -O3 -fPIC -I$(ROCM_PATH)/include -Iggml/include -D__HIP_PLATFORM_AMD__
```

## Notes on llama.cpp Integration Path

The inference engine described here is standalone but designed to coexist with or eventually feed back into the llama.cpp codebase. The key integration points are:

- GGUF type handling: ROCmFP4 is ftype 39 in llama.cpp's `llama_ftype` enum and GGML_TYPE_MXFP4=39 in ggml.h. Our kernels target this exact format. The analysis doc notes that llama.cpp's CUDA backend already implements ROCmFP4 via DP4A -- our work extends this to Strix Halo specifically, replacing the CUDA-specific paths with HIP equivalents using identical intrinsic calls where possible (`__builtin_amdgcn_perm`, `__builtin_amdgcn_sudot4`).

- Weight layout compatibility: our FP4 matmul kernel reads the exact same block structure (16 bytes data + 2 bytes scale per 32 values) that llama.cpp uses. This means models quantized with llama.cpp's `llama-quant` tool can be loaded directly by our engine without any conversion step.

- Future path: once the standalone engine is validated, the kernels can be integrated back into the llama.cpp CUDA backend as a gfx1151-specific implementation layer, reusing the existing GGUF loading infrastructure while providing optimized zero-copy paths for Strix Halo's UMA architecture.
