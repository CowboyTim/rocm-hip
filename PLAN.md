# Project Plan & Analysis: rocm-hip (Strix Halo HIP POC)

## What It Is

A HIP/ROCm proof-of-concept for **AMD Strix Halo (gfx1151, RDNA3.5 APU)** proving out two ideas:

1. `mmap()` file-backed memory as zero-copy GPU data via `hipHostRegister`
2. Low-precision matmul (FP16 WMMA + ROCmFP4 FP4) using RDNA3.5 hardware intrinsics

~35 commits; a second branch (`q4_0_rocmfp4_q8_1`) extends this into full llama.cpp integration work.

## Core Demo Pattern (repeated in every binary)

1. Create buffer in `/dev/shm/*.bin` → `ftruncate` → `mmap(MAP_SHARED)` (file-backed "unified memory")
2. `hipHostRegister(ptr, size, hipHostRegisterMapped/Default)` to pin pages so the GPU can walk them; `hipHostGetDevicePointer` for the device address (identical on UMA APU — printed for verification)
3. Launch kernel with **no `hipMemcpy` anywhere** — GPU reads/writes CPU-visible file-backed RAM directly
4. Kernels are pre-compiled to `.co` objects (`hipcc --genco`, `-mno-wavefrontsize64`) and loaded at runtime by `mmap`ing the `.co` file itself + `hipModuleLoadData`. Host `.cpp` is compiled with plain clang++ (no offload); split build rule in Makefile.

## Components on `main`

| Target | Files | What it does |
|---|---|---|
| `mandelbrot` | main.cpp, kernels.hip | Flagship mmap demo: 800×600 escape-time renderer writes RGBA pixels straight into a `/dev/shm` mapping; verifies known set points (cardioid cusp, seahorse valley, filaments); dumps PPM |
| `(halo)` | mmap.cpp | Original minimal POC: `+1.0f` increment kernel over 100MB of mapped memory; carries the verbose "Strix Halo halo" design comments |
| `wmma_matmul` | wmma_kernel.hip, wmma_main.cpp | 16×16 FP16 matmul via `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`; documents RDNA3 OPSEL=false unpacked-fragment lane/row mapping in detail; CPU-verifies C=(i+1)(j+1)·16 |
| `wmma_matmul_fp4` | wmma_kernel_fp4.hip, wmma_main_fp4.cpp | **ROCmFP4 Codebook10** format (E2M1-derived: 0,±1..±4,±6,±8,±10, tuned on Qwen3 tensors): dequant nibble→FP16 through a 32-byte LDS LUT, then into the FP16 WMMA tensor core — the best gfx1151 path |
| `wmma_matmul_fp4_native` | *_fp4_native.* | FP4×FP4→FP32 32×16 tile; references `__builtin_amdgcn_wmma_f32_32x16x128_f4` but that's gfx1250-only, so ships a scalar fallback for gfx1151 |
| `wmma_bench` + run_bench.py | 5 bench kernels + host + Python orchestrator | Head-to-head FP4 matmul at configurable M/N/K (default 1K³): scalar-LUT vs DP4A-intrinsic vs native-fallback vs FP4→FP16-WMMA+LDS-LUT. CPU reference in-mapping, mismatch%/GFLOPS table, JSON export. DP4A variant expands codebook with `__builtin_amdgcn_perm` (4 const tables) and dot-products with `__builtin_amdgcn_sudot4` — pure integer hardware path |

## Analysis Doc & Second Branch

- **FP4_FP6_FP8_Analysis.md** — deep llama.cpp source study: why ROCmFP4 beats native FP4 on gfx1151 (`perm` is HW on all RDNA; native FP4 cvt needs gfx950), why FP6 can't use DP4A (6-bit codes > 4-bit perm mask), FP8 exists only as UE4M3 scale encoding, ~cycle-count estimates per dequant path.
- **Branch `q4_0_rocmfp4_q8_1`** — the "real" project: llama.cpp submodule (pinned to a fork carrying ROCmFP4 ftype 107/108 integration, 53 commits ahead of upstream at the time), `INTEGRATION_STATUS.md` (incl. an analysis concluding the local Qwen3.8-27B build is a dense Qwen3.5-lineage model, not the official Flash-DGX MoE+n-gram one), `build_llama.cpp.sh` for Strix Halo + rocWMMA headers, mmap zero-copy GGUF loading, 13-week roadmap toward MFMA/WMMA/FP4 + KV-cache fast path.

## Git State & Observations

- Remote `origin/main`; working tree clean. History arc: trivial test → error-handling fixes → Mandelbrot-over-UMa → WMMA intrinsics exploration → FP4 benchmark suite → llama.cpp submodule added Jul 1 then **removed** from main ("no llama.cpp"), leaving `.gitmodules` as an empty file and the gitlink deleted — that work lives only on the q4 branch.
- A prior commit also deleted `.COMMIT_NOTES.md`, `PLAN.md`/`TODO.md`, `rocmfp4-llama/`, old docs (consolidated into INTEGRATION_STATUS.md on q4).

### Issues Worth Fixing

1. README still says "Initial Commit"; AGENTS.md on main is a stub (the useful version is on q4)
2. `wmma_main_fp4_native.cpp:47` has a stale prototype `wmma_matmul_fp4_builtin` that matches no symbol — dead code (dispatch uses string names), but misleading
3. The 5th bench kernel (`wmma_bench_fp4_dp4a_int8`) is compiled by Makefile/run_bench.py but **never executed** by `wmma_bench_main.cpp` (only 4 methods run)
4. DP4A dequant in `wmma_bench_fp4_dp4a_hardware.hip:37-61`: the second `perm` selects between negative tables based on bit 2, but Codebook10's sign is bit 3 of the nibble — positive codes come back as negatives; comments in-file even flag the confusion ("we need two separate perms…"). Scalar/negative data works around it via clamping; this method likely fails verification for real signed data
5. Codebooks + `HIP_CHECK` macros are duplicated across ~9 files instead of shared headers

## Build

ROCm at `/opt/rocm` (hipcc available). Makefile targets: `mandelbrot wmma_matmul wmma_matmul_fp4 wmma_matmul_fp4_native wmma_bench`, all pinned to `--offload-arch=gfx1151`. Nothing built yet (no `.co` artifacts in tree).
