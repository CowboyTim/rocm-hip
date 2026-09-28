# ROCmFP4 Strix Halo HIP POC

HIP/ROCm proof-of-concept for **AMD Strix Halo (gfx1151, RDNA3.5 APU)** proving out two ideas:

1. `mmap()` file-backed memory as zero-copy GPU data via `hipHostRegister`
2. Low-precision matmul (FP16 WMMA + ROCmFP4 FP4) using RDNA3.5 hardware intrinsics

~35 commits; a second branch (`q4_0_rocmfp4_q8_1`) extends this into full llama.cpp integration work.

## Quick Links

- [Project Plan](PLAN.md) — architecture, roadmap, phase breakdown
- [FP4/FP6/FP8 Analysis](FP4_FP6_FP8_Analysis.md) — why ROCmFP4 beats native FP4 on gfx1151
- [Agents](AGENTS.md) — overview of the llama.cpp integration effort

## Components

| Target | Description |
|---|---|
| `mandelbrot` | 800x600 escape-time renderer writing RGBA pixels straight into `/dev/shm` mapping; verifies known set points |
| `halo` (mmap.cpp) | Minimal POC: `+1.0f` increment kernel over 100MB of mapped memory |
| `wmma_matmul` | 16x16 FP16 matmul via `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`; CPU-verifies C=(i+1)(j+1)*16 |
| `wmma_matmul_fp4` | ROCmFP4 Codebook10 format: dequant nibble->FP16 through 32-byte LDS LUT, then into FP16 WMMA tensor core |
| `wmma_matmul_fp4_native` | FP4xFP4->FP32 32x16 tile; ships scalar fallback for gfx1151 (intrinsic is gfx1250-only) |
| `wmma_bench` + run_bench.py | Head-to-head FP4 matmul bench: scalar-LUT vs DP4A-intrinsic vs native-fallback vs FP4->FP16-WMMA+LDS-LUT |

## Build

ROCm at `/opt/rocm` (hipcc available). Makefile targets:

```bash
make all          # build all demos
make mandelbrot   # single target
make clean        # remove artifacts
```

All kernels are compiled with `-mno-wavefrontsize64 --offload-arch=gfx1151`.
