# rocm-hip: zero-copy LLM inference on AMD Strix Halo

A standalone inference engine for **AMD Strix Halo (gfx1151, RDNA3.5 APU, 128 GB unified memory)**: GGUF models
are memory-mapped and read in place by HIP kernels (no copies, no repacking), with ROCmFP4 / K-quant int8 dot
products, WMMA prefill, a KV cache of shareable files and MTP speculative decoding. A C library
(`libinference.so`) and a Perl XS module (`ROCm::Inference`).

| Model | Generation | Prefill |
|---|---|---|
| Qwen3.5-4B (ROCmFP4) | 64 tok/s | 1,826 tok/s |
| Qwen3.6-35B-A3B (ROCmFP4) | 65.7 tok/s | 1,193 tok/s |
| Qwen3.8-Flash-Next (177B, ROCmFP4_FAST) | 38 tok/s, ~50 with its MTP head | 434 tok/s |

All development happens on **`main`**. The `q4_0_rocmfp4_q8_1` branch is an archived reference (an earlier
forked-llama.cpp snapshot).

## Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) — how the engine is built
- [PLAN.md](PLAN.md) — plan, status, measurements and history (the current next step is at "▶ NEXT STEP")
- [FP4_FP6_FP8_Analysis.md](FP4_FP6_FP8_Analysis.md) — the low-precision formats on gfx1151
- [AGENTS.md](AGENTS.md) — working conventions
- [perl/ROCm-Inference/README](perl/ROCm-Inference/README) — the Perl module

## Build and run

ROCm at `/opt/rocm` (hipcc); kernels are built for `--offload-arch=gfx1151`, wave32.

```bash
make -j            # the library (build/libinference.so.1), qwen_demo, tests and demos
./build/qwen_demo /path/model.gguf "The capital of France is" 64 --temp 0
./build/qwen_demo /path/Qwen3.8-Flash-Next-...-00001-of-00003.gguf "Write a sort in C." 256 \
    --draft /path/mtp-Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST.gguf
make install PREFIX=/usr/local   # lib, headers, rocm-inference.pc
```

From C: `inference.h` (`inference_load_model`, `inference_kv_cache_init`, `inference_generate_stream`, ...;
`tests/capi_smoke.c` is a minimal example). From Perl: `cd perl/ROCm-Inference && perl Makefile.PL && make test`.

The tests compare with llama.cpp as a reference oracle (`make ref_dump`, CPU build in `llama.cpp/build-cpu`);
see ARCHITECTURE.md section 9.

## Proof-of-concept demos

The repository started as a proof of concept of its two core ideas, `mmap()`ed memory as zero-copy GPU data
(`hipHostRegister`) and low-precision matmul with RDNA3.5 intrinsics. Of those demos, `wmma_matmul` (a 16x16
FP16 matmul via `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32`, CPU-verified) is still built; the others (a
Mandelbrot renderer into a `/dev/shm` mapping, FP4 WMMA variants and an FP4 matmul bench) were retired in commit
4b4e082 and live in the git history. `src/wmma_main_fp4*.cpp` are their remaining host halves.
