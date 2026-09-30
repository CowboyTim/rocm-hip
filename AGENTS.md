# Agents

This is a proof-of-concept integrating two features into [llama.cpp](https://github.com/ggml-org/llama.cpp):

1. **HIP/ROCm mmap support** — memory-mapped model loading on AMD GPUs via HIP runtime
2. **ROCmFP4 quantization** — native FP4 quantization with FP16 activations on ROCm (ftype 107/108)

Similar in spirit to [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), which adds mmap() support for n-gram MoE Qwen3.8-Flash models in vLLM, but targeting llama.cpp's GGUF format with ROCm hardware acceleration.

See `PLAN.md` for current status and roadmap.

## Branches

All work happens on **`main`**.

- **`main`** — everything: POC demos (Mandelbrot, WMMA matmul, ROCmFP4 benchmarks), Phase 1–2 infrastructure (`gguf_mmap` zero-copy GGUF loader, verified FP4 matmul kernels + `fp4_verify` harness), and the inference-engine roadmap (Phases 3–7). Standalone zero-copy kernels proving the core concepts; no llama.cpp submodule.
- **`q4_0_rocmfp4_q8_1`** — archived reference only: an earlier forked-llama.cpp snapshot (ROCmFP4 ftype 107/108, zero-copy GGUF loading, INTEGRATION_STATUS.md analysis incl. the Qwen3.8-Flash-DGX comparison). No new work goes here; cherry-pick anything still useful into `main`.

## Core Pattern

All binaries use the same zero-copy approach:

1. Create buffer in `/dev/shm/*.bin` -> `ftruncate` -> `mmap(MAP_SHARED)`
2. `hipHostRegister(ptr, size, hipHostRegisterMapped/Default)` to pin pages
3. Launch kernel with **no `hipMemcpy`** — GPU reads/writes CPU-visible file-backed RAM directly
4. Kernels pre-compiled to `.co` objects (`hipcc --genco`) loaded at runtime via `hipModuleLoadData`