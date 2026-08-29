# Agents

This is a proof-of-concept integrating two features into [llama.cpp](https://github.com/ggml-org/llama.cpp):

1. **HIP/ROCm mmap support** — memory-mapped model loading on AMD GPUs via HIP runtime
2. **ROCmFP4 quantization** — native FP4 quantization with FP16 activations on ROCm (ftype 107/108)

Similar in spirit to [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX), which adds mmap() support for n-gram MoE Qwen3.8-Flash models in vLLM, but targeting llama.cpp's GGUF format with ROCm hardware acceleration.

See `PLAN.md` and `TODO.md` for current status and roadmap.
