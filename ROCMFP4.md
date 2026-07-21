# ROCmFP4 Integration with Upstream Master

## Overview

This document describes the integration of ROCmFP4 FP16 activation support into the upstream `llama.cpp` master branch.

## Branch Information

- **Branch**: `rocmfp4-strix-upstream-merged`
- **Base**: `upstream/master` (commit `5735e10c4`)
- **ROCmFP4 Source**: `origin/rocmfp4-strix-original`

## Full Documentation

See `llama.cpp/ROCMFP4.md` for complete integration details, including:
- All merged commits
- Files modified and added
- Build instructions
- Implementation details
- Comparison with rocmfp4-llama submodule

## Quick Reference

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

## Submodules

- **llama.cpp/** - Upstream master with ROCmFP4 integration (branch: `rocmfp4-strix-upstream-merged`)
- **rocmfp4-llama/** - ROCmFP4 fork with MTP support (branch: `nemotron-mtp-rocmfp4-strix`)

## Author

Based on ROCmFP4 implementation by Tim Aerts (aardbeiplantje)

## Date

Created: July 2025
