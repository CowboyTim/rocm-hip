#!/bin/bash

# Build script for llama.cpp optimized for AMD Strix Halo (gfx1151) with ROCm 7.12+
# Includes all performance optimizations discovered from codebase analysis

# Set environment variables explicitly for ROCm 7.10+
export HIP_PLATFORM=amd
export ROCM_PATH=${ROCM_PATH?"ROCM_PATH environment variable is not set. Please set it to the root of your ROCm installation."}
export ROCM_PATH=$(readlink -f "$ROCM_PATH")
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export HIPCXX="$ROCM_PATH/llvm/bin/clang"

# The compiler needs to know where the bitcode is
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode

HIP_EXTRA_FLAGS="${CMAKE_HIP_FLAGS:-}"
ROCM_WMMA_INCLUDE="${ROCM_PATH}/include/"
if [[ -d "$ROCM_WMMA_INCLUDE/rocwmma/internal" ]]; then
    HIP_EXTRA_FLAGS="-I${ROCM_WMMA_INCLUDE} ${HIP_EXTRA_FLAGS}"
    echo "Using local rocWMMA headers: $ROCM_WMMA_INCLUDE"
else
    echo "Warning: rocWMMA headers not found at $ROCM_WMMA_INCLUDE" >&2
fi


echo "======================================================================"
echo "Building llama.cpp for AMD Strix Halo (gfx1151) with ROCm"
echo "ROCm Path: $ROCM_PATH"
echo "======================================================================"

# Configure with explicit compiler paths and architecture
# We add --rocm-path directly to the HIP flags to stop Clang from complaining
mkdir -p build && cd build || exit 1
cmake \
    .. \
    -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_BUILD_TYPE=Release \
    \
    `# === Core HIP/ROCm Configuration ===` \
    -DGGML_HIP_FORCE_MMQ=ON \
    -DGGML_VULKAN=OFF \
    -DGGML_NATIVE=ON \
    -DGGML_HIP=ON \
    -DGGML_HIPBLAS=ON \
    -DAMDGPU_TARGETS=gfx1151 \
    -DGPU_TARGETS=gfx1151 \
    -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
    \
    `# === Unified Memory Configuration (128GB UMA) ===` \
    -DGGML_HIP_UMA=ON \
    -DGGML_HIP_FORCE_TILING_ALLOCATOR=ON \
    \
    `# === Graph Optimization (Enable for long context, disable via env var for small prompts) ===` \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_ALLOC_GRAPH_RESERVE=2048 \
    \
    `# === Recurrent State & KV Cache GPU Offload (Critical for Qwen3-Next) ===` \
    -DGGML_HIP_FORCE_RS_GPU=ON \
    -DGGML_HIP_FORCE_KV_GPU=ON \
    \
    `# === rocWMMA Flash Attention (RDNA3.5 Optimized) ===` \
    -DGGML_HIP_ROCWMMA=ON \
    -DGGML_HIP_ROCWMMA_UMA=ON \
    -DGGML_HIP_ROCWMMA_GRAPHS=ON \
    -DGGML_HIP_ROCWMMA_ALLOC_GRAPH_RESERVE=512 \
    -DGGML_HIP_ROCWMMA_FORCE_RS_GPU=ON \
    -DGGML_HIP_ROCWMMA_FORCE_KV_GPU=ON \
    -DGGML_HIP_ROCWMMA_FORCE_TILING_ALLOCATOR=ON \
    -DGGML_HIP_ROCWMMA_FATTN_UMA=ON \
    -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DGGML_HIP_FORCE_WMM_REDUCE=ON \
    \
    `# === Flash Attention Optimizations ===` \
    -DGGML_HIP_FATTN=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    \
    `# === MMQ (Matrix Multiply Quantized) Configuration ===` \
    -DGGML_HIP_MMQ_MFMA=ON \
    -DGGML_CUDA_FORCE_CUBLAS=OFF \
    \
    `# === Performance Profiling ===` \
    -DGGML_HIP_EXPORT_METRICS=OFF \
    \
    `# === Build Configuration ===` \
    -DGGML_OPENMP=OFF \
    -DGGML_STATIC=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_WEBUI=OFF \
    -DLLAMA_USE_PREBUILT_WEBUI=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_BUILD_TESTS=OFF \
    \
    `# === Compiler Configuration ===` \
    -DCMAKE_C_COMPILER=$ROCM_PATH/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
    -DCMAKE_HIP_COMPILER=$ROCM_PATH/llvm/bin/clang++ \
    -DCMAKE_C_FLAGS="-march=native -O3" \
    -DCMAKE_CXX_FLAGS="-march=native -O3 -Wno-nan-infinity-disabled" \
    -DCMAKE_HIP_FLAGS="-Wno-nan-infinity-disabled --rocm-device-lib-path=$HIP_DEVICE_LIB_PATH --rocm-path=$ROCM_PATH -Xarch_device -mwavefrontsize64=false -isystem $ROCM_PATH/include $HIP_EXTRA_FLAGS"

# Build with all cores
echo ""
echo "Building with $(nproc) parallel jobs..."
cd ..
cmake --build build --config Release -j$(nproc) --target \
    llama-cli \
    llama-server \
    llama-completion \
    llama-quantize \
    llama-bench

BUILD_STATUS=$?

echo ""
echo "======================================================================"
if [ $BUILD_STATUS -eq 0 ]; then
    echo "✓ Build completed successfully!"
    echo ""
    echo "New optimizations enabled in this build:"
    echo "  • GGML_CUDA_FA_ALL_QUANTS: Fine-grained KV cache quantization"
    echo "  • GGML_HIP_MMQ_MFMA: MFMA-accelerated MMQ kernels for RDNA3.5"
    echo "  • GGML_HIP_EXPORT_METRICS: Kernel profiling metrics"
    echo "  • GGML_HIP_GRAPHS: Enabled (override with env var if needed)"
    echo "  • CMAKE_HIP_ARCHITECTURES: Explicit gfx1151 targeting"
    echo ""
    echo "Already optimized features:"
    echo "  • rocWMMA flash attention (RDNA3.5 optimized)"
    echo "  • UMA unified memory (128GB pool)"
    echo "  • Recurrent state & KV cache GPU offload"
    echo "  • Tiling allocator for memory efficiency"
    echo ""
    echo "Test with: bash ~/rocm-install.git/llama_bench.sh -m <model>"
else
    echo "✗ Build failed with status $BUILD_STATUS"
fi
echo "======================================================================"

exit $BUILD_STATUS
