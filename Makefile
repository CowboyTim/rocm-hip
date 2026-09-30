# --- Configuration ---
ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc
CLANG     := $(ROCM_PATH)/lib/llvm/bin/clang++

ARCH      := gfx1151

# Host C++ flags (no offloading)
CXXFLAGS  := -O3 -Wall -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__

# Device/HIP kernel flags (with offloading)
HIPFLAGS  := -Xarch_device -mno-wavefrontsize64 -isystem $(ROCM_PATH)/include --rocm-path=$(ROCM_PATH) -O3 --offload-arch=$(ARCH) --genco -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__

LDFLAGS   := -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,$(ROCM_PATH)/lib

# --- Demos ---
# Mandelbrot: loads pre-compiled .co module with hipModuleLoadData
TARGET    := mandelbrot
KERNEL_SRC = kernels.hip
KERNEL_CO  = kernels.co
OBJ        = main.o

# WMMA matmul (FP16): uses hipModuleLoadData + mmap zero-copy
WMMA_SRC  = wmma_kernel.hip
WMMA_CO   = wmma_kernel.co
WMMA_OBJ  = wmma_main.o
WMMA_TARGET = wmma_matmul

# WMMA matmul (FP4 ROCmFP4): FP4 dequant via DP4A perm + WMMA FP16
WMMA_FP4_SRC  = wmma_kernel_fp4.hip
WMMA_FP4_CO   = wmma_kernel_fp4.co
WMMA_FP4_OBJ  = wmma_main_fp4.o
WMMA_FP4_TARGET = wmma_matmul_fp4

# WMMA matmul (FP4 native): uses __builtin_amdgcn_wmma_f32_32x16x128_f4
WMMA_FP4_NATIVE_SRC  = wmma_kernel_fp4_native.hip
WMMA_FP4_NATIVE_CO   = wmma_kernel_fp4_native.co
WMMA_FP4_NATIVE_OBJ  = wmma_main_fp4_native.o
WMMA_FP4_NATIVE_TARGET = wmma_matmul_fp4_native

# FP4 Matmul Benchmark
BENCH_TARGET = wmma_bench
BENCH_HOST = wmma_bench_main.o
BENCH_LUT_CO = wmma_bench_fp4_lut.co
BENCH_DP4A_INT8_CO = wmma_bench_fp4_dp4a_int8.co

# --- Phase 1: GGUF mmap reader ---
GGUF_MMAP_SRC = gguf_mmap.cpp
DEMO_GGUF     = demo_gguf

# Phase 2: ROCmFP4 Codebook10 matmul verification harness (CPU reference vs GPU kernels)
FP4_VERIFY   = fp4_verify

# --- Build Rules ---
all: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET) $(DEMO_GGUF) $(FP4_VERIFY)

$(TARGET): $(OBJ) $(KERNEL_CO)
	$(CLANG) $(OBJ) $(LDFLAGS) -o $@

$(WMMA_TARGET): $(WMMA_OBJ) $(WMMA_CO)
	$(CLANG) $(WMMA_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_TARGET): $(WMMA_FP4_OBJ) $(WMMA_CO)
	$(CLANG) $(WMMA_FP4_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_NATIVE_TARGET): $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO)
	$(CLANG) $(WMMA_FP4_NATIVE_OBJ) $(LDFLAGS) -o $@

# Pre-compiled kernel objects (.hip -> .co)
%.co: %.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Host C++ files (.cpp -> .o) -- compiled without device offloading
%.o: %.cpp
	$(CLANG) $(CXXFLAGS) -c $< -o $@

# Benchmark kernels -> .co files (same approach as wmma_matmul)
wmma_bench_fp4_dp4a.co: wmma_bench_fp4_dp4a.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

wmma_bench_fp4_native.co: wmma_bench_fp4_native.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# DP4A hardware intrinsics kernel (uses __builtin_amdgcn_perm + sudot4)
wmma_bench_fp4_dp4a_hardware.co: wmma_bench_fp4_dp4a_hardware.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Build the benchmark binary (links host .o, loads .co at runtime)
$(BENCH_TARGET): $(BENCH_HOST) wmma_bench_fp4_dp4a.co wmma_bench_fp4_dp4a_hardware.co wmma_bench_fp4_native.co $(BENCH_LUT_CO) $(BENCH_DP4A_INT8_CO)
	$(CLANG) $(BENCH_HOST) $(LDFLAGS) -o $@

# Phase 1 demo: gguf_mmap reader linked into a single binary
$(DEMO_GGUF): $(GGUF_MMAP_SRC) demo_gguf.cpp gguf_mmap.h
	$(CLANG) -O3 -Wall -I. -D__HIP_PLATFORM_AMD__ \
		$(GGUF_MMAP_SRC) demo_gguf.cpp -o $@

# Phase 2 verify: host harness loads both .co files at runtime, compares against CPU reference
$(FP4_VERIFY): fp4_verify.cpp fp4_common.h fp4_matmul_kernel.co fp4_matmul_wmma.co
	$(CLANG) $(CXXFLAGS) -I. fp4_verify.cpp $(LDFLAGS) -o $@

# Build all demos and benchmark
both: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET)

clean:
	rm -f $(TARGET) $(OBJ) $(KERNEL_CO) $(WMMA_TARGET) $(WMMA_OBJ) $(WMMA_CO) $(WMMA_FP4_TARGET) $(WMMA_FP4_OBJ) $(WMMA_FP4_CO) $(WMMA_FP4_NATIVE_TARGET) $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO) $(BENCH_TARGET) $(BENCH_HOST) wmma_bench_fp4_dp4a.co wmma_bench_fp4_native.co wmma_bench_fp4_dp4a_hardware.co $(BENCH_LUT_CO) $(BENCH_DP4A_INT8_CO) $(DEMO_GGUF) $(FP4_VERIFY) *.co

.PHONY: all clean both
