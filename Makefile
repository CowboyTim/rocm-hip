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
# wmma_bench_fp4_native.hip: scalar FP4 dequant + FP32 accumulate (works on gfx1151)
# wmma_bench_fp4_native_gfx1250.hip: native FP4 WMMA using __builtin_amdgcn_wmma_f32_32x16x128_f4 (requires gfx1250+)
BENCH_TARGET = wmma_bench
BENCH_HOST = wmma_bench_main.o
BENCH_KERNS = wmma_bench_fp4_dp4a.hip wmma_bench_fp4_native.hip
BENCH_NATIVE_GFX1250 = wmma_bench_fp4_native_gfx1250.o
BENCH_NATIVE_GFX1250_SRC = wmma_bench_fp4_native_gfx1250.hip

# --- Build Rules ---
all: $(TARGET) $(WMMA_TARGET) $(BENCH_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_NATIVE_GFX1250)

$(TARGET): $(OBJ) $(KERNEL_CO)
	$(CLANG) $(OBJ) $(LDFLAGS) -o $@

$(WMMA_TARGET): $(WMMA_OBJ) $(WMMA_CO)
	$(CLANG) $(WMMA_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_TARGET): $(WMMA_FP4_OBJ) $(WMMA_FP4_CO)
	$(CLANG) $(WMMA_FP4_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_NATIVE_TARGET): $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO)
	$(CLANG) $(WMMA_FP4_NATIVE_OBJ) $(LDFLAGS) -o $@

# Pre-compiled kernel objects (.hip -> .co)
%.co: %.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Host C++ files (.cpp -> .o) -- compiled without device offloading
%.o: %.cpp
	$(CLANG) $(CXXFLAGS) -c $< -o $@

# Device HIP kernel files (.hip -> .o) -- compiled with offloading
wmma_bench_fp4_dp4a.o: wmma_bench_fp4_dp4a.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

wmma_bench_fp4_native.o: wmma_bench_fp4_native.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Native FP4 WMMA kernel for gfx1250+ (RDNA4 / AMD AI engines)
# Uses __builtin_amdgcn_wmma_f32_32x16x128_f4 intrinsic
# REQUIRES: --offload-arch=gfx1250 or later
native_gfx1250: $(BENCH_NATIVE_GFX1250)
	@echo "Built native FP4 WMMA kernel for gfx1250+"

$(BENCH_NATIVE_GFX1250): $(BENCH_NATIVE_GFX1250_SRC)
	$(HIPCC) -Xarch_device -mno-wavefrontsize64 -isystem $(ROCM_PATH)/include --rocm-path=$(ROCM_PATH) -O3 --offload-arch=gfx1250 --genco -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__ -c $< -o $@

# Build the benchmark for gfx1151 (FP4 native uses scalar fallback)
$(BENCH_TARGET): $(BENCH_HOST) $(BENCH_KERNS)
	$(HIPCC) -isystem $(ROCM_PATH)/include --rocm-path=$(ROCM_PATH)  -o $@ $(BENCH_HOST) $(BENCH_KERNS) $(LDFLAGS) -mno-wavefrontsize64

# Build the benchmark for gfx1250 (FP4 native uses hardware intrinsic)
$(BENCH_TARGET)_gfx1250: $(BENCH_HOST) $(BENCH_KERNS) $(BENCH_NATIVE_GFX1250)
	$(HIPCC) -isystem $(ROCM_PATH)/include --rocm-path=$(ROCM_PATH) -DROCWMMA_BUILD_GFX1250 --offload-arch=gfx1250 --genco -o $@ $(BENCH_HOST) $(BENCH_KERNS) $(BENCH_NATIVE_GFX1250) $(LDFLAGS) -mno-wavefrontsize64

# Build all demos and benchmark
both: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET)

clean:
	rm -f $(TARGET) $(OBJ) $(KERNEL_CO) $(WMMA_TARGET) $(WMMA_OBJ) $(WMMA_CO) $(WMMA_FP4_TARGET) $(WMMA_FP4_OBJ) $(WMMA_FP4_CO) $(WMMA_FP4_NATIVE_TARGET) $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO) $(BENCH_TARGET) $(BENCH_HOST)

.PHONY: all clean both
