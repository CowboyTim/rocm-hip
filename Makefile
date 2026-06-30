# --- Configuration ---
ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc
CLANG     := $(ROCM_PATH)/lib/llvm/bin/clang++

ARCH      := gfx1151

CXXFLAGS  := -O3 -Wall -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__
HIPFLAGS  := -O3 --offload-arch=$(ARCH) --genco

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

# --- Build Rules ---
all: $(TARGET)

$(TARGET): $(OBJ) $(KERNEL_CO)
	$(CLANG) $(OBJ) $(LDFLAGS) -o $@

$(WMMA_TARGET): $(WMMA_OBJ) $(WMMA_CO)
	$(CLANG) $(WMMA_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_TARGET): $(WMMA_FP4_OBJ) $(WMMA_FP4_CO)
	$(CLANG) $(WMMA_FP4_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_NATIVE_TARGET): $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO)
	$(CLANG) $(WMMA_FP4_NATIVE_OBJ) $(LDFLAGS) -o $@

%.co: %.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

%.o: %.cpp
	$(HIPCC) -O3 -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__ -c $< -o $@

# Build the benchmark
$(BENCH_TARGET): wmma_bench_main.o
	$(HIPCC) -o $@ wmma_bench_main.o $(LDFLAGS)

# Build all demos and benchmark
both: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET)

clean:
	rm -f $(TARGET) $(OBJ) $(KERNEL_CO) $(WMMA_TARGET) $(WMMA_OBJ) $(WMMA_CO) $(WMMA_FP4_TARGET) $(WMMA_FP4_OBJ) $(WMMA_FP4_CO) $(WMMA_FP4_NATIVE_TARGET) $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO) $(BENCH_TARGET) wmma_bench_main.o wmma_bench_kernels.* $(BENCH_TARGET) $(BENCH_OBJ) $(BENCH_CO)

.PHONY: all clean both
