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

# WMMA matmul: uses hipLaunchKernelGGL with mmap zero-copy
WMMA_SRC  = wmma_kernel.hip
WMMA_CO   = wmma_kernel.co
WMMA_OBJ  = wmma_main.o
WMMA_TARGET = wmma_matmul

# --- Build Rules ---
all: $(TARGET)

$(TARGET): $(OBJ) $(KERNEL_CO)
	$(CLANG) $(OBJ) $(LDFLAGS) -o $@

$(WMMA_TARGET): $(WMMA_OBJ) $(WMMA_CO)
	$(CLANG) $(WMMA_OBJ) $(LDFLAGS) -o $@

%.co: %.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

%.o: %.cpp
	$(CLANG) $(CXXFLAGS) -c $< -o $@

# Build both demos
both: $(TARGET) $(WMMA_TARGET)

clean:
	rm -f $(TARGET) $(OBJ) $(KERNEL_CO) $(WMMA_TARGET) $(WMMA_OBJ) $(WMMA_CO)

.PHONY: all clean both
