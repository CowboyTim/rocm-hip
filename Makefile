# --- Configuration ---
# Update this path to your "The Rock" distribution location
ROCM_PATH ?= therock-dist-linux-gfx1151-7.12.0a20260203
HIPCC     := $(ROCM_PATH)/bin/hipcc
CLANG     := $(ROCM_PATH)/lib/llvm/bin/clang++

# Architecture for Strix Halo
ARCH      := gfx1151

# Compilation Flags
CXXFLAGS  := -O3 -Wall -x hip -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__
HIPFLAGS  := -O3 --offload-arch=$(ARCH) --genco

# Linker Flags
# We need amdhip64 and hsa-runtime64 for the IOMMU/Memory features
LDFLAGS   := -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,$(ROCM_PATH)/lib

# Target Binary
TARGET    := halo_test

# --- Files ---
# Kernels (GPU code)
GPU_SRC   := kernels.hip
GPU_CO    := kernels.co

# Host (CPU code)
HOST_SRC  := main.cpp
HOST_OBJ  := main.o

# --- Build Rules ---
all: $(TARGET) $(GPU_CO)

# 1. Compile Kernels separately with hipcc
%.co: %.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# 2. Compile Host code with standard clang++
%.o: %.cpp
	$(CLANG) $(CXXFLAGS) -c $< -o $@

# 3. Link everything together with clang++
$(TARGET): $(HOST_OBJ) $(GPU_OBJ)
	$(CLANG) $^ $(LDFLAGS) -o $@

clean:
	rm -f $(TARGET) $(HOST_OBJ) $(GPU_OBJ)

.PHONY: all clean
