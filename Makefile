# --- Configuration ---
ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc
CLANG     := $(ROCM_PATH)/lib/llvm/bin/clang++

ARCH      := gfx1151

CXXFLAGS  := -O3 -Wall -x hip -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__
HIPFLAGS  := -O3 --offload-arch=$(ARCH) --genco

LDFLAGS   := -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,$(ROCM_PATH)/lib

TARGET    := mandelbrot
KERNEL_SRC = kernels.hip
KERNEL_CO  = kernels.co
OBJ        = main.o

# --- Build Rules ---
all: $(TARGET)

$(TARGET): $(OBJ) $(KERNEL_CO)
	$(CLANG) $(OBJ) $(LDFLAGS) -o $@

%.co: %.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

%.o: %.cpp
	$(CLANG) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJ) $(KERNEL_CO)

.PHONY: all clean
