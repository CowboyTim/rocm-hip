# --- Configuration ---
ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc
CLANG     := $(ROCM_PATH)/lib/llvm/bin/clang++

ARCH      := gfx1151

# Directories
SRCDIR     := src
BUILDDIR   := build
OBJDIR     := $(BUILDDIR)/obj
BINDIR     := $(BUILDDIR)
CODIR      := $(BUILDDIR)/co
SRCDIR     := src
TESTSDIR   := tests

# Host C++ flags (no offloading)
CXXFLAGS  := -O3 -Wall -I$(ROCM_PATH)/include -I$(SRCDIR) -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__

# Device/HIP kernel flags (with offloading)
HIPFLAGS  := -Xarch_device -mno-wavefrontsize64 -isystem $(ROCM_PATH)/include --rocm-path=$(ROCM_PATH) -O3 --offload-arch=$(ARCH) --genco -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__

LDFLAGS   := -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,$(ROCM_PATH)/lib

# Ensure build directories exist
$(shell mkdir -p $(OBJDIR) $(BINDIR) $(CODIR))

# --- Demos ---

# WMMA matmul (FP16): uses hipModuleLoadData + mmap zero-copy
WMMA_SRC  = $(SRCDIR)/wmma_kernel.hip
WMMA_CO   = $(CODIR)/wmma_kernel.co
WMMA_OBJ  = $(OBJDIR)/wmma_main.o
WMMA_TARGET = $(BINDIR)/wmma_matmul

# WMMA matmul (FP4 ROCmFP4): FP4 dequant via DP4A perm + WMMA FP16
WMMA_FP4_SRC  = $(SRCDIR)/wmma_kernel_fp4.hip
WMMA_FP4_CO   = $(CODIR)/wmma_kernel_fp4.co
WMMA_FP4_OBJ  = $(OBJDIR)/wmma_main_fp4.o
WMMA_FP4_TARGET = $(BINDIR)/wmma_matmul_fp4

# WMMA matmul (FP4 native): uses __builtin_amdgcn_wmma_f32_32x16x128_f4
WMMA_FP4_NATIVE_SRC  = $(SRCDIR)/wmma_kernel_fp4_native.hip
WMMA_FP4_NATIVE_CO   = $(CODIR)/wmma_kernel_fp4_native.co
WMMA_FP4_NATIVE_OBJ  = $(OBJDIR)/wmma_main_fp4_native.o
WMMA_FP4_NATIVE_TARGET = $(BINDIR)/wmma_matmul_fp4_native

# FP4 Matmul Benchmark
BENCH_TARGET = $(BINDIR)/wmma_bench
BENCH_HOST = $(OBJDIR)/wmma_bench_main.o
BENCH_LUT_CO = $(CODIR)/wmma_bench_fp4_lut.co

# --- Phase 1: GGUF mmap reader ---
GGUF_MMAP_SRC = $(SRCDIR)/gguf_mmap.cpp
DEMO_GGUF     = $(BINDIR)/demo_gguf

# Phase 2: ROCmFP4 Codebook10 matmul verification harness (CPU reference vs GPU kernels)
FP4_VERIFY   = $(BINDIR)/fp4_verify

# Phase 3: KV cache manager + zero-copy self-test on hardware
KV_VERIFY    = $(BINDIR)/kv_verify

# Phase 3: FP16 WMMA GEMM verification harness (CPU reference vs GPU kernel)
GEMM_VERIFY  = $(BINDIR)/gemm_verify

# Phase 3: RMSNorm / RoPE / masked-softmax verification harnesses
LAYERNORM_VERIFY = $(BINDIR)/layernorm_verify
ROPE_VERIFY      = $(BINDIR)/rope_verify
SOFTMAX_VERIFY   = $(BINDIR)/softmax_verify

# Phase 3: FP16 activation x Codebook10-FP4 weight GEMM verification (the projection path)
ACTFP4_VERIFY    = $(BINDIR)/actfp4_verify

# Phase 9: Tokenizer verification
TOKENIZER_VERIFY = $(BINDIR)/tokenizer_verify

# Phase 3: end-to-end attention block integration test (chains every Phase-3 kernel)
ATTN_VERIFY      = $(BINDIR)/attn_verify

# Phase 4: Model architecture kernels
SWIGLU_CO        = $(CODIR)/swiglu.co
EMBEDDING_CO     = $(CODIR)/embedding_lookup.co
QWEN_MODEL_CO    = $(CODIR)/qwen_model.co

# Phase 4: Inference engine library and demo
INFERENCE_LIB    = $(BUILDDIR)/libinference.a
INFERENCE_DEMO   = $(BINDIR)/inference_demo

# Phase 4: Qwen 3 model library and demo
QWEN_MODEL_LIB   = $(BUILDDIR)/libqwen.a
QWEN_DEMO        = $(BINDIR)/qwen_demo

# --- Build Rules ---
all: $(WMMA_TARGET) $(KV_VERIFY) $(GEMM_VERIFY) $(LAYERNORM_VERIFY) $(ROPE_VERIFY) $(SOFTMAX_VERIFY) $(ACTFP4_VERIFY) $(TOKENIZER_VERIFY) $(ATTN_VERIFY) $(SWIGLU_CO) $(EMBEDDING_CO) $(QWEN_MODEL_CO) $(INFERENCE_LIB) $(QWEN_MODEL_LIB) $(QWEN_DEMO)

$(WMMA_TARGET): $(WMMA_OBJ) $(WMMA_CO)
	$(CLANG) $(WMMA_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_TARGET): $(WMMA_FP4_OBJ) $(WMMA_FP4_CO)
	$(CLANG) $(WMMA_FP4_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_NATIVE_TARGET): $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO)
	$(CLANG) $(WMMA_FP4_NATIVE_OBJ) $(LDFLAGS) -o $@

# Pre-compiled kernel objects (.hip -> .co)
$(CODIR)/%.co: $(SRCDIR)/%.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@
	cp $@ $(BINDIR)/
	cp $@ .

# Host C++ files (.cpp -> .o) -- compiled without device offloading
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CLANG) $(CXXFLAGS) -c $< -o $@

# Benchmark kernels -> .co files (same approach as wmma_matmul)
$(CODIR)/wmma_bench_fp4_dp4a.co: $(SRCDIR)/wmma_bench_fp4_dp4a.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

$(CODIR)/wmma_bench_fp4_native.co: $(SRCDIR)/wmma_bench_fp4_native.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# DP4A hardware intrinsics kernel (uses __builtin_amdgcn_perm + sudot4)
$(CODIR)/wmma_bench_fp4_dp4a_hardware.co: $(SRCDIR)/wmma_bench_fp4_dp4a_hardware.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Build the benchmark binary (links host .o, loads .co at runtime)
$(BENCH_TARGET): $(BENCH_HOST) $(CODIR)/wmma_bench_fp4_dp4a.co $(CODIR)/wmma_bench_fp4_dp4a_hardware.co $(CODIR)/wmma_bench_fp4_native.co $(BENCH_LUT_CO)
	$(CLANG) $(BENCH_HOST) $(LDFLAGS) -o $@

# Phase 1 demo: gguf_mmap reader linked into a single binary
$(DEMO_GGUF): $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/demo_gguf.cpp $(SRCDIR)/gguf_mmap.h
	$(CLANG) -O3 -Wall -I. -D__HIP_PLATFORM_AMD__ \
		$(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/demo_gguf.cpp -o $@

# Phase 2 verify: host harness loads both .co files at runtime, compares against CPU reference
$(FP4_VERIFY): $(SRCDIR)/fp4_verify.cpp $(SRCDIR)/fp4_common.h $(CODIR)/fp4_matmul_kernel.co $(CODIR)/fp4_matmul_wmma.co
	$(CLANG) $(CXXFLAGS) -I. $(SRCDIR)/fp4_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: KV cache manager + zero-copy self-test (loads kv_cache_kernel.co at runtime)
$(KV_VERIFY): $(SRCDIR)/kv_cache.cpp $(TESTSDIR)/kv_verify.cpp $(SRCDIR)/kv_cache.h $(CODIR)/kv_cache_kernel.co
	$(CLANG) $(CXXFLAGS) -I. $(SRCDIR)/kv_cache.cpp $(TESTSDIR)/kv_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: FP16 WMMA GEMM vs CPU reference over /dev/shm (loads fp16_gemm.co at runtime)
$(GEMM_VERIFY): $(TESTSDIR)/gemm_verify.cpp $(CODIR)/fp16_gemm.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/gemm_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: RMSNorm kernel vs CPU reference (loads layernorm.co at runtime)
$(LAYERNORM_VERIFY): $(TESTSDIR)/layernorm_verify.cpp $(CODIR)/layernorm.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/layernorm_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: RoPE rotate-half kernel vs CPU reference (loads rope_embedding.co at runtime)
$(ROPE_VERIFY): $(TESTSDIR)/rope_verify.cpp $(CODIR)/rope_embedding.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/rope_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: masked row-softmax vs CPU reference (loads softmax_masked.co at runtime)
$(SOFTMAX_VERIFY): $(TESTSDIR)/softmax_verify.cpp $(CODIR)/softmax_masked.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/softmax_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: Act x FP4 GEMM vs CPU reference over /dev/shm (loads gemm_act_fp4.co at runtime)
$(ACTFP4_VERIFY): $(TESTSDIR)/actfp4_verify.cpp $(CODIR)/gemm_act_fp4.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/actfp4_verify.cpp $(LDFLAGS) -o $@

# Phase 9 verify: Tokenizer BPE encode/decode round‑trip
$(TOKENIZER_VERIFY): $(TESTSDIR)/tokenizer_verify.cpp $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/tokenizer_verify.cpp $(INFERENCE_LIB) $(LDFLAGS) -o $@

# Phase 3 verify: full attention block chain; loads layernorm/gemm_act_fp4/rope_embedding/kv_cache_store/fp16_gemm/softmax_masked .co files
$(ATTN_VERIFY): $(TESTSDIR)/attn_verify.cpp $(SRCDIR)/kv_cache.cpp $(SRCDIR)/kv_cache.h $(CODIR)/kv_cache_store.co $(CODIR)/gemm_act_fp4.co $(CODIR)/fp16_gemm.co $(CODIR)/layernorm.co $(CODIR)/rope_embedding.co $(CODIR)/softmax_masked.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/attn_verify.cpp $(SRCDIR)/kv_cache.cpp $(LDFLAGS) -o $@

# Phase 4: SwiGLU kernel (.co for runtime loading)
$(SWIGLU_CO): $(SRCDIR)/swiglu.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Phase 4: Embedding lookup kernel (.co for runtime loading)
$(EMBEDDING_CO): $(SRCDIR)/embedding_lookup.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Phase 4: Qwen 3 model kernel (.co for runtime loading)
$(QWEN_MODEL_CO): $(SRCDIR)/qwen_model.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Phase 4: Inference engine static library
$(INFERENCE_LIB): $(SRCDIR)/inference.cpp $(SRCDIR)/inference.h $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/gguf_mmap.h $(SRCDIR)/kv_cache.cpp $(SRCDIR)/kv_cache.h $(SRCDIR)/tokenizer.cpp $(SRCDIR)/tokenizer.h
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/inference.cpp -o $(OBJDIR)/inference.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/gguf_mmap.cpp -o $(OBJDIR)/gguf_mmap.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/kv_cache.cpp -o $(OBJDIR)/kv_cache.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/tokenizer.cpp -o $(OBJDIR)/tokenizer.o
	ar rcs $@ $(OBJDIR)/inference.o $(OBJDIR)/gguf_mmap.o $(OBJDIR)/kv_cache.o $(OBJDIR)/tokenizer.o



# Phase 4: Qwen 3 model static library
$(QWEN_MODEL_LIB): $(SRCDIR)/qwen_model.cpp $(SRCDIR)/qwen_model.h $(SRCDIR)/inference.h $(SRCDIR)/gguf_mmap.h $(SRCDIR)/kv_cache.h
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/qwen_model.cpp -o $(OBJDIR)/qwen_model.o
	ar rcs $@ $(OBJDIR)/qwen_model.o

# Phase 4: Qwen 3 demo binary
$(QWEN_DEMO): $(SRCDIR)/qwen_model.cpp $(SRCDIR)/qwen_model.h $(SRCDIR)/inference.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/kv_cache.cpp $(SRCDIR)/demo_qwen.cpp $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(SRCDIR)/qwen_model.cpp $(SRCDIR)/inference.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/kv_cache.cpp $(SRCDIR)/demo_qwen.cpp $(INFERENCE_LIB) $(LDFLAGS) -o $@

# Build all demos and benchmark
both: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET)

clean:
	rm -rf $(BUILDDIR)

.PHONY: all clean both tokenizer_verify

tokenizer_verify: $(TOKENIZER_VERIFY)
