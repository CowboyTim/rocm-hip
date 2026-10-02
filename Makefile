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

# --- Phase 1: GGUF mmap reader ---
GGUF_MMAP_SRC = gguf_mmap.cpp
DEMO_GGUF     = demo_gguf

# Phase 2: ROCmFP4 Codebook10 matmul verification harness (CPU reference vs GPU kernels)
FP4_VERIFY   = fp4_verify

# Phase 3: KV cache manager + zero-copy self-test on hardware
KV_VERIFY    = kv_verify

# Phase 3: FP16 WMMA GEMM verification harness (CPU reference vs GPU kernel)
GEMM_VERIFY  = gemm_verify

# Phase 3: RMSNorm / RoPE / masked-softmax verification harnesses
LAYERNORM_VERIFY = layernorm_verify

# Phase 3: FP16 activation x Codebook10-FP4 weight GEMM verification (the projection path)
ACTFP4_VERIFY    = actfp4_verify

# Phase 3: end-to-end attention block integration test (chains every Phase-3 kernel)
ATTN_VERIFY      = attn_verify
ROPE_VERIFY      = rope_verify
SOFTMAX_VERIFY   = softmax_verify

# Phase 4: Model architecture kernels
SWIGLU_CO        = swiglu.co
EMBEDDING_CO     = embedding_lookup.co
QWEN_MODEL_CO    = qwen_model.co

# Phase 4: Inference engine library and demo
INFERENCE_LIB    = libinference.a
INFERENCE_DEMO   = inference_demo

# Phase 4: Qwen 3 model library and demo
QWEN_MODEL_LIB   = libqwen.a
QWEN_DEMO        = qwen_demo

# --- Build Rules ---
all: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET) $(DEMO_GGUF) $(FP4_VERIFY) $(KV_VERIFY) $(GEMM_VERIFY) $(LAYERNORM_VERIFY) $(ROPE_VERIFY) $(SOFTMAX_VERIFY) $(ACTFP4_VERIFY) $(ATTN_VERIFY) $(SWIGLU_CO) $(EMBEDDING_CO) $(QWEN_MODEL_CO)

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

# Benchmark kernels -> .co files (same approach as wmma_matmul)
wmma_bench_fp4_dp4a.co: wmma_bench_fp4_dp4a.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

wmma_bench_fp4_native.co: wmma_bench_fp4_native.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# DP4A hardware intrinsics kernel (uses __builtin_amdgcn_perm + sudot4)
wmma_bench_fp4_dp4a_hardware.co: wmma_bench_fp4_dp4a_hardware.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Build the benchmark binary (links host .o, loads .co at runtime)
$(BENCH_TARGET): $(BENCH_HOST) wmma_bench_fp4_dp4a.co wmma_bench_fp4_dp4a_hardware.co wmma_bench_fp4_native.co $(BENCH_LUT_CO)
	$(CLANG) $(BENCH_HOST) $(LDFLAGS) -o $@

# Phase 1 demo: gguf_mmap reader linked into a single binary
$(DEMO_GGUF): $(GGUF_MMAP_SRC) demo_gguf.cpp gguf_mmap.h
	$(CLANG) -O3 -Wall -I. -D__HIP_PLATFORM_AMD__ \
		$(GGUF_MMAP_SRC) demo_gguf.cpp -o $@

# Phase 2 verify: host harness loads both .co files at runtime, compares against CPU reference
$(FP4_VERIFY): fp4_verify.cpp fp4_common.h fp4_matmul_kernel.co fp4_matmul_wmma.co
	$(CLANG) $(CXXFLAGS) -I. fp4_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: KV cache manager + zero-copy self-test (loads kv_cache_kernel.co at runtime)
$(KV_VERIFY): kv_cache.cpp kv_verify.cpp kv_cache.h kv_cache_kernel.co
	$(CLANG) $(CXXFLAGS) -I. kv_cache.cpp kv_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: FP16 WMMA GEMM vs CPU reference over /dev/shm (loads fp16_gemm.co at runtime)
$(GEMM_VERIFY): gemm_verify.cpp fp16_gemm.co
	$(CLANG) $(CXXFLAGS) -I. gemm_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: RMSNorm kernel vs CPU reference (loads layernorm.co at runtime)
$(LAYERNORM_VERIFY): layernorm_verify.cpp layernorm.co
	$(CLANG) $(CXXFLAGS) -I. layernorm_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: RoPE rotate-half kernel vs CPU reference (loads rope_embedding.co at runtime)
$(ROPE_VERIFY): rope_verify.cpp rope_embedding.co
	$(CLANG) $(CXXFLAGS) -I. rope_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: masked row-softmax vs CPU reference (loads softmax_masked.co at runtime)
$(SOFTMAX_VERIFY): softmax_verify.cpp softmax_masked.co
	$(CLANG) $(CXXFLAGS) -I. softmax_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: Act x FP4 GEMM vs CPU reference over /dev/shm (loads gemm_act_fp4.co at runtime)
$(ACTFP4_VERIFY): actfp4_verify.cpp gemm_act_fp4.co
	$(CLANG) $(CXXFLAGS) -I. actfp4_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: full attention block chain; loads layernorm/gemm_act_fp4/rope_embedding/kv_cache_store/fp16_gemm/softmax_masked .co files
$(ATTN_VERIFY): attn_verify.cpp kv_cache.cpp kv_cache.h kv_cache_store.co gemm_act_fp4.co fp16_gemm.co layernorm.co rope_embedding.co softmax_masked.co
	$(CLANG) $(CXXFLAGS) -I. attn_verify.cpp kv_cache.cpp $(LDFLAGS) -o $@

# Phase 4: SwiGLU kernel (.co for runtime loading)
$(SWIGLU_CO): swiglu.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Phase 4: Embedding lookup kernel (.co for runtime loading)
$(EMBEDDING_CO): embedding_lookup.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Phase 4: Inference engine static library
$(INFERENCE_LIB): inference.cpp inference.h gguf_mmap.cpp gguf_mmap.h kv_cache.cpp kv_cache.h tokenizer.cpp tokenizer.h
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c inference.cpp -o inference.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c gguf_mmap.cpp -o gguf_mmap.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c kv_cache.cpp -o kv_cache.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c tokenizer.cpp -o tokenizer.o
	ar rcs $@ inference.o gguf_mmap.o kv_cache.o tokenizer.o

# Phase 4: Inference demo binary
$(INFERENCE_DEMO): inference.cpp inference.h demo_inference.cpp
	$(CLANG) $(CXXFLAGS) -I. inference.cpp gguf_mmap.cpp kv_cache.cpp demo_inference.cpp $(LDFLAGS) -o $@

# Phase 4: Qwen 3 model kernel (.co for runtime loading)
$(QWEN_MODEL_CO): qwen_model.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Phase 4: Qwen 3 model static library
$(QWEN_MODEL_LIB): qwen_model.cpp qwen_model.h inference.h gguf_mmap.h kv_cache.h
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c qwen_model.cpp -o qwen_model.o
	ar rcs $@ qwen_model.o

# Phase 4: Qwen 3 demo binary
$(QWEN_DEMO): qwen_model.cpp qwen_model.h inference.cpp gguf_mmap.cpp kv_cache.cpp tokenizer.cpp demo_qwen.cpp
	$(CLANG) $(CXXFLAGS) -I. qwen_model.cpp inference.cpp gguf_mmap.cpp kv_cache.cpp tokenizer.cpp demo_qwen.cpp $(LDFLAGS) -o $@

# Phase 6: Benchmark orchestrator (Python script)
BENCH_SCRIPT = bench_inference.py

# Phase 7: Integration test (runs demo_qwen if MODEL_PATH provided)
INTEGRATION_TEST = integration_test

# --- Build Rules ---
all: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET) $(DEMO_GGUF) $(FP4_VERIFY) $(KV_VERIFY) $(GEMM_VERIFY) $(LAYERNORM_VERIFY) $(ROPE_VERIFY) $(SOFTMAX_VERIFY) $(ACTFP4_VERIFY) $(ATTN_VERIFY) $(SWIGLU_CO) $(EMBEDDING_CO) $(QWEN_MODEL_CO)

# Copy benchmark script to build directory (so it's alongside binaries)
$(BENCH_SCRIPT): bench_inference.py
	cp bench_inference.py $(BENCH_SCRIPT)

# Phase 6: run benchmark (requires MODEL_PATH env var)
bench: $(INFERENCE_DEMO) $(BENCH_SCRIPT)
	@if [ -z "$$MODEL_PATH" ]; then echo "Set MODEL_PATH to run benchmark"; exit 1; fi; \
	python3 $(BENCH_SCRIPT) --model $$MODEL_PATH --prompt "The quick brown fox" --n-tokens 20 --warmup 2 --runs 5 --output bench_report.json

# Phase 7: integration test (runs qwen_demo with a small prompt if MODEL_PATH set)
integration_test: $(QWEN_DEMO)
	@if [ -z "$$MODEL_PATH" ]; then echo "MODEL_PATH not set, skipping integration test"; exit 0; fi; \
	./$(QWEN_DEMO) $$MODEL_PATH "Hello world" && echo "Integration test passed"

clean:
	rm -f $(TARGET) $(OBJ) $(KERNEL_CO) $(WMMA_TARGET) $(WMMA_OBJ) $(WMMA_CO) $(WMMA_FP4_TARGET) $(WMMA_FP4_OBJ) $(WMMA_FP4_CO) $(WMMA_FP4_NATIVE_TARGET) $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO) $(BENCH_TARGET) $(BENCH_HOST) wmma_bench_fp4_dp4a.co wmma_bench_fp4_native.co wmma_bench_fp4_dp4a_hardware.co $(BENCH_LUT_CO) $(DEMO_GGUF) $(FP4_VERIFY) $(KV_VERIFY) $(GEMM_VERIFY) $(LAYERNORM_VERIFY) $(ROPE_VERIFY) $(SOFTMAX_VERIFY) $(ACTFP4_VERIFY) $(ATTN_VERIFY) $(SWIGLU_CO) $(EMBEDDING_CO) $(QWEN_MODEL_CO) $(INFERENCE_LIB) $(INFERENCE_DEMO) $(QWEN_MODEL_LIB) $(QWEN_DEMO) *.co *.o

.PHONY: all clean both
