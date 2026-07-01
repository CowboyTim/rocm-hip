#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <hip/hip_runtime.h>

// FP4 ROCmFP4 Codebook10
static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

// FP4 quantize (CPU)
uint8_t fp4_quant(float val) {
    float best_dist = 1e30f;
    uint8_t best_idx = 0;
    for (int i = 0; i < 16; ++i) {
        float diff = fabsf((float)fp4_codebook[i] - val);
        if (diff < best_dist) {
            best_dist = diff;
            best_idx = i;
        }
    }
    return best_idx;
}

// FP4 dequant (CPU)
float fp4_dequant(uint8_t nibble) {
    return (float)fp4_codebook[nibble];
}

// FP4 matmul on CPU reference
void fp4_matmul_cpu_ref(const unsigned char* A, const unsigned char* B, float* C,
                        int M, int N, int K, float scale) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += fp4_dequant(A[i * K + k]) * fp4_dequant(B[k * N + j]) * scale;
            }
            C[i * N + j] = sum;
        }
    }
}

// DP4A + FP16-WMMA kernel (externally linked from .o)
extern "C" void wmma_bench_fp4_dp4a(const unsigned char*, const unsigned char*,
                                     float*, int, int, int, float);

// Native FP4 WMMA kernel (scalar fallback for gfx1151, externally linked from .o)
extern "C" void wmma_bench_fp4_native(const unsigned char*, const unsigned char*,
                                      float*, int, int, int, float);

// Native FP4 WMMA kernel using __builtin_amdgcn_wmma_f32_32x16x128_f4 (requires gfx1250+, externally linked from .o)
extern "C" void wmma_bench_fp4_native_gfx1250(const unsigned char*, const unsigned char*,
                                              float*, int, int, int, float);

void run_benchmark(const char* name,
                   void (*kernel)(const unsigned char*, const unsigned char*,
                                  float*, int, int, int, float),
                   const unsigned char* d_A, const unsigned char* d_B,
                   float* d_C, int M, int N, int K, float scale,
                   int NUM_RUNS, int NUM_WARMUP) {
    printf("  Benchmarking: %s\n", name);
    
    hipError_t err;
    size_t resultBytes = N * sizeof(float);
    
    // Warmup
    for (int w = 0; w < NUM_WARMUP; ++w) {
        err = hipMemset(d_C, 0, resultBytes);
        if (err != hipSuccess) {
            printf("    hipMemset failed: %s\n", hipGetErrorString(err));
            return;
        }
        kernel(d_A, d_B, d_C, M, N, K, scale);
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("    Kernel launch failed: %s\n", hipGetErrorString(err));
            return;
        }
    }
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NUM_RUNS; ++r) {
        err = hipMemset(d_C, 0, resultBytes);
        kernel(d_A, d_B, d_C, M, N, K, scale);
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("    Kernel failed: %s\n", hipGetErrorString(err));
            return;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_RUNS;
    double gflops = 2.0 * M * K * N / avg_time / 1e6;
    
    printf("    Avg time:   %.3f ms\n", avg_time);
    printf("    Throughput: %.1f GFLOPS\n", gflops);
    printf("    Total MAC:  %.1f million\n", 2.0 * M * K * N / 1e6f);
}

void verify_results(const char* name,
                    const float* C_host, const float* C_cpu_ref,
                    int M, int N, int max_print) {
    int failures = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = fabsf(C_host[i * N + j] - C_cpu_ref[i * N + j]);
            if (diff > max_diff) max_diff = diff;
            if (diff >= 1.0f) failures++;
        }
    }
    
    float pct = 100.0f * failures / (M * N);
    printf("    Mismatch:   %d / %d elements (%.1f%%)\n", failures, M * N, pct);
    printf("    Max diff:   %.3f\n", max_diff);
    
    if (max_print > 0 && failures > 0) {
        printf("    First few mismatches:\n");
        int count = 0;
        for (int i = 0; i < M && count < max_print; ++i) {
            for (int j = 0; j < N && count < max_print; ++j) {
                float diff = fabsf(C_host[i * N + j] - C_cpu_ref[i * N + j]);
                if (diff >= 1.0f) {
                    printf("      C[%d][%d]: GPU=%.3f  CPU=%.3f  diff=%.3f\n",
                           i, j, C_host[i * N + j], C_cpu_ref[i * N + j], diff);
                    count++;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    printf("[DEBUG] main() entered\n");
    fflush(stdout);
    // Parse cmdline options
    int M = 1024, N = 1024, K = 1024;
    int NUM_RUNS = 10;
    int NUM_WARMUP = 5;
    int VERIFY_SAMPLES = 5;
    float scale_val = 1.0f;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "-runs") == 0 && i + 1 < argc) NUM_RUNS = atoi(argv[++i]);
        else if (strcmp(argv[i], "-warmup") == 0 && i + 1 < argc) NUM_WARMUP = atoi(argv[++i]);
        else if (strcmp(argv[i], "-verify") == 0 && i + 1 < argc) VERIFY_SAMPLES = atoi(argv[++i]);
        else if (strcmp(argv[i], "-scale") == 0 && i + 1 < argc) scale_val = atof(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -M N       Matrix rows (default: 1024)\n");
            printf("  -N N       Matrix cols (default: 1024)\n");
            printf("  -K N       Inner dimension (default: 1024)\n");
            printf("  -runs N    Number of benchmark runs (default: 10)\n");
            printf("  -warmup N  Number of warmup runs (default: 5)\n");
            printf("  -verify N  Number of mismatch samples to print (default: 5, 0=none)\n");
            printf("  -scale F   FP4 scale factor (default: 1.0)\n");
            printf("  -h, --help Show this help\n");
            return 0;
        }
    }
    
    printf("==========================================================\n");
    printf("  FP4 Matmul Benchmark (gfx1151)\n");
    printf("==========================================================\n");
    printf("  Matrix size:  %d x %d x %d\n", M, K, N);
    printf("  FP4 format:   ROCmFP4 Codebook10\n");
    printf("  Values:       0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10\n");
    printf("  Scale:        %.1f\n", scale_val);
    printf("  Runs:         %d (after %d warmup)\n", NUM_RUNS, NUM_WARMUP);
    printf("  Total MAC:    %.1f million\n", 2.0f * M * K * N / 1e6f);
    printf("==========================================================\n\n");
    
    // Allocate host memory
    size_t matA_bytes = M * K;
    size_t matB_bytes = K * N;
    size_t matBytes = matA_bytes;
    size_t outBytes = M * N * sizeof(float);
    size_t resultBytes = N * sizeof(float);
    size_t totalBytes = matA_bytes + matB_bytes + outBytes + outBytes;
    
    const char* filepath = "/dev/shm/fp4_bench.bin";
    
    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return 1; }
    ftruncate(fd, totalBytes);
    
    void* hostPtr = mmap(NULL, totalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    
    unsigned char* A_fp4 = (unsigned char*)hostPtr;
    unsigned char* B_fp4 = (unsigned char*)((char*)hostPtr + matA_bytes);
    float* C_gpu_dp4a = (float*)((char*)hostPtr + matA_bytes + matB_bytes);
    float* C_cpu_ref = (float*)((char*)hostPtr + matA_bytes + matB_bytes + outBytes);
    
    // Generate random FP4 matrices (same seed for reproducibility)
    printf("Generating FP4 matrices...\n");
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        float val = ((float)rand() / (float)RAND_MAX) * 10.0f;
        A_fp4[i] = fp4_quant(val);
    }
    for (int i = 0; i < K * N; ++i) {
        float val = ((float)rand() / (float)RAND_MAX) * 10.0f;
        B_fp4[i] = fp4_quant(val);
    }
    printf("  Done.\n\n");
    
    // Compute CPU reference result (same for both benchmarks)
    printf("Computing CPU reference...\n");
    fp4_matmul_cpu_ref(A_fp4, B_fp4, C_cpu_ref, M, N, K, scale_val);
    printf("  Done. (C_cpu_ref[0] = %.3f)\n\n", C_cpu_ref[0]);
    
    // Allocate device memory
    unsigned char* d_A = NULL, *d_B = NULL;
    float* d_C_dp4a = NULL, *d_C_native = NULL;
    
    printf("Allocating device memory...\n");
    printf("  matA_bytes = %zu, matB_bytes = %zu, outBytes = %zu\n", matA_bytes, matB_bytes, outBytes);
    fflush(stdout);
    
    printf("  Step 1: hipMalloc d_A\n"); fflush(stdout);
    hipError_t err = hipMalloc(&d_A, matA_bytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_A failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Step 2: hipMalloc d_B\n"); fflush(stdout);
    err = hipMalloc(&d_B, matB_bytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_B failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Step 3: hipMalloc d_C_dp4a\n"); fflush(stdout);
    err = hipMalloc(&d_C_dp4a, outBytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_C_dp4a failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Step 4: hipMalloc d_C_native\n"); fflush(stdout);
    err = hipMalloc(&d_C_native, outBytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_C_native failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Device memory allocated.\n\n");
    
    // Copy input data to device
    printf("Copying data to device...\n");
    err = hipMemcpy(d_A, A_fp4, matBytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        printf("  hipMemcpy A failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    err = hipMemcpy(d_B, B_fp4, matB_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        printf("  hipMemcpy B failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Copy done.\n\n");
    
    // ---- Benchmark DP4A + FP16-WMMA ----
    printf("----------------------------------------------------------\n");
    printf("  Method 1: DP4A + FP16-WMMA\n");
    printf("  (Dequant FP4 via DP4A perm, then FP16 WMMA 16x16x16)\n");
    printf("----------------------------------------------------------\n");
    run_benchmark("  DP4A + FP16-WMMA", wmma_bench_fp4_dp4a,
                  d_A, d_B, d_C_dp4a, M, N, K, scale_val, NUM_RUNS, NUM_WARMUP);
    
    // Allocate device memory for native FP4 WMMA result
    float* d_C_native2 = NULL;
    err = hipMalloc(&d_C_native2, outBytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_C_native2 failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    
    // ---- Benchmark Native FP4 WMMA (scalar fallback for gfx1151) ----
    printf("\n----------------------------------------------------------\n");
    printf("  Method 2: Native FP4 WMMA (scalar fallback)\n");
    printf("  (__builtin_amdgcn_wmma_f32_32x16x128_f4 emulation)\n");
    printf("----------------------------------------------------------\n");
    run_benchmark("  Native FP4 WMMA (scalar)", wmma_bench_fp4_native,
                  d_A, d_B, d_C_native, M, N, K, scale_val, NUM_RUNS, NUM_WARMUP);
    
#if __gfx1250__ || defined(ROCWMMA_BUILD_GFX1250)
    // ---- Benchmark Native FP4 WMMA using hardware intrinsic (gfx1250+) ----
    printf("\n----------------------------------------------------------\n");
    printf("  Method 3: Native FP4 WMMA (hardware intrinsic)\n");
    printf("  (__builtin_amdgcn_wmma_f32_32x16x128_f4)\n");
    printf("----------------------------------------------------------\n");
    run_benchmark("  Native FP4 WMMA (hardware)", wmma_bench_fp4_native_gfx1250,
                  d_A, d_B, d_C_native2, M, N, K, scale_val, NUM_RUNS, NUM_WARMUP);
#endif
    
    printf("\n==========================================================\n");
    printf("  Summary\n");
    printf("==========================================================\n");
    
    // Copy results back
    err = hipMemcpy(C_gpu_dp4a, d_C_dp4a, outBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("  hipMemcpy back dp4a failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    float* C_gpu_native = (float*)((char*)hostPtr + matA_bytes + matB_bytes + outBytes);
    err = hipMemcpy(C_gpu_native, d_C_native, outBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("  hipMemcpy back native failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    
#if __gfx1250__ || defined(ROCWMMA_BUILD_GFX1250)
    float* C_gpu_native_hw = (float*)((char*)hostPtr + matA_bytes + matB_bytes + outBytes + outBytes);
    err = hipMemcpy(C_gpu_native_hw, d_C_native2, outBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("  hipMemcpy back native_hw failed: %s\n", hipGetErrorString(err));
        return 1;
    }
#endif
    
    printf("\n  Verification vs CPU reference:\n");
    printf("    DP4A + FP16-WMMA:\n");
    verify_results("    ", C_gpu_dp4a, C_cpu_ref, M, N, VERIFY_SAMPLES);
    printf("\n    Native FP4 WMMA (scalar):\n");
    verify_results("    ", C_gpu_native, C_cpu_ref, M, N, VERIFY_SAMPLES);
    
#if __gfx1250__ || defined(ROCWMMA_BUILD_GFX1250)
    printf("\n    Native FP4 WMMA (hardware):\n");
    verify_results("    ", C_gpu_native_hw, C_cpu_ref, M, N, VERIFY_SAMPLES);
#endif
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C_dp4a);
    hipFree(d_C_native);
#if __gfx1250__ || defined(ROCWMMA_BUILD_GFX1250)
    hipFree(d_C_native2);
#endif
    munmap(hostPtr, totalBytes);
    close(fd);
    
    printf("\nDone.\n");
    return 0;
}
