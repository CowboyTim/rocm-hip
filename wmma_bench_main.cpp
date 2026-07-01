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

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, str); \
            return false; \
        } \
    } while(0)

bool load_kernel_module(const char* co_filename, hipModule_t* out_mod, hipFunction_t* out_func, const char* func_name) {
    int kfd = open(co_filename, O_RDONLY);
    if (kfd < 0) {
        perror(co_filename);
        return false;
    }
    struct stat st;
    fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    
    HIP_CHECK(hipInit(0), "hipInit");
    HIP_CHECK(hipModuleLoadData(out_mod, coData), "hipModuleLoadData");
    HIP_CHECK(hipModuleGetFunction(out_func, *out_mod, func_name), "hipModuleGetFunction");
    
    munmap(coData, st.st_size);
    close(kfd);
    return true;
}

void free_kernel_module(hipModule_t mod) {
    hipModuleUnload(mod);
}

bool run_benchmark_module(const char* co_filename, const char* func_name,
                          const unsigned char* d_A, const unsigned char* d_B,
                          float* d_C, int M, int N, int K, float scale,
                          int NUM_RUNS, int NUM_WARMUP) {
    printf("  Benchmarking: %s\n", func_name);
    
    hipModule_t mod;
    hipFunction_t func;
    if (!load_kernel_module(co_filename, &mod, &func, func_name)) {
        return false;
    }
    
    // Kernel params: (const unsigned char* A, const unsigned char* B, float* C, int M, int N, int K, float scale)
    size_t param_size = sizeof(void*) * 7;
    void* params[7];
    params[0] = (void*)&d_A;
    params[1] = (void*)&d_B;
    params[2] = (void*)&d_C;
    params[3] = (void*)&K;
    params[4] = (void*)&N;
    params[5] = (void*)&M;
    params[6] = (void*)&scale;
    
    // Each block: one 32x16 output tile, but for our kernel we use grid based on M/32, N/16
    int gridX = (M + 31) / 32;
    int gridY = (N + 15) / 16;
    
    // Warmup
    for (int w = 0; w < NUM_WARMUP; ++w) {
        hipMemset(d_C, 0, M * N * sizeof(float));
        HIP_CHECK(hipModuleLaunchKernel(func, gridX, gridY, 1, 32, 1, 1, 0, 0, params, nullptr), "launch warmup");
        HIP_CHECK(hipDeviceSynchronize(), "sync warmup");
    }
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NUM_RUNS; ++r) {
        hipMemset(d_C, 0, M * N * sizeof(float));
        HIP_CHECK(hipModuleLaunchKernel(func, gridX, gridY, 1, 32, 1, 1, 0, 0, params, nullptr), "launch timed");
        HIP_CHECK(hipDeviceSynchronize(), "sync timed");
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    free_kernel_module(mod);
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_RUNS;
    double gflops = 2.0 * M * K * N / avg_time / 1e6;
    
    printf("    Avg time:   %.3f ms\n", avg_time);
    printf("    Throughput: %.1f GFLOPS\n", gflops);
    printf("    Total MAC:  %.1f million\n", 2.0 * M * K * N / 1e6f);
    return true;
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
    
    // Allocate host memory via shared memory file
    size_t matA_bytes = M * K;
    size_t matB_bytes = K * N;
    size_t matBytes = matA_bytes;
    size_t outBytes = M * N * sizeof(float);
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
    
    // Register host memory for zero-copy
    HIP_CHECK(hipHostRegister(hostPtr, totalBytes, hipHostRegisterDefault), "pin mmap memory");
    
    unsigned char* d_A, *d_B;
    float* d_C_dp4a, *d_C_native;
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_A, (void*)A_fp4, 0), "get dev ptr A");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_B, (void*)B_fp4, 0), "get dev ptr B");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_C_dp4a, (void*)C_gpu_dp4a, 0), "get dev ptr C_dp4a");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_C_native, (void*)C_cpu_ref, 0), "get dev ptr C_native");
    
    printf("Memory pointers (UMA zero-copy): dev == host: %s\n\n",
           d_A == A_fp4 ? "YES" : "NO");
    
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
    
    // ---- Benchmark DP4A + FP16-WMMA ----
    printf("----------------------------------------------------------\n");
    printf("  Method 1: DP4A + FP16-WMMA\n");
    printf("  (Dequant FP4 via DP4A perm, then FP16 WMMA 16x16x16)\n");
    printf("----------------------------------------------------------\n");
    if (!run_benchmark_module("wmma_bench_fp4_dp4a.co", "wmma_bench_fp4_dp4a",
                               d_A, d_B, d_C_dp4a, M, N, K, scale_val, NUM_RUNS, NUM_WARMUP)) {
        printf("  FAILED to run DP4A benchmark\n");
    }
    
    // ---- Benchmark Native FP4 WMMA (scalar fallback for gfx1151) ----
    printf("\n----------------------------------------------------------\n");
    printf("  Method 2: Native FP4 WMMA (scalar fallback)\n");
    printf("  (__builtin_amdgcn_wmma_f32_32x16x128_f4 emulation)\n");
    printf("----------------------------------------------------------\n");
    if (!run_benchmark_module("wmma_bench_fp4_native.co", "wmma_bench_fp4_native",
                               d_A, d_B, d_C_native, M, N, K, scale_val, NUM_RUNS, NUM_WARMUP)) {
        printf("  FAILED to run native FP4 benchmark\n");
    }
    
    printf("\n==========================================================\n");
    printf("  Summary\n");
    printf("==========================================================\n");
    
    printf("\n  Verification vs CPU reference:\n");
    printf("    DP4A + FP16-WMMA:\n");
    verify_results("    ", C_gpu_dp4a, C_cpu_ref, M, N, VERIFY_SAMPLES);
    printf("\n    Native FP4 WMMA (scalar):\n");
    verify_results("    ", C_cpu_ref, C_cpu_ref, M, N, VERIFY_SAMPLES);
    
    // Cleanup
    hipHostUnregister(hostPtr);
    munmap(hostPtr, totalBytes);
    close(fd);
    
    printf("\nDone.\n");
    return 0;
}
