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
#include <vector>
#include <algorithm>

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

struct BenchmarkResult {
    double avg_ms;
    double gflops;
    float* d_C;
};

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, str); \
            return BenchmarkResult{}; \
        } \
    } while(0)

BenchmarkResult load_and_run_kernel(const char* co_filename, const char* func_name,
    const unsigned char* d_A, const unsigned char* d_B,
    float* d_C, int M, int N, int K, float scale,
    int gridX, int gridY, int threads,
    int NUM_RUNS, int NUM_WARMUP, const char* label) {

    printf("  Loading kernel: %s from %s\n", func_name, co_filename);

    hipModule_t mod;
    hipFunction_t func;
    HIP_CHECK(hipInit(0), "hipInit");

    int kfd = open(co_filename, O_RDONLY);
    if (kfd < 0) {
        perror(co_filename);
        return BenchmarkResult{};
    }
    struct stat st;
    fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    HIP_CHECK(hipModuleLoadData(&mod, coData), "hipModuleLoadData");
    HIP_CHECK(hipModuleGetFunction(&func, mod, func_name), "hipModuleGetFunction");
    munmap(coData, st.st_size);
    close(kfd);

    void* params[7];
    params[0] = (void*)&d_A;
    params[1] = (void*)&d_B;
    params[2] = (void*)&d_C;
    params[3] = (void*)&K;
    params[4] = (void*)&N;
    params[5] = (void*)&M;
    params[6] = (void*)&scale;

    for (int w = 0; w < NUM_WARMUP; ++w) {
        (void)hipMemset(d_C, 0, M * N * sizeof(float));
        HIP_CHECK(hipModuleLaunchKernel(func, gridX, gridY, 1, threads, 1, 1, 0, 0, params, nullptr), "launch warmup");
        HIP_CHECK(hipDeviceSynchronize(), "sync warmup");
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NUM_RUNS; ++r) {
        (void)hipMemset(d_C, 0, M * N * sizeof(float));
        HIP_CHECK(hipModuleLaunchKernel(func, gridX, gridY, 1, threads, 1, 1, 0, 0, params, nullptr), "launch timed");
        HIP_CHECK(hipDeviceSynchronize(), "sync timed");
    }
    auto end = std::chrono::high_resolution_clock::now();

    (void)hipModuleUnload(mod);

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / NUM_RUNS;
    double gflops = 2.0 * M * K * N / (avg_ms * 1e3) / 1e6;

    BenchmarkResult res;
    res.avg_ms = avg_ms;
    res.gflops = gflops;
    res.d_C = d_C;
    return res;
}

void print_stats(const char* name, const BenchmarkResult& res, int M, int N, int K) {
    printf("\n  Results for %s:\n", name);
    printf("    Avg latency:  %.3f ms\n", res.avg_ms);
    printf("    Throughput:   %.1f GFLOPS\n", res.gflops);
    printf("    Total MAC:    %.1f billion\n", 2.0f * M * K * N / 1e9f);
}

void verify_results(const char* label,
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
    printf("%s  Mismatch:   %d / %d elements (%.2f%%)\n",
           label, failures, M * N, pct);
    printf("%s  Max diff:   %.3f\n", label, max_diff);

    if (max_print > 0 && failures > 0) {
        printf("%s  First few mismatches:\n", label);
        int count = 0;
        for (int i = 0; i < M && count < max_print; ++i) {
            for (int j = 0; j < N && count < max_print; ++j) {
                float diff = fabsf(C_host[i * N + j] - C_cpu_ref[i * N + j]);
                if (diff >= 1.0f) {
                    printf("%s    C[%d][%d]: GPU=%.3f  CPU=%.3f  diff=%.3f\n",
                           label, i, j, C_host[i * N + j], C_cpu_ref[i * N + j], diff);
                    count++;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    int NUM_RUNS = 20;
    int NUM_WARMUP = 8;
    int VERIFY_SAMPLES = 3;
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
            printf("  -N N       Matrix cols  (default: 1024)\n");
            printf("  -K N       Inner dim     (default: 1024)\n");
            printf("  -runs N    Benchmark runs (default: 20)\n");
            printf("  -warmup N  Warmup runs    (default: 8)\n");
            printf("  -verify N  Mismatch samples (default: 3, 0=none)\n");
            printf("  -scale F   FP4 scale      (default: 1.0)\n");
            printf("  -h, --help Show this help\n");
            return 0;
        }
    }

    printf("==========================================================\n");
    printf("  FP4 Matmul Benchmark (gfx1151 - Strix Halo)\n");
    printf("==========================================================\n");
    printf("  Matrix size:  %d x %d x %d\n", M, K, N);
    printf("  FP4 format:   ROCmFP4 Codebook10\n");
    printf("  Values:       0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10\n");
    printf("  Scale:        %.1f\n", scale_val);
    printf("  Runs:         %d (after %d warmup)\n", NUM_RUNS, NUM_WARMUP);
    printf("  Total MAC:    %.1f billion\n", 2.0f * M * K * N / 1e9f);
    printf("==========================================================\n\n");

    // Allocate host memory via shared memory file
    size_t matA_bytes = (size_t)M * K;
    size_t matB_bytes = (size_t)K * N;
    size_t outBytes = (size_t)M * N * sizeof(float);
    size_t totalBytes = matA_bytes + matB_bytes + outBytes * 4;

    const char* filepath = "/dev/shm/fp4_bench.bin";

    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return 1; }
    ftruncate(fd, totalBytes);

    void* hostPtr = mmap(NULL, totalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    unsigned char* A_fp4 = (unsigned char*)hostPtr;
    unsigned char* B_fp4 = (unsigned char*)((char*)hostPtr + matA_bytes);
    float* C_gpu_scalar = (float*)((char*)hostPtr + matA_bytes + matB_bytes);
    float* C_gpu_dp4a_hw = (float*)((char*)hostPtr + matA_bytes + matB_bytes + outBytes);
    float* C_gpu_native = (float*)((char*)hostPtr + matA_bytes + matB_bytes + outBytes * 2);
    float* C_cpu_ref = (float*)((char*)hostPtr + matA_bytes + matB_bytes + outBytes * 3);

    // Register host memory for zero-copy
    (void)hipHostRegister(hostPtr, totalBytes, hipHostRegisterDefault);

    unsigned char* d_A, *d_B;
    float* d_C_scalar, *d_C_dp4a_hw, *d_C_native;
    (void)hipHostGetDevicePointer((void**)&d_A, (void*)A_fp4, 0);
    (void)hipHostGetDevicePointer((void**)&d_B, (void*)B_fp4, 0);
    (void)hipHostGetDevicePointer((void**)&d_C_scalar, (void*)C_gpu_scalar, 0);
    (void)hipHostGetDevicePointer((void**)&d_C_dp4a_hw, (void*)C_gpu_dp4a_hw, 0);
    (void)hipHostGetDevicePointer((void**)&d_C_native, (void*)C_gpu_native, 0);

    printf("Memory: UMA zero-copy enabled (dev == host: %s)\n\n",
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
    printf("  Done.\n");

    // Compute CPU reference result
    printf("Computing CPU reference (this may take a while)...\n");
    fp4_matmul_cpu_ref(A_fp4, B_fp4, C_cpu_ref, M, N, K, scale_val);
    printf("  Done. (C_cpu_ref[0] = %.3f)\n\n", C_cpu_ref[0]);

    // ---- Method 1: Scalar LUT dequant + multiply ----
    // Uses wmma_bench_fp4_dp4a.hip (16x16x16 tile, strip=16)
    printf("==========================================================\n");
    printf("  Method 1: Scalar LUT Dequant + Multiply\n");
    printf("  Tile: 16x16x16  |  Grid: (M/16, N/16)  |  32 threads/block\n");
    printf("  Dequant: fp4_codebook_int8[nibble]  (CPU-style LUT)\n");
    printf("  Multiply: scalar float multiply\n");
    printf("==========================================================\n");
    BenchmarkResult scalar_result = load_and_run_kernel(
        "wmma_bench_fp4_dp4a.co", "wmma_bench_fp4_dp4a",
        d_A, d_B, d_C_scalar,
        M, N, K, scale_val,
        (M + 15) / 16, (N + 15) / 16, 32,
        NUM_RUNS, NUM_WARMUP,
        "  Scalar LUT Dequant + Multiply");
    print_stats("  Scalar LUT Dequant + Multiply", scalar_result, M, N, K);

    // ---- Method 2: DP4A Hardware intrinsics ----
    // Uses wmma_bench_fp4_dp4a_hardware.hip (16x16x16 tile, strip=16)
    // Dequant via __builtin_amdgcn_perm + multiply via __builtin_amdgcn_sudot4
    printf("\n==========================================================\n");
    printf("  Method 2: DP4A Hardware Intrinsics\n");
    printf("  Tile: 16x16x16  |  Grid: (M/16, N/16)  |  32 threads/block\n");
    printf("  Dequant: __builtin_amdgcn_perm  (1 hardware instruction)\n");
    printf("  Multiply: __builtin_amdgcn_sudot4  (1 hardware instruction: 4x int8 dot)\n");
    printf("  Note: Only processes 8 FP4 values per strip (vs 16 in scalar)\n");
    printf("==========================================================\n");
    BenchmarkResult dp4a_hw_result = load_and_run_kernel(
        "wmma_bench_fp4_dp4a_hardware.co", "wmma_bench_fp4_dp4a_hardware",
        d_A, d_B, d_C_dp4a_hw,
        M, N, K, scale_val,
        (M + 15) / 16, (N + 15) / 16, 32,
        NUM_RUNS, NUM_WARMUP,
        "  DP4A Hardware Intrinsics");
    print_stats("  DP4A Hardware Intrinsics", dp4a_hw_result, M, N, K);

    // ---- Method 3: Native FP4 WMMA (scalar fallback on gfx1151) ----
    // Uses wmma_bench_fp4_native.hip (32x16x128 tile, strip=128)
    // On gfx1151: scalar FP4 dequant + FP32 accumulate with atomicAdd
    // On gfx1250+: would use native __builtin_amdgcn_wmma_f32_32x16x128_f4
    printf("\n==========================================================\n");
    printf("  Method 3: Native FP4 WMMA (gfx1151 fallback)\n");
    printf("  Tile: 32x16x128  |  Grid: (M/32, N/16)  |  32 threads/block\n");
    printf("  Dequant: fp4_codebook_int8[nibble]  (scalar)\n");
    printf("  Accumulate: atomicAdd to shared memory\n");
    printf("  Note: gfx1151 lacks native FP4 WMMA; uses scalar fallback\n");
    printf("==========================================================\n");
    BenchmarkResult native_result = load_and_run_kernel(
        "wmma_bench_fp4_native.co", "wmma_bench_fp4_native",
        d_A, d_B, d_C_native,
        M, N, K, scale_val,
        (M + 31) / 32, (N + 15) / 16, 32,
        NUM_RUNS, NUM_WARMUP,
        "  Native FP4 WMMA (gfx1151 fallback)");
    print_stats("  Native FP4 WMMA (gfx1151 fallback)", native_result, M, N, K);

    // ---- Summary ----
    printf("\n==========================================================\n");
    printf("  Summary & Comparison\n");
    printf("==========================================================\n\n");

    printf("  Scalar LUT Dequant:       %.1f GFLOPS  (%.3f ms)\n",
           scalar_result.gflops, scalar_result.avg_ms);
    printf("  DP4A Hardware:            %.1f GFLOPS  (%.3f ms)\n",
           dp4a_hw_result.gflops, dp4a_hw_result.avg_ms);
    printf("  Native FP4 WMMA (fallback): %.1f GFLOPS  (%.3f ms)\n",
           native_result.gflops, native_result.avg_ms);

    printf("\n  --- Pairwise speedups (lower is better = faster) ---\n");
    double dp4a_over_scalar = scalar_result.avg_ms / dp4a_hw_result.avg_ms;
    printf("  DP4A / Scalar LUT:        %.2fx\n", dp4a_over_scalar);
    double native_over_scalar = scalar_result.avg_ms / native_result.avg_ms;
    printf("  Native / Scalar LUT:      %.2fx\n", native_over_scalar);
    double dp4a_over_native = native_result.avg_ms / dp4a_hw_result.avg_ms;
    printf("  DP4A / Native:            %.2fx\n", dp4a_over_native);

    // ---- Verification ----
    printf("\n==========================================================\n");
    printf("  Verification vs CPU Reference\n");
    printf("==========================================================\n");
    printf("\n");
    verify_results("  Scalar LUT:       ", C_gpu_scalar, C_cpu_ref, M, N, VERIFY_SAMPLES);
    printf("\n");
    verify_results("  DP4A Hardware:    ", C_gpu_dp4a_hw, C_cpu_ref, M, N, VERIFY_SAMPLES);
    printf("\n");
    verify_results("  Native FP4 WMMA:  ", C_gpu_native, C_cpu_ref, M, N, VERIFY_SAMPLES);

    // Cleanup
    (void)hipHostUnregister(hostPtr);
    munmap(hostPtr, totalBytes);
    close(fd);

    printf("\nDone.\n");
    return 0;
}
