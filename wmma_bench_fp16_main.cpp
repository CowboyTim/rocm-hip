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
#include <hip/hip_fp16.h>
#include <vector>
#include <algorithm>

static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

struct BenchmarkResult {
    double avg_ms;
    double gflops;
    __half* d_C;
};

hipError_t hip_check(hipError_t e, const char* msg) {
    if (e != hipSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", msg, hipGetErrorString(e));
    }
    return e;
}

BenchmarkResult run_kernel_fp16(const char* co_filename, const char* func_name,
    const unsigned char* d_A, const __half* d_B,
    __half* d_C, int M, int N, int K, float scale,
    int gridX, int gridY, int threads,
    int NUM_RUNS, int NUM_WARMUP, const char* label) {

    printf("  Loading kernel: %s from %s\n", func_name, co_filename);

    hipModule_t mod;
    int kfd = open(co_filename, O_RDONLY);
    if (kfd < 0) { perror("open"); return BenchmarkResult{}; }
    
    struct stat st;
    fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    
    hip_check(hipInit(0), "init");
    hip_check(hipModuleLoadData(&mod, coData), "load module");
    
    hipFunction_t func;
    hip_check(hipModuleGetFunction(&func, mod, func_name), "get func");
    
    void* params[] = { &d_A, &d_B, &d_C, &M, &N, &K, &scale };
    
    for (int w = 0; w < NUM_WARMUP; ++w) {
        (void)hipMemset(d_C, 0, M * N * sizeof(__half));
        hip_check(hipModuleLaunchKernel(func, gridX, gridY, 1, threads, 1, 1, 0, 0, params, nullptr), "launch warmup");
        hipDeviceSynchronize();
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NUM_RUNS; ++r) {
        (void)hipMemset(d_C, 0, M * N * sizeof(__half));
        hip_check(hipModuleLaunchKernel(func, gridX, gridY, 1, threads, 1, 1, 0, 0, params, nullptr), "launch timed");
        hipDeviceSynchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / NUM_RUNS;
    double gflops = 2.0 * M * K * N / (avg_ms * 1e3) / 1e6;
    
    (void)hipModuleUnload(mod);
    
    BenchmarkResult res;
    res.avg_ms = avg_ms;
    res.gflops = gflops;
    res.d_C = d_C;
    return res;
}

void verify_results_fp16(const char* label, const __half* C_host, const float* C_cpu_ref, int M, int N) {
    int failures = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float gpu_val = __half2float(C_host[i * N + j]);
            float cpu_val = C_cpu_ref[i * N + j];
            float diff = fabsf(gpu_val - cpu_val);
            if (diff > 0.01f * (fabsf(cpu_val) + 1e-5f)) {
                failures++;
                if (failures <= 5) {
                    printf("    [%d,%d]: GPU=%.4f CPU=%.4f diff=%.4f\n", i, j, gpu_val, cpu_val, diff);
                }
            }
            max_diff = fmaxf(max_diff, diff);
        }
    }
    
    float mismatch_pct = 100.0f * failures / (M * N);
    printf("  %s:\n", label);
    printf("    Mismatch: %d/%d (%.2f%%)\n", failures, M * N, mismatch_pct);
    printf("    Max diff: %.6f\n", max_diff);
}

int main(int argc, char** argv) {
    int M = 256, N = 256, K = 256;
    int runs = 20, warmup = 8;
    float scale = 1.0f;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-M") == 0) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-N") == 0) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-K") == 0) K = atoi(argv[++i]);
        else if (strcmp(argv[i], "-runs") == 0) runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "-warmup") == 0) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "-scale") == 0) scale = atof(argv[++i]);
    }
    
    printf("\n=== FP4 × FP16 WMMA Benchmark (direct FP16 activations) ===\n");
    printf("  Matrix size: %d x %d x %d\n", M, K, N);
    printf("  Scale: %.2f\n", scale);
    printf("  Runs: %d (after %d warmup)\n", runs, warmup);
    
    size_t W_bytes = M * K;
    size_t A_bytes = K * N * sizeof(__half);
    size_t C_bytes = M * N * sizeof(__half);
    size_t total_bytes = W_bytes + A_bytes + C_bytes + M * N * sizeof(float);
    
    int fd = open("/dev/shm/wmma_fp16_bench.bin", O_RDWR | O_CREAT, 0666);
    ftruncate(fd, total_bytes);
    void* hostPtr = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    
    unsigned char* W_fp4 = (unsigned char*)hostPtr;
    __half* A_fp16 = (__half*)(W_fp4 + W_bytes);
    __half* C_gpu = (__half*)(A_fp16 + K * N);
    float* C_cpu = (float*)(C_gpu + M * N);
    
    hip_check(hipHostRegister(hostPtr, total_bytes, hipHostRegisterDefault), "pin");
    
    unsigned char* d_W; __half* d_A, *d_C;
    hipHostGetDevicePointer((void**)&d_W, (void*)W_fp4, 0);
    hipHostGetDevicePointer((void**)&d_A, (void*)A_fp16, 0);
    hipHostGetDevicePointer((void**)&d_C, (void*)C_gpu, 0);
    
    for (int i = 0; i < M * K; ++i) W_fp4[i] = i % 8;
    for (int i = 0; i < K * N; ++i) A_fp16[i] = __half((float)(i % 16) - 8.0f + 0.5f * (i % 2));
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                int8_t w_val = fp4_codebook[W_fp4[i * K + k]];
                float a_val = __half2float(A_fp16[k * N + j]);
                sum += w_val * a_val;
            }
            C_cpu[i * N + j] = sum * scale;
        }
    }
    
    int gridX = (N + 15) / 16;
    int gridY = (M + 15) / 16;
    int threads = 32;
    
    BenchmarkResult res = run_kernel_fp16(
        "wmma_bench_fp4_lut.co", "wmma_bench_fp4_lut",
        d_W, d_A, d_C, M, N, K, scale, gridX, gridY, threads, runs, warmup,
        "FP4→FP16 WMMA + LDS LUT"
    );
    
    printf("\n  Results for FP4→FP16 WMMA + LDS LUT:\n");
    printf("    Avg latency:  %.3f ms\n", res.avg_ms);
    printf("    Throughput:   %.1f GFLOPS\n", res.gflops);
    printf("    Total MAC:    %.1f billion\n", 2.0f * M * K * N / 1e9f);
    
    hipError_t merr = hipMemcpy(C_gpu, res.d_C, C_bytes, hipMemcpyDeviceToHost);
    if (merr != hipSuccess) {
        fprintf(stderr, "Memcpy failed: %s\n", hipGetErrorString(merr));
    }
    verify_results_fp16("Verification vs CPU Reference", C_gpu, C_cpu, M, N);
    
    hipHostUnregister(hostPtr);
    munmap(hostPtr, total_bytes);
    close(fd);
    
    return 0;
}
