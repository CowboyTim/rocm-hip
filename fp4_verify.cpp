// fp4_verify.cpp -- ROCmFP4 Codebook10 Matmul Verification Harness
//
// CPU-side reference implementation that runs on the same mapped buffer as the GPU kernels,
// produces expected FP32 results, and compares against GPU output with tolerance threshold,
// mismatch %, and GFLOPS measurement. Zero-copy: everything lives in one /dev/shm mapping.
//
// Data contract (fp4_common.h):
//   A: M packed lines of K values (2 nibbles/byte, low first)          -> M*(K/2) bytes
//   B: N packed lines of K values (line j = column j of B)             -> N*(K/2) bytes
//   A_scales[i][s], B_scales[j][s]: UE4M3 byte per 16-elem half-block  -> *(K/16) bytes each
//   C: M x N float32 row-major per kernel result + cpu reference       -> M*N*4 bytes x 3
//
// Usage: ./fp4_verify [--M m] [--N n] [--K k] [--dp4a|--wmma|--both]
//   Default: both kernels at 32x32x64

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include "fp4_common.h"

using namespace rocmfp4;

#define HIP_CHECK(cmd) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, #cmd); \
            exit(1); \
        } \
    } while(0)

struct FP4MatmulResult {
    std::string name;       // "DP4A" or "WMMA"
    bool passed;            // all values within tolerance?
    float max_mismatch_pct; // maximum % mismatch vs reference
    float avg_mismatch_pct; // average % mismatch
    int total_elements;     // M * N elements compared
    int mismatched_elements;// elements exceeding tolerance
    double elapsed_ms;      // kernel execution time in milliseconds
};

// Encode a positive scale into canonical UE4M3 bits by brute-force matching the decoder.
static uint8_t encode_ue4m3(float target) {
    for (int e = 1; e <= 15; e++) {
        for (int man = 0; man <= 7; man++) {
            uint8_t enc = (uint8_t)((e << 3) | man);
            if (std::fabs(ue4m3_to_float(enc) - target) < 1e-6f) return enc;
        }
    }
    fprintf(stderr, "ERROR: no UE4M3 encoding for %.6g\n", target);
    exit(1);
}

// Initialize test matrices with known values covering all 16 Codebook10 entries.
void init_test_matrices(unsigned char* A, unsigned char* B,
                        unsigned char* scales_A, unsigned char* scales_B,
                        int M, int N, int K) {
    int numStrips = K / HALF_BLOCK;

    // A line i: value at k -> nibble (i + k) % 16 ; packed two per byte
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; k += 2) {
            uint8_t lo = (uint8_t)((i + k) % 16);
            uint8_t hi = (uint8_t)((i + k + 1) % 16);
            A[i * (K / NIBBLES_BYTE) + k / 2] = (uint8_t)(lo | (hi << 4));
        }
        for (int s = 0; s < numStrips; ++s) {
            float scale_val = 1.0f + 0.5f * ((i % 4 == 0 && s % 2 == 1) ? 1 : 0);
            scales_A[i * numStrips + s] = encode_ue4m3(scale_val);
        }
    }

    // B line j (= column j): value at k -> nibble (3*k + j) % 16
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; k += 2) {
            uint8_t lo = (uint8_t)(((3 * k) + j) % 16);
            uint8_t hi = (uint8_t)(((3 * k + 1) + j) % 16);
            B[j * (K / NIBBLES_BYTE) + k / 2] = (uint8_t)(lo | (hi << 4));
        }
        for (int s = 0; s < numStrips; ++s) {
            float scale_val = 1.0f + 0.5f * ((j % 3 == 0 && s % 2 == 0) ? 1 : 0);
            scales_B[j * numStrips + s] = encode_ue4m3(scale_val);
        }
    }
}

// CPU reference: C[i][j] = sum_k deq(A[i,k])*aS[i][k/16] * deq(B[j,k])*bS[j][k/16]
void cpu_reference_fp4(const unsigned char* A, const unsigned char* B,
                       const unsigned char* A_scales, const unsigned char* B_scales,
                       float* C, int M, int N, int K) {
    int numStrips = K / HALF_BLOCK;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float aVal = dequant_code(packed_nibble(A + i * (K / NIBBLES_BYTE), k)) *
                             ue4m3_to_float(A_scales[i * numStrips + k / HALF_BLOCK]);
                float bVal = dequant_code(packed_nibble(B + j * (K / NIBBLES_BYTE), k)) *
                             ue4m3_to_float(B_scales[j * numStrips + k / HALF_BLOCK]);
                sum += aVal * bVal;
            }
            C[i * N + j] = sum;
        }
    }
}

// Compare GPU result against CPU reference with tolerance and percentage metrics
FP4MatmulResult compare_results(const std::string& name,
                                float* gpu_result, float* cpu_result,
                                int M, int N) {
    FP4MatmulResult res;
    res.name = name;

    double max_mismatch_pct = 0.0f;
    double avg_mismatch_pct = 0.0f;
    int total_elements = M * N;
    int mismatched_elements = 0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float gpu_val = gpu_result[i * N + j];
            float ref_val = cpu_result[i * N + j];

            float abs_ref = std::abs(ref_val);
            float abs_diff = std::abs(gpu_val - ref_val);

            float pct_mismatch = (abs_ref > 1e-6f) ?
                (abs_diff / abs_ref * 100.0f) : abs_diff;

            if (pct_mismatch > max_mismatch_pct) {
                max_mismatch_pct = pct_mismatch;
            }

            avg_mismatch_pct += pct_mismatch;

            // Tolerance: 5% mismatch or absolute difference > 2.0f
            const float tolerance_pct = 5.0f;
            const float tolerance_abs = 2.0f;

            if (pct_mismatch > tolerance_pct || abs_diff > tolerance_abs) {
                mismatched_elements++;
                if (getenv("FP4_DUMP_MISMATCH") && mismatched_elements <= 12) {
                    printf("    [%d,%d] gpu=%.6f ref=%.6f diff=%+.6f\n", i, j, gpu_val, ref_val, gpu_val - ref_val);
                }
            }
        }
    }

    res.max_mismatch_pct = max_mismatch_pct;
    res.avg_mismatch_pct = avg_mismatch_pct / total_elements;
    res.total_elements = total_elements;
    res.mismatched_elements = mismatched_elements;
    res.passed = (mismatched_elements == 0);

    return res;
}

int main(int argc, char* argv[]) {
    int M = 32, N = 32, K = 64;   // Default dimensions (K must be a multiple of 32)

    bool run_dp4a = true, run_wmma = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--dp4a")      { run_dp4a = true;  run_wmma = false; }
        else if (arg == "--wmma")      { run_dp4a = false; run_wmma = true; }
        else if (arg == "--both")      { run_dp4a = true;  run_wmma = true; }
        else if (arg == "--help") {
            std::cout << "Usage: ./fp4_verify [--M m] [--N n] [--K k] [--dp4a|--wmma|--both]\n";
            return 0;
        }
    }

    const int numStrips = K / HALF_BLOCK;

    printf("=== ROCmFP4 Codebook10 Matmul Verification Harness ===\n");
    printf("Matrix dimensions: %d x %d x %d\n", M, N, K);
    printf("FP4 format: Codebook10 (0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10)\n");
    printf("Layout: packed 2 nibbles/byte + UE4M3 scale per 16-elem half-block (%d strips)\n", numStrips);
    printf("Dequant path: __builtin_amdgcn_perm chain -> sudot4 (DP4A) or LUT->FP16 WMMA\n\n");

    // Allocate unified memory via mmap in /dev/shm
    size_t bytes_A     = (size_t)M * (K / NIBBLES_BYTE);
    size_t bytes_B     = (size_t)N * (K / NIBBLES_BYTE);
    size_t bytes_Asc   = (size_t)M * numStrips;
    size_t bytes_Bsc   = (size_t)N * numStrips;
    size_t total_bytes = bytes_A + bytes_B + bytes_Asc + bytes_Bsc + (size_t)M * N * sizeof(float) * 3;

    const char* filepath = "/dev/shm/fp4_verify.bin";
    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open mmap"); return 1; }
    ftruncate(fd, total_bytes);

    void* hostPtr = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    // Memory layout: [A][B][A_scales][B_scales][C_dp4a][C_wmma][C_cpu_ref]
    unsigned char* weights_A   = reinterpret_cast<unsigned char*>(hostPtr);
    unsigned char* weights_B   = weights_A + bytes_A;
    unsigned char* scales_A    = weights_B + bytes_B;
    unsigned char* scales_B    = scales_A + bytes_Asc;

    float* result_dp4a_gpu = (float*)(scales_B + bytes_Bsc);
    float* result_wmma_gpu = result_dp4a_gpu + M * N;
    float* cpu_result      = result_wmma_gpu + M * N;

    init_test_matrices(weights_A, weights_B, scales_A, scales_B, M, N, K);

    printf("Test matrices initialized\n");
    printf("  A [%dx%d]: nibbles (i+k)%%16, packed lines of %d bytes\n", M, K, K / NIBBLES_BYTE);
    printf("  B [%dx%d lines x %d values]: nibbles (3k+j)%%16, packed\n", N, K, K / NIBBLES_BYTE);
    printf("  Scales: UE4M3 per 16-elem half-block\n\n");

    // CPU reference computation (before GPU kernels run)
    printf("Computing CPU reference...\n");
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_reference_fp4(weights_A, weights_B, scales_A, scales_B, cpu_result, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("  CPU reference completed in %.2f ms\n\n", cpu_ms);

    // Pin mapped memory for GPU access
    HIP_CHECK(hipHostRegister(hostPtr, total_bytes, hipHostRegisterDefault));

    unsigned char* dev_weights_A;
    unsigned char* dev_weights_B;
    unsigned char* dev_scales_A;
    unsigned char* dev_scales_B;
    float* dev_result_dp4a;
    float* dev_result_wmma;

    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_weights_A, (void*)weights_A, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_weights_B, (void*)weights_B, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_scales_A, (void*)scales_A, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_scales_B, (void*)scales_B, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_result_dp4a, (void*)result_dp4a_gpu, 0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_result_wmma, (void*)result_wmma_gpu, 0));

    printf("Memory pointers (UMA zero-copy):\n");
    printf("  dev_weights_A == host_weights_A: %s\n",
           dev_weights_A == weights_A ? "YES" : "NO");
    printf("  dev_weights_B == host_weights_B: %s\n",
           dev_weights_B == weights_B ? "YES" : "NO");
    printf("  Same physical pages for CPU and GPU access\n\n");

    dim3 block(32);   // 32 threads per work-group
    dim3 grid((M + 15) / 16, (N + 15) / 16);

    FP4MatmulResult dp4a_result{"DP4A", false, 0.f, 0.f, M * N, 0, 0.0};
    FP4MatmulResult wmma_result{"WMMA", false, 0.f, 0.f, M * N, 0, 0.0};

    void* args[] = {
        &dev_weights_A, &dev_weights_B,
        &dev_scales_A, &dev_scales_B,
        nullptr,                     // C pointer slot filled per kernel below
        (void*)&M, (void*)&N, (void*)&K
    };

    if (run_dp4a) {
        printf("--- Running DP4A Kernel ---\n");

        int kfd = open("fp4_matmul_kernel.co", O_RDONLY);
        if (kfd < 0) {
            perror("fp4_matmul_kernel.co not found");
            printf("  Skipping DP4A test -- compile fp4_matmul_kernel.hip first\n\n");
            dp4a_result.passed = false;
            goto check_results;
        }

        struct stat st; fstat(kfd, &st);
        void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);

        HIP_CHECK(hipInit(0));
        hipModule_t mod;
        HIP_CHECK(hipModuleLoadData(&mod, coData));

        hipFunction_t func;
        HIP_CHECK(hipModuleGetFunction(&func, mod, "fp4_matmul_kernel"));

        args[4] = &dev_result_dp4a;

        printf("  Launching fp4_matmul_kernel<<<%d,%d x %d>>, %d>>>...\n",
               grid.x, grid.y, block.x, block.y);

        auto gpu_start = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1,
                                        block.x, block.y, 1,
                                        0, 0, args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        auto gpu_end = std::chrono::high_resolution_clock::now();

        dp4a_result.elapsed_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        munmap(coData, st.st_size);
        close(kfd);
        HIP_CHECK(hipModuleUnload(mod));

        printf("  DP4A kernel completed in %.2f ms\n\n", dp4a_result.elapsed_ms);
    } else {
        printf("--- Skipping DP4A Kernel (--wmma only) ---\n");
        dp4a_result.passed = false;
    }

    if (run_wmma) {
        printf("--- Running WMMA Kernel ---\n");

        int kfd = open("fp4_matmul_wmma.co", O_RDONLY);
        if (kfd < 0) {
            perror("fp4_matmul_wmma.co not found");
            printf("  Skipping WMMA test -- compile fp4_matmul_wmma.hip first\n\n");
            wmma_result.passed = false;
            goto check_results;
        }

        struct stat st; fstat(kfd, &st);
        void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);

        HIP_CHECK(hipInit(0));
        hipModule_t mod;
        HIP_CHECK(hipModuleLoadData(&mod, coData));

        hipFunction_t func;
        HIP_CHECK(hipModuleGetFunction(&func, mod, "fp4_matmul_wmma"));

        args[4] = &dev_result_wmma;

        printf("  Launching fp4_matmul_wmma<<<%d,%d x %d>>, %d>>>...\n",
               grid.x, grid.y, block.x, block.y);

        auto gpu_start = std::chrono::high_resolution_clock::now();
        HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1,
                                        block.x, block.y, 1,
                                        0, 0, args, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        auto gpu_end = std::chrono::high_resolution_clock::now();

        wmma_result.elapsed_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

        munmap(coData, st.st_size);
        close(kfd);
        HIP_CHECK(hipModuleUnload(mod));

        printf("  WMMA kernel completed in %.2f ms\n\n", wmma_result.elapsed_ms);
    } else {
        printf("--- Skipping WMMA Kernel (--dp4a only) ---\n");
        wmma_result.passed = false;
    }

check_results:
    if (!run_dp4a && !run_wmma) {
        printf("ERROR: No kernels specified. Use --dp4a, --wmma, or --both\n");
        return 1;
    }

    if (run_dp4a) {
        dp4a_result = compare_results("DP4A", result_dp4a_gpu, cpu_result, M, N);
        dp4a_result.elapsed_ms = std::max(dp4a_result.elapsed_ms, 0.001);
        double gflops = (2.0 * M * N * K / 1e9) / (dp4a_result.elapsed_ms / 1e3);
        printf("Kernel: %s\n", dp4a_result.name.c_str());
        printf("  Status: %s\n", dp4a_result.passed ? "PASSED" : "FAILED");
        printf("  Max mismatch: %.2f%%\n", dp4a_result.max_mismatch_pct);
        printf("  Avg mismatch: %.2f%%\n", dp4a_result.avg_mismatch_pct);
        printf("  Elements compared: %d (%d mismatched)\n", dp4a_result.total_elements, dp4a_result.mismatched_elements);
        printf("  Execution time: %.2f ms\n", dp4a_result.elapsed_ms);
        printf("  Throughput: %.2f GFLOPS\n\n", gflops);
    }

    if (run_wmma) {
        wmma_result = compare_results("WMMA", result_wmma_gpu, cpu_result, M, N);
        wmma_result.elapsed_ms = std::max(wmma_result.elapsed_ms, 0.001);
        double gflops = (2.0 * M * N * K / 1e9) / (wmma_result.elapsed_ms / 1e3);
        printf("Kernel: %s\n", wmma_result.name.c_str());
        printf("  Status: %s\n", wmma_result.passed ? "PASSED" : "FAILED");
        printf("  Max mismatch: %.2f%%\n", wmma_result.max_mismatch_pct);
        printf("  Avg mismatch: %.2f%%\n", wmma_result.avg_mismatch_pct);
        printf("  Elements compared: %d (%d mismatched)\n", wmma_result.total_elements, wmma_result.mismatched_elements);
        printf("  Execution time: %.2f ms\n", wmma_result.elapsed_ms);
        printf("  Throughput: %.2f GFLOPS\n\n", gflops);
    }

    int total_tests = (run_dp4a ? 1 : 0) + (run_wmma ? 1 : 0);
    int passed_tests = (run_dp4a && dp4a_result.passed ? 1 : 0) +
                       (run_wmma && wmma_result.passed ? 1 : 0);

    printf("--- Summary ---\n");
    printf("  Total tests: %d\n", total_tests);
    printf("  Passed: %d | Failed: %d\n", passed_tests, total_tests - passed_tests);

    if (passed_tests == total_tests) {
        printf("\n  ALL KERNELS MATCH CPU REFERENCE -- Phase 2 verified on gfx1151\n");
    } else {
        printf("\n  WARNING: Some kernels did not match CPU reference within tolerance.\n");
        printf("  Check max/avg mismatch percentages above.\n");
    }

    printf("\nDone.\n");
    return (passed_tests == total_tests) ? 0 : 1;
}
