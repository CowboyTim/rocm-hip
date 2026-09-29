// fp4_verify.cpp -- ROCmFP4 Codebook10 Matmul Verification Harness
//
// CPU-side reference implementation that runs on the same mapped buffer,
// produces expected FP32 results, and compares against GPU output with
// tolerance threshold, mismatch %, and GFLOPS measurement.
//
// Usage: ./fp4_verify [M] [N] [K] [--wmma] [--dp4a] [--both]
//   Default: runs both DP4A and WMMA kernels, compares each against CPU reference

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

#define HIP_CHECK(cmd) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, #cmd); \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// ROCmFP4 Codebook10 constants (matches fp4_matmul_kernel.hip)
// ============================================================================

static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

__host__ __forceinline__ float fp4_dequant_cpu(uint8_t nibble) {
    return (float)fp4_codebook[nibble];
}

// ============================================================================
// Test matrix generation and CPU reference computation
// ============================================================================

struct FP4MatmulResult {
    std::string name;       // "DP4A" or "WMMA"  
    bool passed;            // all values within tolerance?
    float max_mismatch_pct; // maximum % mismatch vs reference
    float avg_mismatch_pct; // average % mismatch  
    int total_elements;     // M * N elements compared
    int mismatched_elements;// elements exceeding tolerance  
    double elapsed_ms;      // kernel execution time in milliseconds
};

// Initialize test matrices with known values for verification
void init_test_matrices(unsigned char* A, unsigned char* B, 
                        unsigned char* scales_A, unsigned char* scales_B, 
                        int M, int N, int K) {
    const int blockSize = 32;  // FP4 block size  
    int numBlocksK = (K + blockSize - 1) / blockSize;
    
    // Matrix A: row-major packed nibbles
    // Each value quantized based on (row+col)%16 index into codebook[0..7]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            uint8_t val = (uint8_t)((i + j) % 16); 
            A[i * K + j] = val & 0x0F;  // only low nibble matters
        }
        
        // Scales for A: UE4M3 format per block of 32 columns
        for (int blk = 0; blk < numBlocksK; ++blk) {
            float scale_val = 1.0f + 0.5f * ((i % 4 == 0) ? 1 : 0);
            
            // Encode scale as UE4M3: bias=7, 4-bit exp, 1-bit mantissa, sign bit
            int exp_raw = std::min(std::max(0, (int)(std::log2(scale_val) + 7)), 15);  
            int mant = (scale_val >= 2.0f) ? 1 : 0;
            int sign = (scale_val < 0) ? 1 : 0;
            
            scales_A[i * numBlocksK + blk] = 
                ((sign << 7) | (mant << 3)) | ((uint8_t)exp_raw & 0x0F);
            scales_A[i * numBlocksK + blk + numBlocksK] = 0x00;
        }
    }
    
    // Matrix B: packed layout  
    // Each value quantized based on (row-col+K)%16 index into codebook[0..7]
    for (int i = 0; i < K; ++i) {  
        for (int j = 0; j < N; ++j) {
            uint8_t val = (uint8_t)((i - j + K) % 16);
            B[i * N + j] = val & 0x0F;
        }
        
        // Scales for B: per block of 32 rows, one scale per column
        for (int c = 0; c < N; ++c) {
            for (int blk = 0; blk < numBlocksK; ++blk) {
                float scale_val = 1.0f + 0.5f * ((c % 3 == 0) ? 1 : 0);
                
                int exp_raw = std::min(std::max(0, (int)(std::log2(scale_val) + 7)), 15);
                int mant = (scale_val >= 2.0f) ? 1 : 0; 
                int sign = (scale_val < 0) ? 1 : 0;
                
                scales_B[blk * N + c] = 
                    ((sign << 7) | (mant << 3)) | ((uint8_t)exp_raw & 0x0F);  
                scales_B[blk * N + c + N] = 0x00;
            }
        }
    }
}

// CPU reference: C = A * B using FP4 dequant via codebook lookup
void cpu_reference_fp4(const unsigned char* A, const unsigned char* B,
                       const unsigned char* A_scales, const unsigned char* B_scales,
                       float* C, int M, int N, int K) {
    const int blockSize = 32;
    int numBlocksK = (K + blockSize - 1) / blockSize;
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            
            // Accumulate over K dimension in blocks of 32 values
            for (int blk = 0; blk < numBlocksK; ++blk) {
                int kStart = blk * blockSize;
                int kEnd = std::min(kStart + blockSize, K);
                
                // Load scale factors for this block from shared memory region
                float aScale = 1.0f;  
                if (i < M && kStart < K) {
                    uint8_t raw_scale = A_scales[i * numBlocksK + blk]; 
                    // UE4M3 decode inline: exp_raw bias=7
                    int exp_a = ((raw_scale >> 4) & 0x0F) - 7;  
                    int mant_a = raw_scale & 0x08 ? 1 : 0;  
                    int sign_a = raw_scale & 0x80 ? -1 : 1;  
                    
                    float val_a = mant_a == 0 ? 1.0f : 2.0f;  
                    val_a *= std::pow(2.0f, (float)exp_a); 
                    
                    aScale = val_a * sign_a;
                }
                
                float bScale = 1.0f;  
                if (j < N && kStart < K) {
                    uint8_t raw_scale = B_scales[blk * N + j];
                    int exp_b = ((raw_scale >> 4) & 0x0F) - 7;
                    int mant_b = raw_scale & 0x08 ? 1 : 0;
                    int sign_b = raw_scale & 0x80 ? -1 : 1;
                    
                    float val_b = mant_b == 0 ? 1.0f : 2.0f;
                    val_b *= std::pow(2.0f, (float)exp_b);  
                    
                    bScale = val_b * sign_b;  
                }
                
                // Accumulate dot products using FP4 dequant via codebook lookup
                for (int k = kStart; k < kEnd; ++k) {
                    uint8_t a_nibble = A[i * K + k] & 0x0F;
                    uint8_t b_nibble = B[k * N + j] & 0x0F;
                    
                    float a_val = fp4_dequant_cpu(a_nibble) * aScale;
                    float b_val = fp4_dequant_cpu(b_nibble) * bScale;
                    
                    sum += a_val * b_val;  
                }
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
            
            // Calculate % mismatch relative to reference magnitude
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

// ============================================================================ 
// Main test harness
// ============================================================================

int main(int argc, char* argv[]) { 
    int M = 32, N = 32, K = 64;  // Default dimensions
    
    // Parse command-line arguments  
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--M" && i + 1 < argc) M = std::atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) K = std::atoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: fp4_verify [M] [N] [K]\\n";
            std::cout << "  M,N,K: matrix dimensions (default: 32x32x64)\\n";
            std::cout << "  --help: show this message\\n";
            return 0;  
        }
    }
    
    const int blockSize = 32;  // FP4 block size for scales
    int numBlocksK = (K + blockSize - 1) / blockSize;
    
    printf("=== ROCmFP4 Codebook10 Matmul Verification Harness ===\\n");
    printf("Matrix dimensions: %d x %d x %d\\n", M, N, K);
    printf("FP4 format: Codebook10 (0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10)\\n"); 
    printf("Block size: 32 values per block, UE4M3 scale factors\\n");
    printf("Dequant path: __builtin_amdgcn_perm + sudot4 (DP4A) or LDS LUT -> WMMA\\n\\n");
    
    // Allocate unified memory via mmap in /dev/shm
    size_t total_bytes = 
        (size_t)M * K +                          // A weights (packed FP4 nibbles)  
        (size_t)K * N +                          // B weights (packed FP4 nibbles)
        (size_t)M * numBlocksK +                 // A scales (UE4M3 bytes)
        (size_t)numBlocksK * N +                 // B scales (UE4M3 bytes)  
        (size_t)M * N * sizeof(float) * 3;      // GPU result DP4A + GPU result WMMA + CPU reference
    
    const char* filepath = "/dev/shm/fp4_verify.bin";
    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open mmap"); return 1; }
    ftruncate(fd, total_bytes);
    
    void* hostPtr = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);  
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    
    // Memory layout:  
    // [0:M*K]              A weights (unsigned char), row-major packed nibbles  
    // [M*K : M*K+K*N]     B weights (unsigned char), packed layout
    // [M*K+K*N : M*K+K*N+M*numBlocksK] A scales (unsigned char/byte), UE4M3 format
    // [M*K+K*N+M*numBlocksK : M*K+K*N+M*numBlocksK+numBlocksK*N] B scales
    // [offset:offset+M*N*4] DP4A GPU result (float), row-major
    // [offset2:offset2+M*N*4] WMMA GPU result (float), row-major 
    // [offset3:offset3+M*N*4] CPU reference result (float), row-major
    
    unsigned char* weights_A = reinterpret_cast<unsigned char*>(hostPtr);
    unsigned char* weights_B = (unsigned char*)((char*)hostPtr + M * K);
    unsigned char* scales_A = reinterpret_cast<unsigned char*>(weights_B + K * N);  
    unsigned char* scales_B = reinterpret_cast<unsigned char*>(scales_A + M * numBlocksK);
    
    float* result_dp4a_gpu = (float*)(scales_B + numBlocksK * N);  
    float* result_wmma_gpu = (float*)(result_dp4a_gpu + M * N);
    float* cpu_result = (float*)(result_wmma_gpu + M * N);  
    
    // Initialize test matrices
    init_test_matrices(weights_A, weights_B, scales_A, scales_B, M, N, K);
    
    printf("Test matrices initialized\\n");
    printf("  A [%dx%d]: quantized values based on (row+col)%%16\\n", M, K);  
    printf("  B [%dx%d]: quantized values based on (row-col+K)%%16\\n", K, N);
    printf("  Scales: UE4M3 format per block of 32 values\\n\\n");
    
    // CPU reference computation (before GPU kernels run)  
    printf("Computing CPU reference...\\n");
    auto cpu_start = std::chrono::high_resolution_clock::now(); 
    cpu_reference_fp4(weights_A, weights_B, scales_A, scales_B, 
                     cpu_result, M, N, K);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("  CPU reference completed in %.2f ms\\n\\n", cpu_ms);
    
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
    
    printf("Memory pointers (UMA zero-copy):\\n");
    printf("  dev_weights_A == host_weights_A: %s\\n", 
           dev_weights_A == weights_A ? "YES" : "NO");
    printf("  dev_weights_B == host_weights_B: %s\\n",  
           dev_weights_B == weights_B ? "YES" : "NO");
    printf("  Same physical pages for CPU and GPU access\\n\\n");
    
    // ========================================================================
    // Run DP4A kernel if requested  
    // ========================================================================  
    
    bool run_dp4a = true;
    bool run_wmma = true;  
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--dp4a") { run_wmma = false; }
        else if (arg == "--wmma") { run_dp4a = false; }
    }
    
    FP4MatmulResult dp4a_result, wmma_result;
    
    if (run_dp4a) {
        printf("--- Running DP4A Kernel ---\\n");
        
        // Load pre-compiled kernel module
        int kfd = open("fp4_matmul_kernel.co", O_RDONLY);
        if (kfd < 0) { 
            perror("fp4_matmul_kernel.co not found"); 
            printf("  Skipping DP4A test -- compile fp4_matmul_kernel.hip first\\n\\n");
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
        
        // Launch kernel: grid dimension covers output tiles of 16x16  
        dim3 block(32);  // 32 threads per work-group  
        dim3 grid((M + 15) / 16, (N + 15) / 16);
        
        printf("  Launching fp4_matmul_kernel<<<%d,%d x %d>>, %d>>>...\\n", 
               grid.x, grid.y, block.x, block.y);
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        void* args[] = { 
            &dev_weights_A, &dev_weights_B, 
            &dev_scales_A, &dev_scales_B, 
            &dev_result_dp4a,
            (void*)&M, (void*)&N, (void*)&K 
        };
        
        HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1, 
                                        block.x, block.y, 1,
                                        0, 0, args, nullptr));
        HIP_CHECK(hipDeviceSynchronize()); 
        
        auto gpu_end = std::chrono::high_resolution_clock::now();
        dp4a_result.elapsed_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
        
        munmap(coData, st.st_size);  
        close(kfd);
        HIP_CHECK(hipModuleUnload(mod));
        
        printf("  DP4A kernel completed in %.2f ms\\n\\n", dp4a_result.elapsed_ms);
    } else {
        printf("--- Skipping DP4A Kernel (--wmma only) ---\\n\\n");  
        dp4a_result.passed = false;  
    }
    
    // ========================================================================
    // Run WMMA kernel if requested  
    // ========================================================================  
    
    if (run_wmma) {
        printf("--- Running WMMA Kernel ---\\n");
        
        // Load pre-compiled kernel module  
        int kfd = open("fp4_matmul_wmma.co", O_RDONLY);
        if (kfd < 0) {
            perror("fp4_matmul_wmma.co not found");
            printf("  Skipping WMMA test -- compile fp4_matmul_wmma.hip first\\n\\n"); 
            wmma_result.passed = false;
            goto check_results;
        }
        
        struct stat st; fstat(kfd, &st);
        void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
        
        HIP_CHECK(hipInit(0));
        hipModule_t mod;  
        HIP_CHECK(hipModuleLoadData(&mod, coData)); 
        
        hipFunction_t func;
        HIP_CHECK(hipModuleGetFunction(&func, mod, "fp4_matmul_kernel"));
        
        dim3 block(32);  // 32 threads per work-group  
        dim3 grid((M + 15) / 16, (N + 15) / 16);
        
        printf("  Launching fp4_matmul_wmma<<<%d,%d x %d>>, %d>>>...\\n",
               grid.x, grid.y, block.x, block.y);
                
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        void* args[] = {
            &dev_weights_A, &dev_weights_B, 
            &dev_scales_A, &dev_scales_B, 
            &dev_result_wmma,
            (void*)&M, (void*)&N, (void*)&K 
        };
        
        HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1,
                                        block.x, block.y, 1,
                                        0, 0, args, nullptr));
        HIP_CHECK(hipDeviceSynchronize()); 
        
        auto gpu_end = std::chrono::high_resolution_clock::now(); 
        wmma_result.elapsed_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
        
        munmap(coData, st.st_size);
        close(kfd);  
        HIP_CHECK(hipModuleUnload(mod));
        
        printf("  WMMA kernel completed in %.2f ms\\n\\n", wmma_result.elapsed_ms);
    } else {
        printf("--- Skipping WMMA Kernel (--dp4a only) ---\\n\\n");
        wmma_result.passed = false;
    }

check_results:  
    // ========================================================================  
    // Compare results against CPU reference and print summary table  
    // ========================================================================  
    
    if (!run_dp4a && !run_wmma) {
        printf("ERROR: No kernels specified. Use --dp4a, --wmma, or --both\\n");
        return 1;  
    }
    
    if (run_dp4a) {
        dp4a_result = compare_results("DP4A", result_dp4a_gpu, cpu_result, M, N);  
    } 
    
    if (run_wmma) {
        wmma_result = compare_results("WMMA", result_wmma_gpu, cpu_result, M, N);
    }
    
    // Print verification summary
    printf("=== Verification Results ===\\n\\n");
    
    int total_tests = 0;  
    int passed_tests = 0;
    
    if (run_dp4a) {  
        total_tests++;  
        printf("Kernel: %s\\n", dp4a_result.name.c_str());
        printf("  Status: %s\\n", dp4a_result.passed ? "PASSED" : "FAILED");
        printf("  Max mismatch: %.2f%%\\n", dp4a_result.max_mismatch_pct);
        printf("  Avg mismatch: %.2f%%\\n", dp4a_result.avg_mismatch_pct);  
        printf("  Elements compared: %d (%d mismatched)\\n", 
               dp4a_result.total_elements, dp4a_result.mismatched_elements);
        printf("  Execution time: %.2f ms\\n", dp4a_result.elapsed_ms);
        
        // Calculate GFLOPS for DP4A kernel
        double gflops = (2.0 * M * N * K) / (dp4a_result.elapsed_ms * 1e-3) / 1e9;
        printf("  Throughput: %.2f GFLOPS\\n", gflops);
        
        if (dp4a_result.passed) passed_tests++;
        printf("\\n");
    }
    
    if (run_wmma) {
        total_tests++;
        printf("Kernel: %s\\n", wmma_result.name.c_str());
        printf("  Status: %s\\n", wmma_result.passed ? "PASSED" : "FAILED");
        printf("  Max mismatch: %.2f%%\\n", wmma_result.max_mismatch_pct);
        printf("  Avg mismatch: %.2f%%\\n", wmma_result.avg_mismatch_pct);  
        printf("  Elements compared: %d (%d mismatched)\\n", 
               wmma_result.total_elements, wmma_result.mismatched_elements);
        printf("  Execution time: %.2f ms\\n", wmma_result.elapsed_ms);
        
        // Calculate GFLOPS for WMMA kernel  
        double gflops = (2.0 * M * N * K) / (wmma_result.elapsed_ms * 1e-3) / 1e9;
        printf("  Throughput: %.2f GFLOPS\\n", gflops); 
        
        if (wmma_result.passed) passed_tests++;
        printf("\\n");
    }
    
    // Overall summary
    printf("--- Summary ---\\n");
    printf("  Total tests: %d\\n", total_tests);
    printf("  Passed: %d | Failed: %d\\n", passed_tests, total_tests - passed_tests);  
    
    if (passed_tests == total_tests) {  
        printf("\\n  All kernels verified successfully against CPU reference!\\n");
    } else {
        printf("\\n  WARNING: Some kernels did not match CPU reference within tolerance.\\n");
        printf("  Check max/avg mismatch percentages above.\\n");  
    }
    
    // Cleanup  
    HIP_CHECK(hipHostUnregister(hostPtr));
    munmap(hostPtr, total_bytes);
    close(fd);
    
    printf("\\nDone.\\n");
    return (passed_tests == total_tests) ? 0 : 1;
}
