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

// FP4 matmul on CPU
void fp4_matmul_cpu(const unsigned char* A, const unsigned char* B, float* C,
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

// FP4 matmul kernel
__global__ void fp4_matmul_kernel(const unsigned char* A, const unsigned char* B,
                                   float* C, int M, int N, int K, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    for (int r = row; r < M; r += blockDim.x * gridDim.x) {
        for (int c = col; c < N; c += blockDim.y * gridDim.y) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                uint8_t a_nibble = A[r * K + k];
                uint8_t b_nibble = B[k * N + c];
                sum += (float)fp4_codebook[a_nibble] * (float)fp4_codebook[b_nibble] * scale;
            }
            C[r * N + c] = sum;
        }
    }
}

int main() {
    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int NUM_RUNS = 10;
    const int NUM_WARMUP = 3;
    const float scale_val = 1.0f;
    
    printf("==========================================================\n");
    printf("  FP4 Matmul Benchmark (gfx1151)\n");
    printf("==========================================================\n");
    printf("  Matrix size:  %d x %d x %d\n", M, K, N);
    printf("  FP4 format:   ROCmFP4 Codebook10\n");
    printf("  Values:       0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10\n");
    printf("  Runs:         %d (after %d warmup)\n", NUM_RUNS, NUM_WARMUP);
    printf("  Total MAC:    %.1f million\n", 2.0f * M * K * N / 1e6f);
    printf("==========================================================\n\n");
    
    // Allocate host memory
    size_t matBytes = M * K;
    size_t resultBytes = N * sizeof(float);
    size_t totalBytes = 3 * matBytes + resultBytes;
    
    const char* filepath = "/dev/shm/fp4_bench.bin";
    
    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return 1; }
    ftruncate(fd, totalBytes);
    
    void* hostPtr = mmap(NULL, totalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }
    
    unsigned char* A_fp4 = (unsigned char*)hostPtr;
    unsigned char* B_fp4 = (unsigned char*)((char*)hostPtr + matBytes);
    float* C_gpu = (float*)((char*)hostPtr + 2 * matBytes);
    
    // Generate random FP4 matrices
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
    
    // Allocate device memory
    unsigned char* d_A = NULL, *d_B = NULL;
    float* d_C = NULL;
    
    printf("Allocating device memory...\n");
    hipError_t err = hipMalloc(&d_A, matBytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_A failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    err = hipMalloc(&d_B, matBytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_B failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    err = hipMalloc(&d_C, resultBytes);
    if (err != hipSuccess) {
        printf("  hipMalloc d_C failed: %s\n", hipGetErrorString(err));
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
    err = hipMemcpy(d_B, B_fp4, matBytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        printf("  hipMemcpy B failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    printf("  Copy done.\n\n");
    
    // Run kernel
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    printf("Running FP4 matmul kernel...\n");
    err = hipMemset(d_C, 0, resultBytes);
    if (err != hipSuccess) {
        printf("  hipMemset failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    
    // Warmup
    for (int w = 0; w < NUM_WARMUP; ++w) {
        hipMemset(d_C, 0, resultBytes);
        fp4_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, scale_val);
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("  Kernel launch failed: %s\n", hipGetErrorString(err));
            return 1;
        }
    }
    printf("  Warmup done.\n\n");
    
    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < NUM_RUNS; ++r) {
        hipMemset(d_C, 0, resultBytes);
        fp4_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, scale_val);
        err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            printf("  Kernel failed: %s\n", hipGetErrorString(err));
            return 1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_RUNS;
    double gflops = 2.0 * M * K * N / avg_time / 1e6;
    
    printf("  Results:\n");
    printf("    Avg time:   %.3f ms\n", avg_time);
    printf("    Throughput: %.1f GFLOPS\n", gflops);
    printf("==========================================================\n");
    
    // Copy results back
    err = hipMemcpy(C_gpu, d_C, resultBytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        printf("  hipMemcpy back failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    munmap(hostPtr, totalBytes);
    close(fd);
    
    printf("\nDone.\n");
    return 0;
}
