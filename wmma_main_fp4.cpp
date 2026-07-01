#include <iostream>
#include <iomanip>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

#define MATRIX_SIZE 16
#define MAT_SIZE (MATRIX_SIZE * MATRIX_SIZE)

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

// FP4 dequant: nibble (0-15) → float via codebook lookup
__host__ __device__ __forceinline__ float fp4_dequant(uint8_t nibble, float scale) {
    int8_t code = fp4_codebook[nibble];
    return (float)code * scale;
}

// FP4 quantize: float → nibble via nearest codebook lookup (positive values only)
// ROCmFP4 Codebook10: 8 positive values (indices 0-7), 8 negative values (indices 8-15)
__host__ __device__ __forceinline__ uint8_t fp4_quant(float val) {
    float best_dist = 1e30f;
    int best_idx = 0;
    // Only search positive half of codebook (indices 0-7)
    for (int i = 0; i < 8; ++i) {
        float diff = fabsf((float)fp4_codebook[i] - val);
        if (diff < best_dist) {
            best_dist = diff;
            best_idx = i;
        }
    }
    return (uint8_t)best_idx;
}

extern "C" __global__ void wmma_matmul_fp4(const unsigned char*, const __half*, __half*, float);

int main() {
    printf("=== FP4 WMMA Matmul Demo (gfx1151) ===\n\n");
    printf("ROCmFP4: E2M1-derived with Codebook10\n");
    printf("Values: 0, +/-1, +/-2, +/-3, +/-4, +/-6, +/-8, +/-10\n");
    printf("Packed: 1 nibble/byte (single nibble per entry)\n\n");

    size_t fp4_bytes = MAT_SIZE;  // 1 nibble per FP4 value = 256 bytes
    size_t fp16_bytes = MAT_SIZE * sizeof(__half);
    size_t total_bytes = fp4_bytes + fp16_bytes + fp16_bytes + fp16_bytes + sizeof(float);

    // Allocate unified memory via mmap
    const char* filepath = "/dev/shm/wmma_fp4.bin";

    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open mmap"); return 1; }
    ftruncate(fd, total_bytes);

    void* hostPtr = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    // Layout:
    // [0:fp4_bytes]         FP4-packed weights A (256 bytes)
    // [fp4_bytes:fp16_bytes] FP16 activations B (512 bytes)
    // [fp4+fp16:fp4+fp16]   FP16 GPU result C (512 bytes)
    // [fp4+fp16+512:fp4+fp16+512] FP16 CPU result C (512 bytes)
    // [fp4+fp16+1024:fp4+fp16+1024] FP32 scale value (4 bytes)
    unsigned char* weights_fp4 = (unsigned char*)hostPtr;
    __half* activations = (__half*)((char*)hostPtr + fp4_bytes);
    __half* result_gpu = (__half*)((char*)hostPtr + fp4_bytes + fp16_bytes);
    __half* result_cpu = (__half*)((char*)hostPtr + fp4_bytes + fp16_bytes + fp16_bytes);
    float* scale_ptr = (float*)((char*)hostPtr + fp4_bytes + fp16_bytes + fp16_bytes + fp16_bytes);

    float scale_val = 1.0f;
    *scale_ptr = scale_val;

    printf("FP4 scale: %.1f\n\n", scale_val);

  // Fill FP4 weights: A[i][j] = quantize((i % 8) + 1)
    // Kernel loads column-major: weights_fp4[lane + ele*16]
    // Fill column-major: weights_fp4[j*M + i] = A[i][j]
    printf("Filling FP4 weights (column-major):\n");
    for (int j = 0; j < MATRIX_SIZE; ++j) {
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            float val = (float)((i % 8) + 1);
            uint8_t nibble = fp4_quant(val);
            weights_fp4[j * MATRIX_SIZE + i] = nibble;
            float dq = fp4_dequant(nibble, scale_val);
            if (j < MATRIX_SIZE - 1) {
                printf("  %4.1f", dq);
            } else {
                printf("  %4.1f\n", dq);
            }
        }
    }

    // Fill FP16 activations: B[i][j] = codebook[(j % 8) + 1]
    // B stored row-major: activations[i*M + j]
    printf("\nFP16 activations B (row-major, B[i][j] = codebook[(j %% 8)+1]):\n");
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float bval = (float)((j % 8) + 1);
            uint8_t bidx = fp4_quant(bval);
            activations[i * MATRIX_SIZE + j] = __half(fp4_dequant(bidx, 1.0f));
            printf("  %4.1f", (float)activations[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    printf("\nFP16 activations B (row-major, B[i][j] = j+1):\n");
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            activations[i * MATRIX_SIZE + j] = (__half)(float)(j + 1);
            printf("  %4.1f", (float)activations[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    memset(result_gpu, 0, fp16_bytes);

    // Pin mmap'ed pages for GPU access
    HIP_CHECK(hipHostRegister(hostPtr, total_bytes, hipHostRegisterDefault), "pin mmap memory");

    unsigned char* dev_weights_fp4;
    __half* dev_activations;
    __half* dev_result;

    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_weights_fp4, (void*)weights_fp4, 0), "get dev ptr fp4");
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_activations, (void*)activations, 0), "get dev ptr act");
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_result, (void*)result_gpu, 0), "get dev ptr result");
    __half* dev_result_cpu;
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_result_cpu, (void*)result_cpu, 0), "get dev ptr cpu");

    float dev_scale = scale_val;

    printf("\nMemory pointers (UMA zero-copy): dev == host: %s\n",
           dev_weights_fp4 == weights_fp4 ? "YES" : "NO");

    // Load kernel module from pre-compiled .co file
    int kfd = open("wmma_kernel_fp4.co", O_RDONLY);
    if (kfd < 0) { perror("wmma_kernel_fp4.co"); return 1; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);

    HIP_CHECK(hipInit(0), "init");
    hipModule_t mod;
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    
    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, mod, "wmma_matmul_fp4"), "get function");

    // Launch kernel: 1 block of 32 threads
    dim3 block(32);
    dim3 grid(1);

    printf("\nLaunching wmma_matmul_fp4<<<%d, %d>>>...\n", grid.x, block.x);
    void* args[] = { &dev_weights_fp4, &dev_activations, &dev_result, &dev_scale };
    HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1, block.x, block.y, 1,
                                    0, 0, args, nullptr), "launch");
    HIP_CHECK(hipDeviceSynchronize(), "sync after kernel");
    
    munmap(coData, st.st_size);
    close(kfd);
    HIP_CHECK(hipModuleUnload(mod), "unload");

    // Read results (already in host-visible memory via mmap)
   printf("\nGPU Result C = A_fp4 x B (16x16 FP16 matrix):\n");
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("  %7.1f", (float)result_gpu[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

   // CPU verification: dequant FP4 on CPU, compute matmul
    // weights_fp4 stored column-major: weights_fp4[j*M+i] = A[i][j]
    printf("\nCPU Verification:\n");
    printf("  Dequant A_fp4[i][k] using ROCmFP4 Codebook10\n");
    printf("  Compute C[i][j] = sum_k A_fp4[i][k] x B[k][j]\n\n");
    
    bool all_match = true;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                // Column-major storage: A[i][k] = weights_fp4[k*M+i]
                uint8_t nibble = weights_fp4[k * MATRIX_SIZE + i];
                float a_val = fp4_dequant(nibble, scale_val);
                // B[k][j] from activations (row-major): activations[k*M+j]
                float b_val = (float)activations[k * MATRIX_SIZE + j];
                sum += a_val * b_val;
            }
            result_cpu[i * MATRIX_SIZE + j] = (__half)sum;
            float gpu_val = (float)result_gpu[i * MATRIX_SIZE + j];
            float diff = fabsf(gpu_val - sum);
            char status = (diff < 2.0f) ? '=' : '!';
            if (j == 0) {
                printf("  Row %2d: GPU=[", i);
            }
            printf(" %5.1f%c", gpu_val, status);
            if (j == MATRIX_SIZE - 1) {
                printf(" ]  CPU=[");
                for (int jj = 0; jj < MATRIX_SIZE; ++jj) {
                    printf(" %5.1f", (float)result_cpu[i * MATRIX_SIZE + jj]);
                }
                printf(" ]\n");
            }
            if (diff >= 2.0f) all_match = false;
        }
    }
    
    if (all_match) {
        printf("\n  All values match within tolerance!\n");
    }

    // Cleanup
    HIP_CHECK(hipHostUnregister(hostPtr), "unpin memory");
    munmap(hostPtr, total_bytes);
    close(fd);

    printf("\nDone.\n");
    return 0;
}
