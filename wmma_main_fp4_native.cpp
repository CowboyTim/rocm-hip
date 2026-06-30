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

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

// FP4 dequant helper
__host__ __forceinline__ float fp4_dequant(uint8_t nibble, float scale) {
    int8_t code = fp4_codebook[nibble];
    return (float)code * scale;
}

// FP4 quantize
__host__ __forceinline__ uint8_t fp4_quant(float val) {
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

extern "C" __global__ void wmma_matmul_fp4_builtin(const unsigned char*, const unsigned char*, float*, float);

int main() {
    printf("=== FP4 Native WMMA (gfx1151) ===\n\n");
    printf("Using: __builtin_amdgcn_wmma_f32_32x16x128_f4\n");
    printf("ROCmFP4: E2M1-derived with Codebook10\n");
    printf("Matmul: A[32x128] FP4 x B[16x128] FP4 -> C[32x16] FP32\n\n");

    // Output tile: 32x16 FP32 = 2048 bytes
    // Input A: 32x128 FP4 = 4096 bytes
    // Input B: 16x128 FP4 = 2048 bytes
    size_t A_bytes = 32 * 128;  // FP4: 1 nibble per byte
    size_t B_bytes = 16 * 128;  // FP4: 1 nibble per byte
    size_t C_bytes = 32 * 16 * sizeof(float);
    size_t total_bytes = A_bytes + B_bytes + C_bytes + C_bytes + sizeof(float);

    const char* filepath = "/dev/shm/wmma_fp4_native.bin";

    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open mmap"); return 1; }
    ftruncate(fd, total_bytes);

    void* hostPtr = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    unsigned char* A_fp4 = (unsigned char*)hostPtr;
    unsigned char* B_fp4 = (unsigned char*)((char*)hostPtr + A_bytes);
    float* C_gpu = (float*)((char*)hostPtr + A_bytes + B_bytes);
    float* C_cpu = (float*)((char*)hostPtr + A_bytes + B_bytes + C_bytes);
    float* scale_ptr = (float*)((char*)hostPtr + A_bytes + B_bytes + C_bytes + C_bytes);

    float scale_val = 1.0f;
    *scale_ptr = scale_val;

    printf("FP4 scale: %.1f\n\n", scale_val);

    // Fill A[32x128]: A[i][j] = quantize((i+1) * (j%10 + 1))
    printf("Filling A[32x128] FP4:\n");
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 128; ++j) {
            float val = (float)(i + 1) * (float)(j % 10 + 1);
            uint8_t nibble = fp4_quant(val);
            A_fp4[i * 128 + j] = nibble;
            if (j < 16 && i < 4) {
                printf("  %4.1f", (float)fp4_codebook[nibble]);
            }
        }
        printf("\n");
    }

    // Fill B[16x128]: B[i][j] = quantize((i+1) + (j%8 + 1))
    printf("\nFilling B[16x128] FP4:\n");
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 128; ++j) {
            float val = (float)(i + 1) + (float)(j % 8 + 1);
            uint8_t nibble = fp4_quant(val);
            B_fp4[i * 128 + j] = nibble;
            if (j < 16 && i < 4) {
                printf("  %4.1f", (float)fp4_codebook[nibble]);
            }
        }
        printf("\n");
    }

    memset(C_gpu, 0, C_bytes);

    // Pin mmap'ed pages for GPU access
    HIP_CHECK(hipHostRegister(hostPtr, total_bytes, hipHostRegisterDefault), "pin mmap memory");

    unsigned char* dev_A, *dev_B;
    float* dev_C;

    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_A, (void*)A_fp4, 0), "get dev ptr A");
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_B, (void*)B_fp4, 0), "get dev ptr B");
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_C, (void*)C_gpu, 0), "get dev ptr C");

    float dev_scale = scale_val;

    printf("\nMemory pointers (UMA zero-copy): dev == host: %s\n",
           dev_A == A_fp4 ? "YES" : "NO");

    // Load kernel module
    int kfd = open("wmma_kernel_fp4_native.co", O_RDONLY);
    if (kfd < 0) { perror("wmma_kernel_fp4_native.co"); return 1; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);

    HIP_CHECK(hipInit(0), "init");
    hipModule_t mod;
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    
    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, mod, "wmma_matmul_fp4_native"), "get function");

    // Launch: 1 warp (32 threads) per tile
    dim3 block(32);
    dim3 grid(1);

    printf("\nLaunching wmma_matmul_fp4_native<<<%d, %d>>>...\n", grid.x, block.x);
    void* args[] = { &dev_A, &dev_B, &dev_C, &dev_scale };
    HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1, block.x, block.y, 1,
                                    0, 0, args, nullptr), "launch");
    HIP_CHECK(hipDeviceSynchronize(), "sync after kernel");
    
    munmap(coData, st.st_size);
    close(kfd);
    HIP_CHECK(hipModuleUnload(mod), "unload");

    // Print GPU result
    printf("\nGPU Result C[32x16] = A[32x128] FP4 x B[16x128] FP4:\n");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            printf("  %8.1f", C_gpu[i * 16 + j]);
        }
        printf("\n");
    }
    printf("  ...\n");

    // CPU verification
    printf("\nCPU Verification (first 8 rows):\n");
    bool all_match = true;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 16; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 128; ++k) {
                uint8_t a_nibble = A_fp4[i * 128 + k];
                uint8_t b_nibble = B_fp4[j * 128 + k];
                sum += fp4_dequant(a_nibble, 1.0f) * fp4_dequant(b_nibble, 1.0f);
            }
            C_cpu[i * 16 + j] = sum;
        }
        printf("  Row %2d: GPU=[", i);
        for (int j = 0; j < 16; ++j) {
            printf(" %8.1f", C_gpu[i * 16 + j]);
        }
        printf(" ]  CPU=[");
        for (int j = 0; j < 16; ++j) {
            printf(" %8.1f", C_cpu[i * 16 + j]);
        }
        printf(" ]");
        
        bool row_match = true;
        for (int j = 0; j < 16; ++j) {
            if (fabsf(C_cpu[i * 16 + j] - C_gpu[i * 16 + j]) > 1.0f) {
                row_match = false;
                break;
            }
        }
        printf(" [%s]\n", row_match ? "OK" : "MISMATCH");
        if (!row_match) all_match = false;
    }
    
    if (all_match) {
        printf("\n  All values match!\n");
    }

    // Cleanup
    HIP_CHECK(hipHostUnregister(hostPtr), "unpin memory");
    munmap(hostPtr, total_bytes);
    close(fd);

    printf("\nDone.\n");
    return 0;
}
