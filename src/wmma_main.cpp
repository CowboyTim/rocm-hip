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

// Kernel loaded from wmma_kernel.co module

int main() {
    size_t mat_bytes = MAT_SIZE * sizeof(__half);
    size_t total_bytes = 3 * mat_bytes; // 3 matrices: A, B, C

    printf("=== WMMA Matmul Demo (gfx1151) ===\n\n");
    printf("Matrix size: %dx%d fp16\n", MATRIX_SIZE, MATRIX_SIZE);
    printf("Using __builtin_amdgcn_wmma_f16_16x16x16_f16_w32 intrinsic\n");
    printf("Zero-copy via mmap + hipHostRegisterMapped\n\n");

    // --- Allocate unified memory via mmap (RAM-disk for speed) ---
    const char* filepath = "/dev/shm/wmma_buffers.bin";

    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open mmap file"); return 1; }
    ftruncate(fd, total_bytes);

    void* hostPtr = mmap(NULL, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (hostPtr == MAP_FAILED) { perror("mmap"); close(fd); return 1; }

    // Initialize matrices on CPU side
    // A[i][j] = (float)(i + 1), B[i][j] = (float)(j + 1)
    __half* a = (__half*)hostPtr;
    __half* b = (__half*)((char*)hostPtr + mat_bytes);
    __half* c = (__half*)((char*)hostPtr + 2 * mat_bytes);

    memset(c, 0, total_bytes);

    printf("Filling matrices on CPU:\n");
    printf("  A[i][j] = i + 1  (column vectors)\n");
    printf("  B[i][j] = j + 1  (row vectors)\n\n");

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            a[i * MATRIX_SIZE + j] = (__half)(i + 1.0f);
            b[i * MATRIX_SIZE + j] = (__half)(j + 1.0f);
        }
    }

    // Print A and B
    printf("Matrix A (%dx%d):\n", MATRIX_SIZE, MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("  %5.1f", (float)a[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B (%dx%d):\n", MATRIX_SIZE, MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("  %5.1f", (float)b[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // Pin mmap'ed pages for GPU access
    // Use hipHostRegisterDefault (more portable than hipHostRegisterMapped)
    HIP_CHECK(hipHostRegister(hostPtr, total_bytes, hipHostRegisterDefault), "pin mmap memory");

    // Get device pointer (may equal hostPtr on APU)
    __half* devA, *devB, *devC;
    HIP_CHECK(hipHostGetDevicePointer((void**)&devA, (void*)a, 0), "get dev ptr A");
    HIP_CHECK(hipHostGetDevicePointer((void**)&devB, (void*)b, 0), "get dev ptr B");
    HIP_CHECK(hipHostGetDevicePointer((void**)&devC, (void*)c, 0), "get dev ptr C");

    printf("\nMemory pointers:\n");
    printf("  host A: %p\n", (void*)a);
    printf("  dev  A: %p\n", (void*)devA);
    printf("  (dev == host on Strix Halo UMA: %s)\n", devA == a ? "YES" : "NO");

    // Load kernel module from pre-compiled .co file
    int kfd = open("wmma_kernel.co", O_RDONLY);
    if (kfd < 0) { perror("wmma_kernel.co"); return 1; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);

    HIP_CHECK(hipInit(0), "init");
    hipModule_t mod;
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");

    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, mod, "wmma_matmul"), "get function");

    // Launch kernel: 1 block of 32 threads
    // WMMA 16x16x16 on RDNA3 requires 32 threads to cover all 16 result rows
    dim3 block(32);
    dim3 grid(1);

    printf("\nLaunching wmma_matmul<<<%d, %d>>>...\n", grid.x, block.x);
    void* args[] = { &devA, &devB, &devC };
    HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1, block.x, block.y, 1,
                                    0, 0, args, nullptr), "launch");
    HIP_CHECK(hipDeviceSynchronize(), "sync after kernel");

    munmap(coData, st.st_size);
    close(kfd);
    HIP_CHECK(hipModuleUnload(mod), "unload");

    // Read results back (already in host-visible memory via mmap)
    printf("\nResult C = A * B (%dx%d) from GPU WMMA:\n", MATRIX_SIZE, MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("  %7.1f", (float)c[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // Verification: C[i][j] = sum_k A[i][k] * B[k][j]
    // For our matrices: A[i][k]=i+1, B[k][j]=j+1
    // C[i][j] = (j+1) * sum_k (i+1) = (j+1) * 16 * (i+1)
    printf("\nVerification (C[i][j] should equal (i+1)*(j+1)*16):\n");
    bool all_match = true;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            float expected = (float)(i + 1) * (j + 1) * 16;
            float actual = (float)c[i * MATRIX_SIZE + j];
            if (fabs(expected - actual) > 0.1f) {
                printf("  MISMATCH C[%d][%d]: expected %.1f, got %.1f\n", i, j, expected, actual);
                all_match = false;
            }
        }
    }
    if (all_match) {
        printf("  All values match expected results!\n");
    }

    // Cleanup
    HIP_CHECK(hipHostUnregister(hostPtr), "unpin memory");
    munmap(hostPtr, total_bytes);
    close(fd);

    printf("\nDone.\n");
    return 0;
}
