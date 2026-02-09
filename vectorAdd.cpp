#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define HIP_CHECK(command) \
{ \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "Error: %s at %s:%d\n", hipGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel declaration: __global__ signifies a function called from the host and run on the device
__global__ void vectorAdd(const float* A, const float* B, float* C, size_t N) {
    // Calculate the global thread ID
    size_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    // Perform the addition if the thread ID is within the bounds of the array
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    size_t N = 100 * 1024 * 1024; // Size of the vectors
    size_t nr_bytes = N * sizeof(float);

    // Host (CPU) memory allocation
    std::vector<float> h_A(N), h_B(N), h_C(N);

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    printf("Vector Addition using HIP\n");

    // Device (GPU) memory allocation
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    HIP_CHECK(hipMalloc(&d_A, nr_bytes));
    printf("Allocated device memory for vector A\n");
    HIP_CHECK(hipMalloc(&d_B, nr_bytes));
    printf("Allocated device memory for vector B\n");
    HIP_CHECK(hipMalloc(&d_C, nr_bytes));
    printf("Allocated device memory for vector C\n");


    // Copy data from host to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), nr_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), nr_bytes, hipMemcpyHostToDevice));


    // Configure kernel launch parameters
    size_t blockSize = 256;
    size_t gridSize = (N + blockSize - 1) / blockSize;

    printf("Launching kernel...\n");

    // Launch the kernel with the specified grid and block dimensions
    // The triple chevron syntax <<<gridSize, blockSize>>> is used to pass launch configuration
    hipLaunchKernelGGL(vectorAdd, dim3(gridSize), dim3(blockSize), 0, 0, d_A, d_B, d_C, N);

    // Synchronize to ensure the kernel completes before proceeding on the host
    HIP_CHECK(hipDeviceSynchronize());

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, nr_bytes, hipMemcpyDeviceToHost));

    // Verify results
    for (size_t i = 0; i < N; ++i) {
        if (h_C[i] != 3.0f) {
            std::cerr << "Verification failed at index " << i << std::endl;
            break;
        }
    }
    std::cout << "Verification successful!" << std::endl;

    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}
