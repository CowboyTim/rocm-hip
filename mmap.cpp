#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(command, errstr) \
{ \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "Error: %s at %s:%d, while %s\n", hipGetErrorString(status), __FILE__, __LINE__, errstr); \
        exit(EXIT_FAILURE); \
    } \
}

// Simple kernel to verify access
__global__ void halo_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // GPU writes directly to the mmap'ed file/RAM
        // This will increment each element by 1.0f, demonstrating GPU access to the memory.
        // On APUs, this is a direct pointer to the same memory the CPU sees.
        // On discrete GPUs, this is a GPU-mapped pointer that the GPU can access without copying.
        // This is the "halo" of Strix Halo: the GPU can walk the CPU's memory pages directly.
        // This avoids the need for explicit hipMemcpy and allows for zero-copy access to large datasets.
        // This is especially beneficial for workloads that require frequent CPU-GPU synchronization or large datasets that would be costly to copy.
        // In this example, we simply increment each element to demonstrate that the GPU can write to the memory. In a real application, this could be any computation that benefits from GPU acceleration while still allowing the CPU to access the results without copying.
        // Note: In a real application, you would want to ensure proper synchronization between the CPU and GPU when accessing this shared
        // memory to avoid race conditions. In this simple example, we assume that the GPU kernel runs to completion before the CPU accesses the data again.
        // This is a powerful feature of Strix Halo that enables new programming models and performance optimizations by allowing the GPU to directly access CPU memory without the overhead of copying data back and forth.
        // This is particularly useful for applications that have large datasets or require frequent updates between the CPU and GPU, as it eliminates the need for costly data transfers and allows for more efficient use of memory and computational resources.
        // In summary, this kernel demonstrates the core capability of Strix Halo: enabling the GPU to directly access and modify CPU memory through a file-backed mmap, providing a seamless and efficient way to share data between the CPU and GPU without the overhead of copying.
        // This is a key feature of Strix Halo that can significantly improve performance for certain workloads by allowing the GPU to work directly with data in CPU memory, eliminating the need for explicit data transfers and enabling new programming models that take advantage of this shared memory access.
        data[idx] += 1.0f; 
    }
}

int main() {
    const char* filepath = "/dev/shm/test_data.bin"; // RAM-disk for speed
    //const char* filepath = "/var/tmp/test_data.bin"; // RAM-disk for speed
    size_t size = 1024 * 1024 * 100; // 100MB
    size_t n = size / sizeof(float);

    // 1. Create a file-backed memory region
    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    ftruncate(fd, size);

    // Standard CPU mmap (Pageable memory)
    void* hostPtr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // 2. Register the mmap'ed memory for GPU access
    // This 'pins' the pages in the IOMMU so the GPU can walk them.
    // hipHostRegisterMapped is the key flag for Strix Halo.
    HIP_CHECK(hipHostRegister(hostPtr, size, hipHostRegisterMapped), "registering host memory");

    // 3. Get the Device Pointer
    // On APUs, devPtr might be equal to hostPtr, but always use this API for safety.
    float* devPtr;
    HIP_CHECK(hipHostGetDevicePointer((void**)&devPtr, hostPtr, 0), "getting device pointer");

    // 4. Launch Kernel (No hipMemcpy needed!)
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(halo_kernel, dim3(blocks), dim3(threads), 0, 0, devPtr, n);

    HIP_CHECK(hipDeviceSynchronize(), "synchronizing device");

    // 5. Success! The data is updated in the file via the GPU.
    std::cout << "GPU finished. First element: " << ((float*)hostPtr)[0] << std::endl;
    std::cout << "GPU finished. Last element: " << ((float*)hostPtr)[n-1] << std::endl;

    // Cleanup
    HIP_CHECK(hipHostUnregister(hostPtr), "unregistering host memory");
    munmap(hostPtr, size);
    close(fd);
    return 0;
}
