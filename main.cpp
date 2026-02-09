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

// Declare the kernel from the other file
extern "C" __global__ void halo_kernel(float* data, size_t n);

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
