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

int main() {
    const char* filepath = "/dev/shm/test_data.bin"; // RAM-disk for speed
    //const char* filepath = "/var/tmp/test_data.bin"; // RAM-disk for speed
    size_t size = 1024 * 1024 * 100; // 100MB
    size_t n = size / sizeof(float);

    // 1. Create a file-backed memory region
    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
        perror("Failed to open file");
        return EXIT_FAILURE;
    }
    int r = ftruncate(fd, size);
    if (r != 0) {
        perror("Failed to set file size");
        close(fd);
        return EXIT_FAILURE;
    }

    // Standard CPU mmap (Pageable memory)
    void* hostPtr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // Simple kernel to modify the data
    HIP_CHECK(hipInit(0), "initializing HIP");

    // Load the Code Object file
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, "kernels.co"), "loading kernel module");

    // Extract the function (use the C-name of the kernel)
    hipFunction_t kernelFunc;
    HIP_CHECK(hipModuleGetFunction(&kernelFunc, module, "halo_kernel"), "getting kernel function");

    // Register the mmap'ed memory for GPU access
    // This 'pins' the pages in the IOMMU so the GPU can walk them.
    // hipHostRegisterMapped is the key flag for Strix Halo.
    HIP_CHECK(hipHostRegister(hostPtr, size, hipHostRegisterMapped), "registering host memory");

    // Get the Device Pointer
    // On APUs, devPtr might be equal to hostPtr, but always use this API for safety.
    float* devPtr;
    HIP_CHECK(hipHostGetDevicePointer((void**)&devPtr, hostPtr, 0), "getting device pointer");

    // Launch Kernel (No hipMemcpy needed!) 
    // Set up Kernel Arguments (This is different from GGL launch)
    // You pass an array of pointers to each argument
    void* args[] = { &devPtr, &n };
    HIP_CHECK(hipModuleLaunchKernel(kernelFunc, 
                          (n+255)/256, 1, 1, // Grid
                          256, 1, 1,         // Block
                          0, 0, args, nullptr), "launching kernel");


    // Wait for the GPU to finish. The data is updated in-place in the file.
    HIP_CHECK(hipDeviceSynchronize(), "synchronizing device");

    // Success! The data is updated in the file via the GPU.
    std::cout << "GPU finished. First element: " << ((float*)hostPtr)[0] << std::endl;
    std::cout << "GPU finished. Last element: " << ((float*)hostPtr)[n-1] << std::endl;

    // Cleanup
    HIP_CHECK(hipHostUnregister(hostPtr), "unregistering host memory");
    munmap(hostPtr, size);
    close(fd);
    return 0;
}
