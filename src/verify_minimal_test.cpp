#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP ERROR: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

// Minimal test: load verify_minimal.co and launch kernel
int main() {
    fprintf(stderr, "Loading verify_minimal.co...\n"); fflush(stderr);

    size_t co_size = 0;
    void* coData = nullptr;

    // Try to open the .co file
    int kfd = open("verify_minimal.co", O_RDONLY);
    if (kfd < 0) {
        perror("open verify_minimal.co");
        fprintf(stderr, "FAIL: Could not open verify_minimal.co\n");
        return 1;
    }
    struct stat st; fstat(kfd, &st);
    coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    co_size = st.st_size;
    close(kfd);

    if (coData == MAP_FAILED) { perror("mmap"); return 1; }

    HIP_CHECK(hipInit(0));

    hipModule_t mod;
    HIP_CHECK(hipModuleLoadData(&mod, coData));

    munmap(coData, co_size);

    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, mod, "verify_minimal"));

    // Create minimal test data
    const int M=16,N=16,K=32;
    size_t total = (size_t)M*K*sizeof(__half)*2 + (size_t)M*N*sizeof(float) + (size_t)N*(K/16);

    char path[]="/dev/shm/verify_minimal.bin";
    int fd=open(path,O_RDWR|O_CREAT,0666); ftruncate(fd,total);
    void* base=mmap(NULL,total,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0); close(fd);

    __half*A=(__half*)base;
    unsigned char*W=(unsigned char*)((char*)base+M*K*sizeof(__half));
    unsigned char*Wscl=W+M*K*sizeof(__half);
    float*C=(float*)(Wscl+M*K*sizeof(unsigned char));

    for(int i=0;i<M*K;++i)A[i]=__half((float)(i%7)/3.0f);
    memset(W,0x00,M*K*sizeof(unsigned char)); memset(Wscl,0x60,M*K*sizeof(unsigned char));
    memset(C,0,M*N*sizeof(float));

    HIP_CHECK(hipHostRegister(base,total,hipHostRegisterDefault));

    __half*d_A=nullptr;unsigned char*d_W=nullptr;unsigned char*d_Sc=nullptr;float*d_C=nullptr;float*d_Wsf=nullptr;
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_A,A,0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_W,W,0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_Sc,Wscl,0));
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_C,C,0));

    dim3 grid((M+15)/16,(N+15)/16);
    void*args[]={&d_A,(void*)&d_W,(void*)&d_Sc,&d_Wsf,(void*)&d_C,(void*)&M,(void*)&N,(void*)&K};

    fprintf(stderr, "[5] Launch verify_minimal<<<%d,%d>, 32>>>\n",grid.x,grid.y); fflush(stderr);

    // THE CRITICAL LAUNCH
    HIP_CHECK(hipModuleLaunchKernel(func,grid.x,grid.y,1,32,1,1,0,0,args,nullptr));

    HIP_CHECK(hipDeviceSynchronize());
    printf("SUCCESS! Minimal kernel launched and executed on gfx1151!\n");

    HIP_CHECK(hipHostUnregister(base));munmap(base,total);
    close(fd);
    HIP_CHECK(hipModuleUnload(mod));

    return 0;
}
