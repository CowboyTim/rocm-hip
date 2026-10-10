#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
    
    hipError_t err = hipInit(0);
    fprintf(stderr, "[1] hipInit: %s\n", hipGetErrorString(err)); fflush(stderr);
    
    hipModule_t mod; 
    err = hipModuleLoadData(&mod, coData);
    fprintf(stderr, "[2] ModuleLoadData: %s\n", hipGetErrorString(err)); fflush(stderr);
    
    munmap(coData, co_size);
    if (err != hipSuccess) { fprintf(stderr, "FAIL: Module load failed\n"); return 1; }
    
    hipFunction_t func;
    err = hipModuleGetFunction(&func, mod, "verify_minimal");  
    fprintf(stderr, "[3] GetFunction: %s\n", hipGetErrorString(err)); fflush(stderr);
    if (err != hipSuccess) { fprintf(stderr, "FAIL: Could not get function\n"); return 1; }
    
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
    
    err=hipHostRegister(base,total,hipHostRegisterDefault);
    fprintf(stderr, "[4] HostRegister: %s\n", hipGetErrorString(err)); fflush(stderr);
    if (err!=hipSuccess){munmap(base,total);close(fd);hipModuleUnload(mod);return 1;}
    
    __half*d_A=nullptr;unsigned char*d_W=nullptr;unsigned char*d_Sc=nullptr;float*d_C=nullptr;
    hipHostGetDevicePointer((void**)&d_A,A,0);
    hipHostGetDevicePointer((void**)&d_W,W,0);  
    hipHostGetDevicePointer((void**)&d_Sc,Wscl,0);
    hipHostGetDevicePointer((void**)&d_C,C,0);
    
    dim3 grid((M+15)/16,(N+15)/16);
    void*args[]={&d_A,(void*)&d_W,(void*)&d_Sc,(void*)NULL,(void*)&M,(void*)&N,(void*)&K};
    
    fprintf(stderr, "[5] Launch verify_minimal<<<%d,%d>, 32>>>\n",grid.x,grid.y); fflush(stderr);
    
    // THE CRITICAL LAUNCH
    err=hipModuleLaunchKernel(func,grid.x,grid.y,1,32,1,1,0,0,args,nullptr);
    
    if(err!=hipSuccess)printf("LAUNCH FAIL: %s\n",hipGetErrorString(err));
    else {
        err=hipDeviceSynchronize();
        if(err!=hipSuccess)printf("SYNC FAIL: %s\n",hipGetErrorString(err));
        else {printf("SUCCESS! Minimal kernel launched and executed on gfx1151!\n");}
    }
    
    hipHostUnregister(base);munmap(base,total);close(fd);
    hipModuleUnload(mod);
    
    return (err==hipSuccess)?0:1;
}
