#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <hip/hip_runtime.h>

#define WIDTH 800
#define HEIGHT 600
#define BUF_SIZE WIDTH * HEIGHT * 4

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s = (cmd); \
        if (s != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

int main() {
    const char* filepath = "/dev/shm/mandelbrot.bin";

    int fd = open(filepath, O_RDWR | O_CREAT, 0666);
    ftruncate(fd, BUF_SIZE);
    unsigned char* hostPtr = (unsigned char*)mmap(NULL, BUF_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    memset(hostPtr, 0, BUF_SIZE);

    // Load kernel module from pre-compiled .co file
    int kfd = open("kernels.co", O_RDONLY);
    if (kfd < 0) { perror("kernels.co"); return 1; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);

    hipModule_t mod;
    HIP_CHECK(hipInit(0), "init");
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    
    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, mod, "mandelbrot_kernel"), "get function");

    // Pin mmap'ed pages for GPU access — the UMA trick!
    HIP_CHECK(hipHostRegister(hostPtr, BUF_SIZE, hipHostRegisterMapped), "pin memory");
    
    unsigned char* devPtr;
    HIP_CHECK(hipHostGetDevicePointer((void**)&devPtr, hostPtr, 0), "get dev ptr");

    // Standard Mandelbrot viewport: center at (-0.5, 0), zoom to show the full set
    float cr0 = -0.5f, ci0 = 0.0f, zoom_val = 2.8f;

    printf("Rendering Mandelbrot %dx%d (center=%.3f+%.3fi)\n", 
           WIDTH, HEIGHT, cr0, ci0);
    printf("%.1f MPixels — thread-per-pixel on unified memory\n", (WIDTH*HEIGHT)/1e6f);

    // Launch 16x8 block grid to cover the image
    dim3 block(16, 8);
    dim3 grid((WIDTH+block.x-1)/block.x, (HEIGHT+block.y-1)/block.y);

    int w = WIDTH, h = HEIGHT;
    void* args[] = { &cr0, &ci0, &zoom_val, &devPtr, &w, &h };
    HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1, block.x, block.y, 1, 
                                    0, 0, args, nullptr), "launch");
    HIP_CHECK(hipDeviceSynchronize(), "sync");

    // --- Verification of known Mandelbrot set points ---
    auto check = [&](float cr, float ci, const char* label) {
        // Map complex coord back to pixel index and read color
        int px = (int)((cr - cr0 + 0.5f * zoom_val) / zoom_val * (WIDTH-1));
        int py = (int)((ci - ci0 + 0.5f * zoom_val) / zoom_val * (HEIGHT-1));
        if (px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT) {
            printf("  %-24s @ (%d,%d): out of bounds\n", label, px, py);
            return;
        }
        int idx = (py * WIDTH + px) * 4;
        bool dark = hostPtr[idx+0] < 32 && hostPtr[idx+1] < 32 && hostPtr[idx+2] < 64;
        printf("  %-24s @ (%d,%d): %3d/%3d/%3d [%s]\n", 
               label, px, py, hostPtr[idx], hostPtr[idx+1], hostPtr[idx+2],
               dark ? "INSIDE SET" : "outside");
    };

    printf("\n=== Verification (known Mandelbrot geometry) ===\n");
    
    check(-0.5f,   0.0f,       "Center (-0.5 + 0i)");      // main cardioid — dark
    check(-0.75f,  0.0f,       "Cardioid (-0.75)");        // cusp of main cardioid — dark
    check(-1.0f,   0.0f,       "Needle tip (-1.0)");       // left boundary — moderate iterations
    check(-0.869249, 0.275811,"Seahorse (real+0.27i)");     // fine details region — bright
    check(0.3f,    0.0f,       "Right side (0.3)");         // way outside set — bright
    check(-0.74601, 0.11468,"Tiny filament near cardioid"); // extreme boundary — interesting colors

    printf("\nWriting mandelbrot.ppm...\n");
    
    FILE* f = fopen("mandelbrot.ppm", "wb");
    if (!f) { perror("write ppm"); return 1; }
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    
    unsigned char tmp[WIDTH * 3];
    for (int y = 0; y < HEIGHT; y++) {
        int src = y * WIDTH * 4;
        int dst = 0;
        for (int x = 0; x < WIDTH; x++) {
            // PPM format: RGB only — our buffer is RGBA so skip alpha (byte[3])
            tmp[dst++] = hostPtr[src++];     // R
            tmp[dst++] = hostPtr[src++];     // G
            tmp[dst++] = hostPtr[src++];     // B
        }
        fwrite(tmp, 1, WIDTH * 3, f);
    }
    fclose(f);

     HIP_CHECK(hipModuleUnload(mod), "unload");
    HIP_CHECK(hipHostUnregister(hostPtr), "unpin");
    munmap(hostPtr, BUF_SIZE); 
    munmap(coData, st.st_size);
    close(fd); close(kfd);
    
    printf("Done! Open mandelbrot.ppm to see the fractal.\n");
    return 0;
}
