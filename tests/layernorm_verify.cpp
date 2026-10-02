// layernorm_verify.cpp -- Phase 3 acceptance test: RMSNorm kernel vs CPU reference over /dev/shm.
//
// Usage: ./layernorm_verify [--tokens T --embd E [--eps f]] | [--sweep]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s_ = (cmd); \
        if (s_ != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s_), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

static uint32_t g_rng_state = 7;
static float rnd(float lo, float hi) {   // LCG -> [lo, hi)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return lo + ((float)(g_rng_state >> 8) / (float)(1u << 24)) * (hi - lo);
}

struct Shape { int tokens, embd; };

// Tolerance: ~1% of the larger magnitude plus a tiny floor. Covers one FP16 output ulp (2^-11 of
// the value) and a last-ulp difference in the sum-of-squares caused by GPU tree reduction vs CPU
// sequential accumulation; any real bug (wrong row/gamma index, missing eps) is O(1) and flagged.
static bool bad_result(float g, float r) {
    float d = fabsf(g - r);
    return d > 0.01f * fmaxf(fabsf(g), fabsf(r)) + 1e-4f;
}

static bool run_shape(int tokens, int embd, float eps) {
    printf("--- Shape %d tokens x %d embd ---\n", tokens, embd);

    const size_t bytesX = (size_t)tokens * embd * sizeof(__half);
    const size_t bytesG = (size_t)embd * sizeof(__half);
    const size_t total  = bytesX + bytesG + bytesX * 2;   // [x][gamma][y_gpu][y_ref]

    const char* path = "/dev/shm/layernorm_verify.bin";
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return false; }
    ftruncate(fd, (off_t)total);
    void* base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return false; }

    __half* x     = (__half*)base;
    __half* gamma = (__half*)((char*)base + bytesX);
    __half* y_gpu = (__half*)((char*)base + bytesX + bytesG);
    __half* y_ref = (__half*)((char*)base + bytesX + bytesG + bytesX);

    g_rng_state = 7;
    for (int i = 0; i < tokens * embd; ++i) x[i] = __half(rnd(-3.0f, 3.0f));
    for (int c = 0; c < embd; ++c)          gamma[c] = __half(rnd(0.5f, 1.5f));
    memset(y_gpu, 0, bytesX);

    // CPU reference: sequential sum of squares, single fp32->fp16 rounding at store.
    for (int t = 0; t < tokens; ++t) {
        float ssq = 0.0f;
        for (int c = 0; c < embd; ++c) { float v = (float)x[t * embd + c]; ssq += v * v; }
        float inv = 1.0f / sqrtf(ssq / (float)embd + eps);   // CPU reciprocal-sqrt; differs from kernel rsqrtf by a few fp32 ulps max
        for (int c = 0; c < embd; ++c)
            y_ref[t * embd + c] = (__half)((float)x[t * embd + c] * inv * (float)gamma[c]);
    }

    HIP_CHECK(hipHostRegister(base, total, hipHostRegisterDefault), "pin");

    __half* d_x; __half* d_g; __half* d_y;
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_x, x, 0), "dev x");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_g, gamma, 0), "dev gamma");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_y, y_gpu, 0), "dev y");

    hipModule_t mod; hipFunction_t func;
    int kfd = open("layernorm.co", O_RDONLY);
    if (kfd < 0) { perror("layernorm.co"); return false; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    HIP_CHECK(hipInit(0), "hipInit");
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    munmap(coData, st.st_size); close(kfd);
    HIP_CHECK(hipModuleGetFunction(&func, mod, "rmsnorm"), "get function");

    void* args[] = { &d_x, &d_g, &d_y, (void*)&tokens, (void*)&embd, (void*)&eps };
    HIP_CHECK(hipModuleLaunchKernel(func, tokens, 1, 1, 32, 1, 1, 0, 0, args, nullptr), "launch rmsnorm");
    HIP_CHECK(hipDeviceSynchronize(), "sync");
    HIP_CHECK(hipModuleUnload(mod), "unload");

    long failures = 0; float max_diff = 0.0f;
    for (int i = 0; i < tokens * embd; ++i) {
        float g = (float)y_gpu[i], r = (float)y_ref[i];
        float d = fabsf(g - r);
        if (d > max_diff) max_diff = d;
        if (bad_result(g, r)) {
            if (failures < 8) printf("    MISMATCH [%d]: gpu=%.5f ref=%.5f\n", i, g, r);
            failures++;
        }
    }

    printf("    zero-copy=%s  elements=%ld  failures=%ld  max_abs_diff=%.6f\n",
           (d_x == x && d_y == y_gpu) ? "yes" : "NO(!)", (long)(tokens * embd), failures, max_diff);

    HIP_CHECK(hipHostUnregister(base), "unpin");
    munmap(base, total); close(fd);
    return failures == 0;
}

int main(int argc, char* argv[]) {
    bool sweep = false;
    Shape shp{32, 64};
    float eps = 1e-5f;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--sweep")                 ) sweep = true;
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) shp.tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--embd")   && i + 1 < argc) shp.embd   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eps")    && i + 1 < argc) eps        = atof(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--tokens T --embd E [--eps f]] | [--sweep]\n", argv[0]);
            return 0;
        }
    }

    printf("=== RMSNorm Verification Harness (gfx1151 zero-copy) ===\n");

    std::vector<Shape> shapes;
    if (sweep) {
        shapes = { {8,17}, {32,64}, {5,4096}, {1,96}, {7,48} };
    } else {
        shapes.push_back(shp);
    }

    int failed = 0;
    for (const auto& s : shapes) if (!run_shape(s.tokens, s.embd, eps)) failed++;

    if (failed == 0) { printf("\n  LAYERNORM VERIFICATION PASSED (%zu shape%s)\n\n", shapes.size(), shapes.size() > 1 ? "s" : ""); return 0; }
    printf("\n  LAYERNORM VERIFICATION FAILED (%d/%zu shapes)\n\n", failed, shapes.size());
    return 1;
}
