// rope_verify.cpp -- Phase 3 acceptance test: rotate-half RoPE vs CPU reference over /dev/shm.
//
// Usage: ./rope_verify [--rows R --dim D] | [--sweep]
// The host builds FP32 cos/sin tables from CPU position IDs (theta_base = 1e4); GPU and CPU both
// consume the identical tables so any difference is pure algebraic rounding, not table divergence.

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

static uint32_t g_rng_state = 13;
static float rnd(float lo, float hi) {   // LCG -> [lo, hi)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return lo + ((float)(g_rng_state >> 8) / (float)(1u << 24)) * (hi - lo);
}

struct Shape { int rows, dim; };   // dim must be even

// Both sides apply identical fp32 algebra to bit-identical inputs and round once to FP16, so any
// difference is at most one output ulp from FMA-contraction variance. Tolerance keeps ~20x margin
// over that quantum; real bugs (wrong pairing, transposed table) are O(1) and get flagged.
static bool bad_result(float g, float r) {
    float d = fabsf(g - r);
    return d > 0.01f * fmaxf(fabsf(g), fabsf(r)) + 1e-4f;
}

static bool run_shape(int rows, int dim, const std::vector<int>& positions) {
    printf("--- Shape %d rows x %d dim (half=%d) ---\n", rows, dim, dim / 2);
    const int H = dim / 2;
    const float theta = 10000.0f;

    const size_t bytesX = (size_t)rows * dim * sizeof(__half);
    const size_t bytesT = (size_t)rows * H * sizeof(float);
    const size_t total  = bytesX + bytesT * 2 + bytesX * 2;   // [x][ctab][stab][y_gpu][y_ref]

    const char* path = "/dev/shm/rope_verify.bin";
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return false; }
    ftruncate(fd, (off_t)total);
    void* base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return false; }

    __half* x     = (__half*)base;
    float* ctab   = (float*)((char*)base + bytesX);
    float* stab   = (float*)((char*)base + bytesX + bytesT);
    __half* y_gpu = (__half*)((char*)base + bytesX + bytesT * 2);
    __half* y_ref = (__half*)((char*)base + bytesX + bytesT * 2 + bytesX);

    g_rng_state = 13;
    for (int i = 0; i < rows * dim; ++i) x[i] = __half(rnd(-3.0f, 3.0f));

    // Host-built RoPE tables from CPU position IDs (double-precision trig -> stored FP32).
    const double log_theta = log((double)theta);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < H; ++i) {
            const double freq = exp(-2.0 * ((double)i / (double)dim) * log_theta);
            const double angle = (double)positions[r] * freq;
            ctab[(size_t)r * H + i] = (float)cos(angle);
            stab[(size_t)r * H + i] = (float)sin(angle);
        }
    }
    memset(y_gpu, 0, bytesX);

    // CPU reference: same algebra on the identical tables.
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < H; ++i) {
            float lo = (float)x[(size_t)r * dim + i];
            float hi = (float)x[(size_t)r * dim + i + H];
            float c = ctab[(size_t)r * H + i], s = stab[(size_t)r * H + i];
            y_ref[(size_t)r * dim + i]      = (__half)(lo * c - hi * s);
            y_ref[(size_t)r * dim + i + H]  = (__half)(hi * c + lo * s);
        }
    }

    HIP_CHECK(hipHostRegister(base, total, hipHostRegisterDefault), "pin");

    __half* d_x; __half* d_y; float* d_c; float* d_s;
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_x, x, 0), "dev x");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_c, ctab, 0), "dev ctab");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_s, stab, 0), "dev stab");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_y, y_gpu, 0), "dev y");

    hipModule_t mod; hipFunction_t func;
    int kfd = open("rope_embedding.co", O_RDONLY);
    if (kfd < 0) { perror("rope_embedding.co"); return false; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    HIP_CHECK(hipInit(0), "hipInit");
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    munmap(coData, st.st_size); close(kfd);
    HIP_CHECK(hipModuleGetFunction(&func, mod, "rope_rotate_half"), "get function");

    void* args[] = { &d_x, &d_c, &d_s, &d_y, (void*)&rows, (void*)&H };
    HIP_CHECK(hipModuleLaunchKernel(func, rows, 1, 1, 32, 1, 1, 0, 0, args, nullptr), "launch rope");
    HIP_CHECK(hipDeviceSynchronize(), "sync");
    HIP_CHECK(hipModuleUnload(mod), "unload");

    long failures = 0; float max_diff = 0.0f;
    for (int i = 0; i < rows * dim; ++i) {
        float g = (float)y_gpu[i], r = (float)y_ref[i];
        float d = fabsf(g - r);
        if (d > max_diff) max_diff = d;
        if (bad_result(g, r)) {
            if (failures < 8) printf("    MISMATCH [%d]: gpu=%.5f ref=%.5f\n", i, g, r);
            failures++;
        }
    }

    printf("    zero-copy=%s  elements=%d  failures=%ld  max_abs_diff=%.7f\n",
           (d_x == x && d_y == y_gpu) ? "yes" : "NO(!)", rows * dim, failures, max_diff);

    HIP_CHECK(hipHostUnregister(base), "unpin");
    munmap(base, total); close(fd);
    return failures == 0;
}

int main(int argc, char* argv[]) {
    bool sweep = false;
    Shape shp{32, 64};

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--sweep")                ) sweep = true;
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) shp.rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim")   && i + 1 < argc) shp.dim  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--rows R --dim D] | [--sweep]\n", argv[0]);
            return 0;
        }
    }

    // CPU position IDs (mapped-buffer source in the engine design).
    std::vector<int> positions;
    uint32_t pstate = 99;
    int npos = sweep ? 64 : shp.rows;
    for (int r = 0; r < npos; ++r) {
        pstate = pstate * 1664525u + 1013904223u;
        positions.push_back(1 + ((int)(pstate >> 8) % 4096));
    }

    printf("=== RoPE Verification Harness (gfx1151 zero-copy, rotate-half, theta=1e4) ===\n");

    std::vector<Shape> shapes;
    if (sweep) {
        shapes = { {16,16}, {5,32}, {64,128} };
    } else {
        shapes.push_back(shp);
    }

    int failed = 0;

    for (const auto& s : shapes) {
        std::vector<int> pos(positions.begin(), positions.begin() + s.rows);
        if (!run_shape(s.rows, s.dim, pos)) failed++;

    }

    if (failed == 0) { printf("\n  ROPE VERIFICATION PASSED (%zu shape%s)\n\n", shapes.size(), shapes.size() > 1 ? "s" : ""); return 0; }
    printf("\n  ROPE VERIFICATION FAILED (%d/%zu shapes)\n\n", failed, shapes.size());
    return 1;
}
