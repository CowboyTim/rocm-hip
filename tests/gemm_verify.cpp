// gemm_verify.cpp -- Phase 3 acceptance test: FP16 WMMA GEMM vs CPU reference over /dev/shm.
//
// Usage: ./gemm_verify [--M m] [--N n] [--K k] | [--sweep]
// The sweep runs a fixed set of shapes incl. partial tiles/strips and a 1K^3 cube in one process.

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

static uint32_t g_rng_state = 42;
static float rnd() {   // LCG -> [-2, 2)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return ((float)(g_rng_state >> 8) / (float)(1u << 24)) * 4.0f - 2.0f;
}

struct Shape { int M, N, K; };

// One shape end-to-end on the shared mapping: fill A/B, CPU ref, GPU kernel, compare.
// Returns true when all elements pass tolerance.
static bool run_shape(int M, int N, int K) {
    printf("--- Shape %d x %d x %d ---\n", M, N, K);

    const size_t bytesA = (size_t)M * K * sizeof(__half);
    const size_t bytesB = (size_t)K * N * sizeof(__half);
    const size_t bytesC = (size_t)M * N * sizeof(float);
    const size_t total  = bytesA + bytesB + bytesC * 2;

    const char* path = "/dev/shm/gemm_verify.bin";
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return false; }
    ftruncate(fd, (off_t)total);
    void* base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return false; }

    __half* A   = (__half*)base;
    __half* B   = (__half*)((char*)base + bytesA);
    float* Cgpu = (float*)((char*)base + bytesA + bytesB);
    float* Ccpu = Cgpu + M * N;

    g_rng_state = 42;   // deterministic across shapes/processes
    for (int i = 0; i < M * K; ++i) A[i] = __half(rnd());
    for (int i = 0; i < K * N; ++i) B[i] = __half(rnd());
    memset(Cgpu, 0, bytesC);
    memset(Ccpu, 0, bytesC);

    // CPU reference from the STORED halves (bit-identical input to the GPU kernel).
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += (float)A[i * K + k] * (float)B[k * N + j];
            Ccpu[i * N + j] = sum;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    HIP_CHECK(hipHostRegister(base, total, hipHostRegisterDefault), "pin mapping");

    __half* d_A; __half* d_B; float* d_C;
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_A, A, 0), "dev A");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_B, B, 0), "dev B");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_C, Cgpu, 0), "dev C");
    bool zero_copy = (d_A == A && d_B == B && d_C == Cgpu);

    hipModule_t mod; hipFunction_t func;
    int kfd = open("fp16_gemm.co", O_RDONLY);
    if (kfd < 0) { perror("fp16_gemm.co"); return false; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    HIP_CHECK(hipInit(0), "hipInit");
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    munmap(coData, st.st_size); close(kfd);
    HIP_CHECK(hipModuleGetFunction(&func, mod, "fp16_gemm"), "get function");

    dim3 grid((M + 15) / 16, (N + 15) / 16);
    void* args[] = { &d_A, &d_B, &d_C, (void*)&M, (void*)&N, (void*)&K };
    HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1, 32, 1, 1, 0, 0, args, nullptr), "launch");
    HIP_CHECK(hipDeviceSynchronize(), "sync");
    HIP_CHECK(hipModuleUnload(mod), "unload");

    // Tolerance calibrated against the measured FP16 per-strip rounding floor. Rounding inside the
    // WMMA accumulator grows like ~0.003*sqrt(K) worst-case (observed max_abs_diff: 0.017@K40,
    // 0.05@K256, 0.094@K1024), so atol = 0.01*sqrt(K) keeps ~3x margin while any real indexing or
    // mapping bug (O(1..100) diffs) is flagged instantly. A relative term catches scaling errors on
    // large-magnitude results; near-zero cancellation sums are judged by atol alone (a bare
    // diff/|ref| ratio explodes there and would flag correct noise).
    const float atol = 0.01f * std::sqrt((float)K);
    const float rtol = 0.01f;   // 1% of |ref|

    long failures = 0;
    float max_abs = 0.0f, max_rel_pct = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float g = Cgpu[i * N + j], r = Ccpu[i * N + j];
            float d = fabsf(g - r);
            if (d > max_abs) max_abs = d;
            float rel_pct = (fabsf(r) > 1e-6f) ? 100.0f * d / fabsf(r) : 0.0f;
            if (rel_pct > max_rel_pct) max_rel_pct = rel_pct;
            bool bad = (d > atol + rtol * fabsf(r));
            if (bad && failures < 8)
                printf("    MISMATCH [%d,%d]: gpu=%.3f cpu=%.3f diff=%.4f tol=%.4f (%.2f%%)\n", i, j, g, r, d, atol + rtol * fabsf(r), rel_pct);
            if (bad) failures++;
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("    zero-copy=%s  elements=%d  failures=%ld  max_abs_diff=%.4f (atol=%.4f)  max_rel_vs_raw_ref=%.2f%%  (cpu ref %.1f ms)\n",
           zero_copy ? "yes" : "NO(!)", M * N, failures, max_abs, atol, max_rel_pct, ms);

    HIP_CHECK(hipHostUnregister(base), "unpin");
    munmap(base, total); close(fd);
    return failures == 0;
}

int main(int argc, char* argv[]) {
    bool sweep = false;
    Shape shp{33, 17, 40};   // default: partial tiles + partial K strip

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--sweep")              ) sweep = true;
        else if (!strcmp(argv[i], "--M") && i + 1 < argc) shp.M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) shp.N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) shp.K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--M m --N n --K k] | [--sweep]\n", argv[0]);
            return 0;
        }
    }

    printf("=== FP16 WMMA GEMM Verification Harness (gfx1151 zero-copy) ===\n");

    std::vector<Shape> shapes;
    if (sweep) {
        shapes = { {16,16,16}, {33,17,40}, {96,80,256}, {256,256,256}, {1024,1024,1024} };
    } else {
        shapes.push_back(shp);
    }

    int failed_shapes = 0;
    for (const auto& s : shapes) {
        if (!run_shape(s.M, s.N, s.K)) failed_shapes++;
    }

    if (failed_shapes == 0) { printf("\n  GEMM VERIFICATION PASSED (%zu shape%s)\n\n", shapes.size(), shapes.size() > 1 ? "s" : ""); return 0; }
    printf("\n  GEMM VERIFICATION FAILED (%d/%zu shapes)\n\n", failed_shapes, shapes.size());
    return 1;
}
