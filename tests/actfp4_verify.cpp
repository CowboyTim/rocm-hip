// actfp4_verify.cpp -- Phase 3 acceptance test: FP16 activation x ROCmFP4 Codebook10 weight GEMM vs CPU reference.
//
// Usage: ./actfp4_verify [--M m --N n --K k] | [--sweep]   (K must be a multiple of 16)
// Weights are generated as exact codebook codes x UE4M3 scales; both sides use the SAME fp16-rounded
// dequantized values (the kernel rounds each dequanted element once to FP16 before the dot product), so
// any difference is pure WMMA per-strip accumulation rounding, bounded like fp16_gemm (~0.003*sqrt(K)).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "fp4_common.h"

using namespace rocmfp4;

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s_ = (cmd); \
        if (s_ != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s_), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

static uint32_t g_rng_state = 55;
static float rndf(float lo, float hi) {   // LCG -> [lo, hi)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return lo + ((float)(g_rng_state >> 8) / (float)(1u << 24)) * (hi - lo);
}
static int rndi(int n) {                  // LCG -> [0, n)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return (int)((g_rng_state >> 8) % (uint32_t)n);
}

struct Shape { int M, N, K; };

static bool run_shape(int M, int N, int K) {
    printf("--- Act x FP4 shape A[%dx%d] W[%dx%d] ---\n", M, K, N, K);

    const size_t bytesA = (size_t)M * K * sizeof(__half);
    const size_t bytesW = (size_t)N * (K / 2);
    const size_t bytesSc= (size_t)N * (K / 16);
    const size_t bytesC = (size_t)M * N * sizeof(float);
    const size_t total  = bytesA + bytesW + bytesSc + bytesC * 2;   // [A][W packed][W scales][C gpu][C ref]

    const char* path = "/dev/shm/actfp4_verify.bin";
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return false; }
    ftruncate(fd, (off_t)total);
    void* base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return false; }

    __half* A       = (__half*)base;
    unsigned char* W     = (unsigned char*)((char*)base + bytesA);
    unsigned char* Wscl  = W + bytesW;
    float* Cgpu      = (float*)((char*)base + bytesA + bytesW + bytesSc);
    float* Cref      = Cgpu + M * N;

    g_rng_state = 55;
    for (int i = 0; i < M * K; ++i) A[i] = __half(rndf(-2.0f, 2.0f));

    // Weights: exact codebook entries x UE4M3 scales decoding into [0.1, 1.0]; effective values are the
    // FP16-rounded dequant (what the kernel actually multiplies with).
    std::vector<uint8_t> pool;
    for (int b = 0; b < 256; ++b) {
        float v = ue4m3_to_float((uint8_t)b);
        if (v >= 0.1f && v <= 1.0f) pool.push_back((uint8_t)b);
    }
    std::vector<float> wval((size_t)N * K);   // fp16-rounded dequanted weights (shared by both sides)
    const size_t strips = K / 16;
    for (int n = 0; n < N; ++n) {
        memset(W + (size_t)n * (K / 2), 0, K / 2);
        for (size_t s = 0; s < strips; ++s) {
            uint8_t sb = pool[rndi((int)pool.size())];
            Wscl[(size_t)n * strips + s] = sb;
            float sc = ue4m3_to_float(sb);
            for (int i = 0; i < 16; ++i) {
                int k = (int)s * 16 + i;
                int code = CODEBOOK[rndi(16)];
                uint8_t nib = (uint8_t)encode_code(code);      // canonical nibble (sign = bit 3, not two's-complement!)
                W[(size_t)n * (K / 2) + (k >> 1)] |= (nib << ((k & 1) * 4));   // canonical packing: even k -> low nibble
                wval[(size_t)n * K + k] = (float)__half((float)code * sc);   // one fp16 round, as in-kernel
            }
        }
    }

    // CPU reference from the shared effective values.
    memset(Cgpu, 0, bytesC);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += (float)A[i * K + k] * wval[(size_t)j * K + k];
            Cref[i * N + j] = sum;
        }

    HIP_CHECK(hipHostRegister(base, total, hipHostRegisterDefault), "pin");

    __half* d_A; unsigned char* d_W; unsigned char* d_Sc; float* d_C;
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_A, A, 0), "dev A");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_W, W, 0), "dev W");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_Sc, Wscl, 0), "dev Wscl");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_C, Cgpu, 0), "dev C");

    hipModule_t mod; hipFunction_t func;
    int kfd = open("gemm_act_fp4.co", O_RDONLY);
    if (kfd < 0) { perror("gemm_act_fp4.co"); return false; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    HIP_CHECK(hipInit(0), "hipInit");
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    munmap(coData, st.st_size); close(kfd);
    HIP_CHECK(hipModuleGetFunction(&func, mod, "gemm_act_fp4"), "get function");

    dim3 grid((M + 15) / 16, (N + 15) / 16);
    void* args[] = { &d_A, (void*)&d_W, (void*)&d_Sc, &d_C, (void*)&M, (void*)&N, (void*)&K };
    HIP_CHECK(hipModuleLaunchKernel(func, grid.x, grid.y, 1, 32, 1, 1, 0, 0, args, nullptr), "launch");
    HIP_CHECK(hipDeviceSynchronize(), "sync");
    HIP_CHECK(hipModuleUnload(mod), "unload");

    const float atol = 0.01f * std::sqrt((float)K);   // same calibrated floor as fp16_gemm
    const float rtol = 0.01f;

    long failures = 0; float max_abs = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float g = Cgpu[i], r = Cref[i];
        float d = fabsf(g - r);
        if (d > max_abs) max_abs = d;
        bool bad = (d > atol + rtol * fabsf(r));
        if (bad && failures < 8) printf("    MISMATCH [%d]: gpu=%.4f ref=%.4f diff=%.4f tol=%.4f\n", i, g, r, d, atol + rtol*fabsf(r));
        if (bad) failures++;
    }

    printf("    zero-copy=%s  elements=%ld  failures=%ld  max_abs_diff=%.5f (atol=%.4f)\n",
           (d_A == A && d_C == Cgpu) ? "yes" : "NO(!)", M * N, failures, max_abs, atol);

    HIP_CHECK(hipHostUnregister(base), "unpin");
    munmap(base, total); close(fd);
    return failures == 0;
}

int main(int argc, char* argv[]) {
    bool sweep = false;
    Shape shp{33, 17, 32};

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--sweep")              ) sweep = true;
        else if (!strcmp(argv[i], "--M") && i + 1 < argc) shp.M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i + 1 < argc) shp.N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K") && i + 1 < argc) shp.K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--M m --N n --K k (multiple of 16)] | [--sweep]\n", argv[0]);
            return 0;
        }
    }

    printf("=== Act x FP4 GEMM Verification Harness (gfx1151 zero-copy Codebook10) ===\n");

    std::vector<Shape> shapes;
    if (sweep) shapes = { {16,16,16}, {33,17,32}, {96,80,256} };
    else       shapes.push_back(shp);

    int failed = 0;
    for (const auto& s : shapes) if (!run_shape(s.M, s.N, s.K)) failed++;

    if (failed == 0) { printf("\n  ACT-FP4 VERIFICATION PASSED (%zu shape%s)\n\n", shapes.size(), shapes.size() > 1 ? "s" : ""); return 0; }
    printf("\n  ACT-FP4 VERIFICATION FAILED (%d/%zu shapes)\n\n", failed, shapes.size());
    return 1;
}
