// softmax_verify.cpp -- Phase 3 acceptance test: masked row softmax vs CPU reference over /dev/shm.
//
// Usage: ./softmax_verify [--rows R --cols C] | [--sweep]
// Scenarios covered by the sweep: plain softmax, causal triangle (qpos=r), and decode with partial
// KV history (n_valid_kv < cols). Masked entries must be exactly zero; valid rows are checked against
// a sequential CPU implementation of the same math.

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

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s_ = (cmd); \
        if (s_ != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s_), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

static uint32_t g_rng_state = 21;
static float rnd(float lo, float hi) {   // LCG -> [lo, hi)
    g_rng_state = g_rng_state * 1664525u + 1013904223u;
    return lo + ((float)(g_rng_state >> 8) / (float)(1u << 24)) * (hi - lo);
}

struct Case {
    int rows, cols;
    bool causal;          // pass qpos (triangular mask); qpos[r] = r + pos_offset
    int n_valid_kv;       // columns beyond this are invalid KV slots (-1 means all valid)
    int pos_offset;       // shifts absolute query positions (decode continues a prefix)
};

static bool bad_result(float g, float r) {
    float d = fabsf(g - r);
    return d > 0.01f * fmaxf(fabsf(g), fabsf(r)) + 1e-5f;
}

// CPU reference: identical math to the kernel, sequential accumulation. Writes ref out of `orig`.
static void cpu_reference(const float* orig, const int* qpos, int rows, int cols, int n_valid_kv, float scale, float* ref) {
    for (int r = 0; r < rows; ++r) {
        const int qmax = qpos ? qpos[r] : cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            if (c >= n_valid_kv || c > qmax) continue;
            float v = orig[(size_t)r * cols + c] * scale;
            if (v > m) m = v;
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            if (c >= n_valid_kv || c > qmax) { ref[(size_t)r * cols + c] = 0.0f; continue; }
            float e = expf(orig[(size_t)r * cols + c] * scale - m);
            ref[(size_t)r * cols + c] = e;
            sum += e;
        }
        if (m == -INFINITY) { for (int c = 0; c < cols; ++c) ref[(size_t)r * cols + c] = 0.0f; continue; }
        for (int c = 0; c < cols; ++c) if (ref[(size_t)r * cols + c] != 0.0f) ref[(size_t)r * cols + c] /= sum;
    }
}

static bool run_case(const Case& t, const char* label) {
    printf("--- %s: %d rows x %d cols%s ---\n", label, t.rows, t.cols,
           t.causal ? " (causal)" : "");

    const size_t bytesS = (size_t)t.rows * t.cols * sizeof(float);
    const size_t total  = bytesS * 3 + (size_t)t.rows * sizeof(int);   // [scores][orig][ref][qpos]

    const char* path = "/dev/shm/softmax_verify.bin";
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) { perror("open"); return false; }
    ftruncate(fd, (off_t)total);
    void* base = mmap(NULL, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); close(fd); return false; }

    float* scores = (float*)base;
    float* orig   = (float*)((char*)base + bytesS);
    float* ref    = (float*)((char*)base + bytesS * 2);
    int*   qpos   = (int*)((char*)base + bytesS * 3);

    g_rng_state = 21;
    for (int i = 0; i < t.rows * t.cols; ++i) orig[i] = rnd(-3.0f, 3.0f);
    memcpy(scores, orig, bytesS);          // kernel mutates `scores`; ref is computed from the copy
    const int n_valid = t.n_valid_kv < 0 ? t.cols : t.n_valid_kv;
    for (int r = 0; r < t.rows; ++r) qpos[r] = r + t.pos_offset;   // filled always; kernel honors it only when causal

    const float scale = 1.0f / sqrtf((float)t.cols > 0 ? (float)t.cols : 1.0f);   // typical attention scaling

    cpu_reference(orig, t.causal ? qpos : nullptr, t.rows, t.cols, n_valid, scale, ref);

    HIP_CHECK(hipHostRegister(base, total, hipHostRegisterDefault), "pin");
    float* d_s; int* d_q;
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_s, scores, 0), "dev scores");
    HIP_CHECK(hipHostGetDevicePointer((void**)&d_q, qpos, 0), "dev qpos");

    hipModule_t mod; hipFunction_t func;
    int kfd = open("softmax_masked.co", O_RDONLY);
    if (kfd < 0) { perror("softmax_masked.co"); return false; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    HIP_CHECK(hipInit(0), "hipInit");
    HIP_CHECK(hipModuleLoadData(&mod, coData), "load module");
    munmap(coData, st.st_size); close(kfd);
    HIP_CHECK(hipModuleGetFunction(&func, mod, "softmax_masked"), "get function");

    // NOTE: every args[] entry must be a real address -- the runtime copies from each one, so a raw
    // NULL entry segfaults. To pass "no qpos", hand over the address of a null pointer variable.
    int* qpass = t.causal ? d_q : nullptr;
    void* args[] = { &d_s, (void*)&t.rows, (void*)&t.cols, (void*)&scale,
                     &qpass, (void*)&n_valid };
    HIP_CHECK(hipModuleLaunchKernel(func, t.rows, 1, 1, 32, 1, 1, 0, 0, args, nullptr), "launch softmax");
    HIP_CHECK(hipDeviceSynchronize(), "sync");
    HIP_CHECK(hipModuleUnload(mod), "unload");

    long failures = 0; float max_diff = 0.0f;
    for (int r = 0; r < t.rows; ++r) {
        const int qmax = t.causal ? (r + t.pos_offset) : t.cols;   // mirror of kernel's qpos
        for (int c = 0; c < t.cols; ++c) {
            bool valid = (c < n_valid && c <= qmax);
            float g = scores[(size_t)r * t.cols + c];
            if (!valid) {
                if (fabsf(g) > 1e-9f) { if (failures < 8) printf("    MASK LEAK [%d,%d]: %.6g\n", r, c, g); failures++; }
                continue;
            }
            float rv = ref[(size_t)r * t.cols + c];
            float d = fabsf(g - rv);
            if (d > max_diff) max_diff = d;
            if (bad_result(g, rv)) { if (failures < 8) printf("    MISMATCH [%d,%d]: gpu=%.6f ref=%.6f\n", r, c, g, rv); failures++; }
        }
    }

    printf("    zero-copy=%s  elements=%d  failures=%ld  max_abs_diff=%.6f\n",
           (d_s == scores) ? "yes" : "NO(!)", t.rows * t.cols, failures, max_diff);

    HIP_CHECK(hipHostUnregister(base), "unpin");
    munmap(base, total); close(fd);
    return failures == 0;
}

int main(int argc, char* argv[]) {
    bool sweep = false;
    Case single{32, 48, false, -1, 0};

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--sweep")                ) sweep = true;
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) single.rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i + 1 < argc) single.cols = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--rows R --cols C] | [--sweep]\n", argv[0]);
            return 0;
        }
    }

    printf("=== Masked Softmax Verification Harness (gfx1151 zero-copy) ===\n");

    std::vector<Case> cases;
    if (sweep) {
        cases = { {8,17,false,-1,0},     // plain softmax, partial-width row
                  {6,32,true,-1,0},      // causal triangle over full history
                  {1,32,true,7,6} };     // decode at pos 6 with only 7 stored KV slots
    } else {
        cases.push_back(single);
    }

    int failed = 0;
    const char* labels[] = {"plain", "causal", "decode-partial"};
    for (size_t i = 0; i < cases.size(); ++i)
        if (!run_case(cases[i], i < 3 ? labels[i] : "case")) failed++;

    if (failed == 0) { printf("\n  SOFTMAX VERIFICATION PASSED (%zu case%s)\n\n", cases.size(), cases.size() > 1 ? "s" : ""); return 0; }
    printf("\n  SOFTMAX VERIFICATION FAILED (%d/%zu cases)\n\n", failed, cases.size());
    return 1;
}
