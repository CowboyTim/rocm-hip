// kv_verify.cpp -- Phase 3 acceptance test: KV cache manager + zero-copy self-test on hardware.
//
// Usage: ./kv_verify [--layers L] [--ctx N] [--embd E] [--hd H] [--hq HQ] [--hkv HKV]
// Exits non-zero if any check fails (unlike the bench demo binaries).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include "kv_cache.h"

#define HIP_CHECK(cmd, str) \
    do { \
        hipError_t s_ = (cmd); \
        if (s_ != hipSuccess) { \
            fprintf(stderr, "ERROR: %s at %s:%d (%s)\n", hipGetErrorString(s_), __FILE__, __LINE__, str); \
            exit(1); \
        } \
    } while(0)

struct Cfg {
    int n_layers = 4;
    int n_ctx = 32;
    int n_embd = 64;
    int head_dim = 8;
    int n_heads_q = 8;
    int n_heads_kv = 2;   // d_kv = 16 -> fits one WMMA tile width for later attention tests
};

// CPU-written slot patterns: small integers so FP16 storage is exact and the check must match bit-for-bit.
static float k_pat(int l, int p, int e) { return (float)((l + 1) * 37 + p * 7 + e); }
static float v_pat(int l, int p, int e) { return -((float)((l + 1) * 13 + p * 5 + e)); }

int g_failures = 0;

static void report(const char* what, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) g_failures++;
}

// Load a pre-compiled kernel module from a .co file by symbol name.
static bool load_kernel(const char* co_path, const char* func_name, hipModule_t& mod_out, hipFunction_t& func_out) {
    int kfd = open(co_path, O_RDONLY);
    if (kfd < 0) { perror(co_path); return false; }
    struct stat st; fstat(kfd, &st);
    void* coData = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, kfd, 0);
    bool ok = true;
    HIP_CHECK(hipInit(0), "hipInit");
    if (hipModuleLoadData(&mod_out, coData) != hipSuccess) { fprintf(stderr, "load %s: %s\n", co_path, hipGetErrorString(hipGetLastError())); ok = false; }
    else if (hipModuleGetFunction(&func_out, mod_out, func_name) != hipSuccess) { fprintf(stderr, "symbol %s: %s\n", func_name, hipGetErrorString(hipGetLastError())); ok = false; }
    munmap(coData, st.st_size);
    close(kfd);
    return ok;
}

int main(int argc, char* argv[]) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--layers") && i + 1 < argc) c.n_layers   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx")    && i + 1 < argc) c.n_ctx      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--embd")   && i + 1 < argc) c.n_embd     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hd")     && i + 1 < argc) c.head_dim   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hq")     && i + 1 < argc) c.n_heads_q  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hkv")    && i + 1 < argc) c.n_heads_kv = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            printf("Usage: %s [--layers L] [--ctx N] [--embd E] [--hd H] [--hq HQ] [--hkv HKV]\n", argv[0]);
            return 0;
        }
    }

    const int d_kv = c.n_heads_kv * c.head_dim;

    printf("=== Phase 3 KV Cache Self-Test (gfx1151 zero-copy) ===\n");
    printf("  layers=%d ctx=%d embd=%d head_dim=%d heads_q=%d heads_kv=%d -> d_kv=%d\n",
           c.n_layers, c.n_ctx, c.n_embd, c.head_dim, c.n_heads_q, c.n_heads_kv, d_kv);

    // One block covering the whole context (block_tokens = the next power of two >= n_ctx): every layer's K and V
    // segments are then contiguous runs of rows, as kv_cache_selftest addresses them
    int kv_block = 1;
    while (kv_block < c.n_ctx) kv_block *= 2;
    kv_cache_config_t cfg{c.n_layers, c.n_ctx, c.n_embd, c.head_dim, c.n_heads_q, c.n_heads_kv, kv_block};
    inference_kv_cache_t* cache = kv_cache_create("/dev/shm/kv_cache_verify.bin", &cfg);
    if (!cache) { fprintf(stderr, "kv_cache_create failed\n"); return 1; }

    printf("  Mapping: %zu bytes in /dev/shm (%d-position blocks of %zu bytes)\n", kv_cache_total_bytes(cache),
           kv_cache_block_tokens(cache), kv_cache_block_bytes(cache));
    report("zero-copy pin identity (device ptr == host base)", kv_cache_zero_copy_ok(cache));
    report("config round-trip (n_layers/n_ctx/d_kv)",
           kv_cache_n_layers(cache) == c.n_layers && kv_cache_n_ctx(cache) >= c.n_ctx && kv_cache_d_kv(cache) == d_kv);
    c.n_ctx = kv_cache_n_ctx(cache);  // rounded up to whole blocks: the checks below use the cache's context

    // --- bounds safety on row accessors ---
    report("row accessors reject out-of-range layer/pos",
           kv_cache_k_row_const(cache, -1, 0) == nullptr &&
           kv_cache_k_row_const(cache, c.n_layers, 0) == nullptr &&
           kv_cache_v_row_const(cache, 0, -1) == nullptr &&
           kv_cache_v_row_const(cache, 0, c.n_ctx) == nullptr &&
           kv_cache_k_row_const(cache, 0, c.n_ctx - 1) != nullptr);

    // --- zero() through the mapping ---
    kv_cache_zero(cache);
    bool zeros_ok = true;
    for (int l = 0; l < c.n_layers; ++l) {
        const __half* kr = kv_cache_k_row_const(cache, l, c.n_ctx / 2);
        const __half* vr = kv_cache_v_row_const(cache, l, c.n_ctx / 2);
        for (int e = 0; e < d_kv; ++e) if ((float)kr[e] != 0.0f || (float)vr[e] != 0.0f) zeros_ok = false;
    }
    report("kv_cache_zero wipes all K/V rows", zeros_ok);

    // --- position bookkeeping ---
    bool adv_ok = true;
    for (int i = 0; i < c.n_ctx; ++i) {
        if (kv_cache_advance(cache) != i) adv_ok = false;
    }
    if (kv_cache_advance(cache) != -1) adv_ok = false;   // context full: next advance must fail
    report("advance() yields 0..n_ctx-1 then -1", adv_ok && kv_cache_n_stored(cache) == c.n_ctx);
    kv_cache_reset_positions(cache);
    report("reset_positions restores n_stored=0", kv_cache_n_stored(cache) == 0);

    // --- CPU pattern write through row accessors ---
    for (int l = 0; l < c.n_layers; ++l) {
        for (int p = 0; p < c.n_ctx; ++p) {
            __half* kr = kv_cache_k_row(cache, l, p);
            __half* vr = kv_cache_v_row(cache, l, p);
            for (int e = 0; e < d_kv; ++e) {
                kr[e] = __half(k_pat(l, p, e));
                vr[e] = __half(v_pat(l, p, e));
            }
        }
    }

    // --- GPU self-test: kernel reads K/V from the pinned KV mapping, writes a check buffer ---
    const size_t total_elems = (size_t)c.n_layers * c.n_ctx * d_kv;
    char scratch_path[] = "/dev/shm/kv_verify_scratch.bin";
    int sfd = open(scratch_path, O_RDWR | O_CREAT, 0666);
    if (sfd < 0) { perror("open scratch"); return 1; }
    ftruncate(sfd, (off_t)(total_elems * sizeof(float)));
    float* check_host = (float*)mmap(NULL, total_elems * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED, sfd, 0);
    if (check_host == MAP_FAILED) { perror("mmap scratch"); return 1; }
    memset(check_host, 0, total_elems * sizeof(float));
    HIP_CHECK(hipHostRegister(check_host, total_elems * sizeof(float), hipHostRegisterDefault), "pin scratch");
    float* dev_check = nullptr;
    HIP_CHECK(hipHostGetDevicePointer((void**)&dev_check, check_host, 0), "scratch dev ptr");

    void* kv_dev_base = kv_cache_device_ptr(cache);
    // Layer l's K rows start at base + 2 l * B rows, its V rows B rows later (kv_cache.h, one block)
    __half* k_dev = (__half*)kv_dev_base;
    __half* v_dev = (__half*)kv_dev_base + (size_t)kv_block * d_kv;

    hipModule_t mod; hipFunction_t func;
    if (!load_kernel("kv_cache_kernel.co", "kv_cache_selftest", mod, func)) return 1;

    const size_t k_halfs = 2 * (size_t)kv_block * d_kv;  // per-layer stride of the K (and V) runs
    const size_t v_halfs = k_halfs;

    void* args[] = { &k_dev, &v_dev, &dev_check,
                     (void*)&c.n_layers, (void*)&c.n_ctx, (void*)&d_kv,
                     (void*)&k_halfs, (void*)&v_halfs };

    int block = 32;
    int grid = (int)((total_elems + block - 1) / block);
    if (grid > 4096) grid = 4096;   // kernel is grid-stride; cap for large configs

    printf("  Launching kv_cache_selftest<<<%d, %d>>> over %zu slots...\n", grid, block, total_elems);
    HIP_CHECK(hipModuleLaunchKernel(func, grid, 1, 1, block, 1, 1, 0, 0, args, nullptr), "launch selftest");
    HIP_CHECK(hipDeviceSynchronize(), "sync selftest");

    // --- verify every check element against the CPU expectation ---
    long mismatches = 0;
    float max_diff = 0.0f;
    for (size_t idx = 0; idx < total_elems; ++idx) {
        const size_t rem  = idx % ((size_t)c.n_ctx * d_kv);
        const int l       = idx / ((size_t)c.n_ctx * d_kv);
        const int p       = rem / d_kv;
        const int e       = rem % d_kv;
        float expected = k_pat(l, p, e) + 2.0f * v_pat(l, p, e) + (float)p;   // FP16-exact: small integers
        float diff = fabsf(check_host[idx] - expected);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f && mismatches < 8) {
            printf("    MISMATCH [%d,%d,%d]: gpu=%.4f expected=%.4f\n", l, p, e, check_host[idx], expected);
        }
        if (diff > 1e-3f) mismatches++;
    }
    report("GPU read+write of pinned KV pages matches CPU pattern", mismatches == 0);
    printf("    elements=%zu mismatches=%ld max_diff=%.6f\n", total_elems, mismatches, max_diff);

    // --- cleanup ---
    HIP_CHECK(hipHostUnregister(check_host), "unpin scratch");
    munmap(check_host, total_elems * sizeof(float));
    close(sfd);
    kv_cache_free(cache);

    // --- block slots (Phase 22b): publish, private copy, attach ---
    {
        kv_cache_config_t scfg{2, 32, c.n_embd, c.head_dim, c.n_heads_q, c.n_heads_kv, 8};  // 4 slots of 8 positions
        inference_kv_cache_t* sc = kv_cache_create("/dev/shm", &scfg);
        report("slot cache: 4 slots of 8 positions", sc && kv_cache_n_slots(sc) == 4);
        if (sc) {
            const int dkv = kv_cache_d_kv(sc);
            for (int p = 0; p < 32; ++p)
                for (int l = 0; l < 2; ++l)
                    for (int i = 0; i < dkv; ++i) {
                        kv_cache_k_row(sc, l, p)[i] = __float2half((float)(p * 3 + l + 1));
                        kv_cache_v_row(sc, l, p)[i] = __float2half((float)(p * 3 + l + 2));
                    }
            char path[96];
            snprintf(path, sizeof(path), "/dev/shm/kv_verify_block.%d", (int)getpid());
            unlink(path);
            const int pub = kv_cache_slot_publish(sc, 1, path);  // positions 8..15
            struct stat stb;
            const bool named = pub == 0 && stat(path, &stb) == 0 && (size_t)stb.st_size == kv_cache_block_bytes(sc);
            report("publish: slot 1 linked under a name, one block long, slot now shared",
                   named && kv_cache_slot_shared(sc, 1) && kv_cache_slot_publish(sc, 1, path) == 1);
            // private copy: same data, and writes no longer reach the published file
            const bool priv = kv_cache_slot_make_private(sc, 1, 1) == 0 && !kv_cache_slot_shared(sc, 1) &&
                              __half2float(kv_cache_k_row(sc, 1, 9)[0]) == 9 * 3 + 2;
            kv_cache_k_row(sc, 1, 9)[0] = __float2half(-1.0f);
            const int pfd = open(path, O_RDONLY);
            __half first_k1;  // the published file still holds the old row: layer 1 K, position 9 = row 1 of segment 2
            const off_t at = (off_t)(2 * 8 + 1) * dkv * sizeof(__half);
            const bool file_kept = pfd >= 0 && pread(pfd, &first_k1, sizeof(first_k1), at) == sizeof(first_k1) &&
                                   __half2float(first_k1) == 9 * 3 + 2;
            report("make_private(copy): same rows; later writes do not reach the published block", priv && file_kept);
            // attach the published block to slot 3 (positions 24..31): it now reads positions 8..15's rows
            const bool att = pfd >= 0 && kv_cache_slot_attach(sc, 3, pfd) == 0 && kv_cache_slot_shared(sc, 3) &&
                             __half2float(kv_cache_k_row(sc, 0, 24)[0]) == 8 * 3 + 1 &&
                             __half2float(kv_cache_v_row(sc, 1, 31)[dkv - 1]) == 15 * 3 + 1 + 2;
            report("attach: a block file mapped into another slot shows its rows", att);
            if (pfd >= 0) close(pfd);
            unlink(path);
            kv_cache_free(sc);
        }
    }

    if (g_failures == 0) {
        printf("\n  KV CACHE SELF-TEST PASSED\n");
        return 0;
    }
    printf("\n  KV CACHE SELF-TEST FAILED (%d checks)\n", g_failures);
    return 1;
}
