// kv_cache.cpp -- Phase 3: unified-memory KV cache manager (implementation)
//
// Pattern matches every other binary in this repo: /dev/shm file -> ftruncate -> mmap(MAP_SHARED)
// -> hipHostRegister to pin pages. No copies anywhere; kernels address these pages directly.

#include "kv_cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

// Full definition of the opaque handle. The tag name MUST match the header's forward
// declaration so this definition completes that type within this translation unit.
struct inference_kv_cache {
    int fd = -1;
    void* base = nullptr;       // host mapping base
    void* dev_base = nullptr;   // device pointer of the same buffer
    bool zero_copy_ok = false;  // true when dev_base == base (UMA identity)
    size_t total_bytes = 0;     // whole mapped size incl. alignment padding
    size_t k_region = 0;        // bytes per K layer region (aligned)
    size_t v_region = 0;        // bytes per V layer region (aligned)
    kv_cache_config_t cfg{};
    int n_stored = 0;           // tokens stored so far (positions [0, n_stored))
};

namespace {

constexpr size_t REGION_ALIGN = 256;   // keep every layer region comfortably aligned for vector loads

size_t align_up(size_t x, size_t a = REGION_ALIGN) { return (x + a - 1) & ~(a - 1); }

__half* row_raw(inference_kv_cache_t* c, bool is_v, int layer, int pos) {
    if (!c || !c->base) return nullptr;
    if (layer < 0 || layer >= c->cfg.n_layers || pos < 0 || pos >= c->cfg.n_ctx) return nullptr;

    const int d = c->cfg.n_heads_kv * c->cfg.head_dim;
    const size_t row_bytes = (size_t)d * sizeof(__half);
    size_t off;
    if (is_v) {
        off = (size_t)c->cfg.n_layers * c->k_region + (size_t)layer * c->v_region;
    } else {
        off = (size_t)layer * c->k_region;
    }
    off += (size_t)pos * row_bytes;   // always inside the mapped file: regions are aligned super-sets of raw rows
    return (__half*)((char*)c->base + off);
}

} // namespace

extern "C" {

inference_kv_cache_t *kv_cache_create(const char *shm_path, const kv_cache_config_t *cfg_in) {
    if (!shm_path || !cfg_in) { fprintf(stderr, "kv_cache: null arg\n"); return nullptr; }
    const kv_cache_config_t& f = *cfg_in;
    if (f.n_layers <= 0 || f.n_ctx <= 0 || f.head_dim <= 0 || f.n_heads_q <= 0 || f.n_heads_kv <= 0) {
        fprintf(stderr, "kv_cache: invalid config\n");
        return nullptr;
    }

    inference_kv_cache_t* c = new inference_kv_cache();
    c->cfg = f;

    const size_t region_raw = (size_t)f.n_heads_kv * f.head_dim * sizeof(__half) * (size_t)f.n_ctx;
    c->k_region = align_up(region_raw);
    c->v_region = align_up(region_raw);
    c->total_bytes = c->k_region * f.n_layers + c->v_region * f.n_layers;   // n_layers > 0 validated above

    c->fd = open(shm_path, O_RDWR | O_CREAT, 0666);
    if (c->fd < 0) { perror("kv_cache: open"); delete c; return nullptr; }
    if (ftruncate(c->fd, (off_t)c->total_bytes) != 0) { perror("kv_cache: ftruncate"); close(c->fd); delete c; return nullptr; }

    c->base = mmap(NULL, c->total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, 0);
    if (c->base == MAP_FAILED) { perror("kv_cache: mmap"); close(c->fd); delete c; return nullptr; }

    hipError_t st = hipHostRegister(c->base, c->total_bytes, hipHostRegisterDefault);
    if (st != hipSuccess) {
        fprintf(stderr, "kv_cache: hipHostRegister failed: %s\n", hipGetErrorString(st));
        munmap(c->base, c->total_bytes); close(c->fd); delete c; return nullptr;
    }

    void* dev = nullptr;
    st = hipHostGetDevicePointer(&dev, c->base, 0);
    if (st == hipSuccess) {
        c->dev_base = dev;
        c->zero_copy_ok = (dev == c->base);
    } else {
        // Not fatal on a UMA APU where identity is expected; keep host address as the device view.
        fprintf(stderr, "kv_cache: WARNING: hipHostGetDevicePointer: %s (using host address)\n",
                hipGetErrorString(st));
        c->dev_base = c->base;
        c->zero_copy_ok = true;
    }

    return c;
}

void kv_cache_free(inference_kv_cache_t *c) {
    if (!c) return;
    if (c->base && c->base != MAP_FAILED) {
        (void)hipHostUnregister(c->base);   // idempotent enough for process-lifetime pinning
        munmap(c->base, c->total_bytes);
    }
    if (c->fd >= 0) close(c->fd);
    delete c;
}

int kv_cache_n_layers(const inference_kv_cache_t *c) { return c ? c->cfg.n_layers : 0; }
int kv_cache_n_ctx(const inference_kv_cache_t *c)    { return c ? c->cfg.n_ctx : 0; }
int kv_cache_d_kv(const inference_kv_cache_t *c)     { return c ? c->cfg.n_heads_kv * c->cfg.head_dim : 0; }
size_t kv_cache_total_bytes(const inference_kv_cache_t *c) { return c ? c->total_bytes : 0; }
size_t kv_cache_k_region_bytes(const inference_kv_cache_t *c) { return c ? c->k_region : 0; }
size_t kv_cache_v_region_bytes(const inference_kv_cache_t *c) { return c ? c->v_region : 0; }

void *kv_cache_host_base(const inference_kv_cache_t *c)       { return c ? c->base : nullptr; }
void *kv_cache_device_ptr(const inference_kv_cache_t *c)      { return c ? c->dev_base : nullptr; }
bool  kv_cache_zero_copy_ok(const inference_kv_cache_t *c)    { return c ? c->zero_copy_ok : false; }

__half        *kv_cache_k_row(inference_kv_cache_t *c, int layer, int pos)          { return row_raw(c, false, layer, pos); }
const __half  *kv_cache_k_row_const(const inference_kv_cache_t *c, int layer, int pos) { return row_raw(const_cast<inference_kv_cache_t*>(c), false, layer, pos); }
__half        *kv_cache_v_row(inference_kv_cache_t *c, int layer, int pos)          { return row_raw(c, true, layer, pos); }
const __half  *kv_cache_v_row_const(const inference_kv_cache_t *c, int layer, int pos) { return row_raw(const_cast<inference_kv_cache_t*>(c), true, layer, pos); }

void kv_cache_zero(inference_kv_cache_t *c) {
    if (c && c->base) memset(c->base, 0, c->total_bytes);
}

int kv_cache_advance(inference_kv_cache_t *c) {
    if (!c) return -1;
    return c->n_stored < c->cfg.n_ctx ? c->n_stored++ : -1;
}

int kv_cache_n_stored(const inference_kv_cache_t *c) { return c ? c->n_stored : 0; }

void kv_cache_reset_positions(inference_kv_cache_t *c) { if (c) c->n_stored = 0; }

} // extern "C"
