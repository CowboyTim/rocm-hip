// kv_cache.cpp -- Phase 3: unified-memory KV cache manager (implementation)
//
// Pattern matches every other binary in this repo: /dev/shm file -> ftruncate -> mmap(MAP_SHARED)
// -> hipHostRegister to pin pages. No copies anywhere; kernels address these pages directly.

#include "inference_log.h"
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
    size_t total_bytes = 0;     // whole mapped size
    int block_tokens = 0;       // positions per block (power of two)
    int block_shift = 0;        // log2(block_tokens)
    size_t row_bytes = 0;       // one position's K (or V) row of one layer
    size_t block_bytes = 0;     // every layer's K and V for block_tokens positions
    kv_cache_config_t cfg{};
    int n_stored = 0;           // tokens stored so far (positions [0, n_stored))
};

namespace {

__half* row_raw(inference_kv_cache_t* c, bool is_v, int layer, int pos) {
    if (!c || !c->base) return nullptr;
    if (layer < 0 || layer >= c->cfg.n_layers || pos < 0 || pos >= c->cfg.n_ctx) {
        INFER_LOG("[KV_CACHE] row out of range: layer=%d n_layers=%d pos=%d n_ctx=%d (is_v=%d)\n", layer,
                  c->cfg.n_layers, pos, c->cfg.n_ctx, is_v);
        return nullptr;
    }
    const size_t B = (size_t)c->block_tokens;
    const size_t off = (size_t)(pos >> c->block_shift) * c->block_bytes + (2 * (size_t)layer + is_v) * B * c->row_bytes +
                       (size_t)(pos & (c->block_tokens - 1)) * c->row_bytes;
    return (__half*)((char*)c->base + off);
}

} // namespace

extern "C" {

inference_kv_cache_t *kv_cache_create(const char *shm_path, const kv_cache_config_t *cfg_in) {
    if (!shm_path || !cfg_in) { INFER_LOG("kv_cache: null arg\n"); return nullptr; }
    const kv_cache_config_t& f = *cfg_in;
    if (f.n_layers <= 0 || f.n_ctx <= 0 || f.head_dim <= 0 || f.n_heads_q <= 0 || f.n_heads_kv <= 0) {
        INFER_LOG("kv_cache: invalid config\n");
        return nullptr;
    }

    const int B = f.block_tokens > 0 ? f.block_tokens : KV_CACHE_DEFAULT_BLOCK;
    if ((B & (B - 1)) != 0) {
        INFER_LOG("kv_cache: block_tokens %d is not a power of two\n", B);
        return nullptr;
    }
    inference_kv_cache_t* c = new inference_kv_cache();
    c->cfg = f;
    c->cfg.block_tokens = B;
    c->cfg.n_ctx = (f.n_ctx + B - 1) / B * B;  // whole blocks
    c->block_tokens = B;
    while ((1 << c->block_shift) < B) ++c->block_shift;
    c->row_bytes = (size_t)f.n_heads_kv * f.head_dim * sizeof(__half);
    c->block_bytes = 2 * (size_t)f.n_layers * B * c->row_bytes;
    c->total_bytes = c->block_bytes * (size_t)(c->cfg.n_ctx / B);

    c->fd = open(shm_path, O_RDWR | O_CREAT, 0666);
    if (c->fd < 0) { perror("kv_cache: open"); delete c; return nullptr; }
    if (ftruncate(c->fd, (off_t)c->total_bytes) != 0) { perror("kv_cache: ftruncate"); close(c->fd); delete c; return nullptr; }

    c->base = mmap(NULL, c->total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, 0);
    if (c->base == MAP_FAILED) { perror("kv_cache: mmap"); close(c->fd); delete c; return nullptr; }

    // Coarse-grained: GPU-cacheable, coherent with the CPU at kernel boundaries and syncs. Still zero-copy.
    hipError_t st = hipHostRegister(c->base, c->total_bytes, hipExtHostRegisterCoarseGrained);
    if (st != hipSuccess) {
        INFER_LOG("kv_cache: hipHostRegister failed: %s\n", hipGetErrorString(st));
        munmap(c->base, c->total_bytes); close(c->fd); delete c; return nullptr;
    }

    void* dev = nullptr;
    st = hipHostGetDevicePointer(&dev, c->base, 0);
    if (st == hipSuccess) {
        c->dev_base = dev;
        c->zero_copy_ok = (dev == c->base);
    } else {
        // Not fatal on a UMA APU where identity is expected; keep host address as the device view.
        INFER_LOG("kv_cache: WARNING: hipHostGetDevicePointer: %s (using host address)\n",
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
int kv_cache_block_tokens(const inference_kv_cache_t *c) { return c ? c->block_tokens : 0; }
int kv_cache_block_shift(const inference_kv_cache_t *c) { return c ? c->block_shift : 0; }
size_t kv_cache_block_bytes(const inference_kv_cache_t *c) { return c ? c->block_bytes : 0; }
long long kv_cache_block_stride(const inference_kv_cache_t *c) { return c ? (long long)(c->block_bytes / sizeof(__half)) : 0; }

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
