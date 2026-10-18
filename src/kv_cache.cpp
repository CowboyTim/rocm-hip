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
#include <errno.h>
#include <unistd.h>

#include <string>
#include <vector>

#include <hip/hip_runtime.h>

// Full definition of the opaque handle. The tag name MUST match the header's forward
// declaration so this definition completes that type within this translation unit.
struct inference_kv_cache {
    void* base = nullptr;       // host address of the whole range (slots back to back)
    void* dev_base = nullptr;   // device pointer of the same range
    bool zero_copy_ok = false;  // true when dev_base == base (UMA identity)
    size_t total_bytes = 0;     // whole range
    int block_tokens = 0;       // positions per block (power of two)
    int block_shift = 0;        // log2(block_tokens)
    size_t row_bytes = 0;       // one position's K (or V) row of one layer
    size_t block_bytes = 0;     // every layer's K and V for block_tokens positions
    kv_cache_config_t cfg{};
    int n_stored = 0;           // tokens stored so far (positions [0, n_stored))
    // Slot i = block i of the range, mapped from its own file (Phase 22b): a private anonymous file
    // (O_TMPFILE in dir) or a shared file of the block store. Each slot is registered on its own, so one can be
    // remapped without touching the others.
    std::string dir;
    std::vector<int> slot_fd;
    std::vector<char> slot_shared;  // 1: a store file (immutable: never written)
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

// A new anonymous file of the cache's size per slot in c->dir: O_TMPFILE (no name until published), else a
// named temp file unlinked at once (works, but cannot be published)
int new_slot_file(inference_kv_cache_t* c, bool* publishable) {
    int fd = open(c->dir.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0644);
    *publishable = fd >= 0;
    if (fd < 0) {
        std::string tmpl = c->dir + "/.kvslot.XXXXXX";
        fd = mkostemp(&tmpl[0], O_CLOEXEC);
        if (fd >= 0) unlink(tmpl.c_str());
    }
    if (fd >= 0 && ftruncate(fd, (off_t)c->block_bytes) != 0) {
        close(fd);
        fd = -1;
    }
    return fd;
}

char* slot_addr(const inference_kv_cache_t* c, int slot) { return (char*)c->base + (size_t)slot * c->block_bytes; }

// Map fd at slot's address (MAP_FIXED over whatever is there) and register it for the GPU. A file opened read-only
// (a shared block) is mapped and registered read-only: a stray write faults instead of corrupting it.
bool map_slot(inference_kv_cache_t* c, int slot, int fd, bool was_registered) {
    char* a = slot_addr(c, slot);
    const bool ro = (fcntl(fd, F_GETFL) & O_ACCMODE) == O_RDONLY;
    if (was_registered) (void)hipHostUnregister(a);
    if (mmap(a, c->block_bytes, ro ? PROT_READ : PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED) {
        INFER_LOG("kv_cache: mmap slot %d: %s\n", slot, strerror(errno));
        return false;
    }
    // Coarse-grained: GPU-cacheable, coherent with the CPU at kernel boundaries and syncs. Still zero-copy.
    const hipError_t st = hipHostRegister(a, c->block_bytes,
                                          hipExtHostRegisterCoarseGrained | (ro ? hipHostRegisterReadOnly : 0));
    if (st != hipSuccess) {
        INFER_LOG("kv_cache: hipHostRegister slot %d: %s\n", slot, hipGetErrorString(st));
        return false;
    }
    return true;
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
    // A block is mapped from its own file, so it is padded to whole pages (the stride kernels use includes it)
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    c->block_bytes = (2 * (size_t)f.n_layers * B * c->row_bytes + page - 1) / page * page;
    c->total_bytes = c->block_bytes * (size_t)(c->cfg.n_ctx / B);

    // The slot files live in shm_path if it is a directory, else in its directory
    struct stat st_dir;
    if (stat(shm_path, &st_dir) == 0 && S_ISDIR(st_dir.st_mode)) {
        c->dir = shm_path;
    } else {
        c->dir = shm_path;
        const size_t slash = c->dir.rfind('/');
        c->dir = slash == std::string::npos ? "." : slash == 0 ? "/" : c->dir.substr(0, slash);
    }
    const int n_slots = c->cfg.n_ctx / B;
    c->base = mmap(NULL, c->total_bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);  // reserve
    if (c->base == MAP_FAILED) { INFER_LOG("kv_cache: reserve: %s\n", strerror(errno)); delete c; return nullptr; }
    c->slot_fd.assign(n_slots, -1);
    c->slot_shared.assign(n_slots, 0);
    for (int i = 0; i < n_slots; ++i) {
        bool publishable;
        c->slot_fd[i] = new_slot_file(c, &publishable);
        if (c->slot_fd[i] < 0 || !map_slot(c, i, c->slot_fd[i], false)) {
            INFER_LOG("kv_cache: slot %d in %s failed\n", i, c->dir.c_str());
            if (c->slot_fd[i] >= 0) close(c->slot_fd[i]);
            c->slot_fd[i] = -1;
            kv_cache_free(c);
            return nullptr;
        }
    }

    void* dev = nullptr;
    hipError_t st = hipHostGetDevicePointer(&dev, c->base, 0);
    if (st == hipSuccess) {
        c->dev_base = dev;
        c->zero_copy_ok = (dev == c->base);
    } else {
        // Not fatal on a UMA APU where identity is expected; keep host address as the device view.
        INFER_LOG("kv_cache: WARNING: hipHostGetDevicePointer: %s (using host address)\n", hipGetErrorString(st));
        c->dev_base = c->base;
        c->zero_copy_ok = true;
    }
    return c;
}

void kv_cache_free(inference_kv_cache_t *c) {
    if (!c) return;
    if (c->base && c->base != MAP_FAILED) {
        for (size_t i = 0; i < c->slot_fd.size(); ++i)
            if (c->slot_fd[i] >= 0) (void)hipHostUnregister(slot_addr(c, (int)i));
        munmap(c->base, c->total_bytes);
    }
    for (int fd : c->slot_fd)
        if (fd >= 0) close(fd);
    delete c;
}

int kv_cache_n_slots(const inference_kv_cache_t *c) { return c ? (int)c->slot_fd.size() : 0; }

int kv_cache_slot_shared(const inference_kv_cache_t *c, int slot) {
    return c && slot >= 0 && slot < (int)c->slot_fd.size() ? c->slot_shared[slot] : 0;
}

int kv_cache_slot_attach(inference_kv_cache_t *c, int slot, int fd) {
    if (!c || slot < 0 || slot >= (int)c->slot_fd.size() || fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size != c->block_bytes) {
        INFER_LOG("kv_cache: attach slot %d: file is not one block (%zu bytes)\n", slot, c->block_bytes);
        return -1;
    }
    const int nfd = dup(fd);
    if (nfd < 0 || !map_slot(c, slot, nfd, true)) {
        if (nfd >= 0) close(nfd);
        return -1;
    }
    close(c->slot_fd[slot]);
    c->slot_fd[slot] = nfd;
    c->slot_shared[slot] = 1;
    return 0;
}

int kv_cache_slot_make_private(inference_kv_cache_t *c, int slot, int copy) {
    if (!c || slot < 0 || slot >= (int)c->slot_fd.size()) return -1;
    bool publishable;
    const int nfd = new_slot_file(c, &publishable);
    if (nfd < 0) return -1;
    if (copy) {  // the current contents, through a temporary mapping of the new file
        void* t = mmap(nullptr, c->block_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, nfd, 0);
        if (t == MAP_FAILED) { close(nfd); return -1; }
        memcpy(t, slot_addr(c, slot), c->block_bytes);
        munmap(t, c->block_bytes);
    }
    if (!map_slot(c, slot, nfd, true)) { close(nfd); return -1; }
    close(c->slot_fd[slot]);
    c->slot_fd[slot] = nfd;
    c->slot_shared[slot] = 0;
    return 0;
}

int kv_cache_slot_publish(inference_kv_cache_t *c, int slot, const char *path) {
    if (!c || slot < 0 || slot >= (int)c->slot_fd.size() || !path) return -1;
    char proc[64];
    snprintf(proc, sizeof(proc), "/proc/self/fd/%d", c->slot_fd[slot]);
    if (linkat(AT_FDCWD, proc, AT_FDCWD, path, AT_SYMLINK_FOLLOW) != 0) {
        if (errno == EEXIST) return 1;  // already published (by another session or process)
        INFER_LOG("kv_cache: publish slot %d as %s: %s\n", slot, path, strerror(errno));
        return -1;
    }
    c->slot_shared[slot] = 1;  // now visible to others: never written again
    return 0;
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
