// kv_cache.h -- Phase 3: unified-memory KV cache manager
//
// One file-backed /dev/shm buffer holds every layer's K and V slots as FP16. The whole buffer is
// mmap(MAP_SHARED) + hipHostRegister'ed so GPU kernels read/write these pages in place -- no copy.
// On Strix Halo's UMA fabric the device pointer equals the host pointer (verified at runtime).
//
// Layout inside the file: blocks of B = block_tokens positions (a power of two; Phase 22b: one block can become
// one file of a block store). Block b holds every layer's K and V rows for positions [b B, (b + 1) B):
//   block b: [ K_0 | V_0 | K_1 | V_1 | ... | K_{L-1} | V_{L-1} ]   each segment B rows of d_kv __half values
// with d_kv = n_heads_kv * head_dim. So row p of layer l is at
//   base + (p / B) * block_bytes + (2 l + is_v) * B * row_bytes + (p % B) * row_bytes
// and kernels address it from the layer's K (or V) row 0 as (p >> log2 B) * block_stride + (p & (B - 1)) * d_kv
// (in __half units; kv_cache_block_shift / kv_cache_block_stride). n_ctx is rounded up to whole blocks.

#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* C++ interface: exposes __half rows. Include from C++ only. */
#if !defined(__cplusplus)
#error "kv_cache.h is a C++ interface (__half storage); include it from C++ translation units"
#endif
#include <hip/hip_fp16.h>

extern "C" {

typedef struct inference_kv_cache inference_kv_cache_t;

typedef struct {
    int n_layers;      /* transformer layers, each with its own K/V region pair */
    int n_ctx;         /* max context tokens per sequence */
    int n_embd;        /* hidden size (informational: q/k/v projections map into it) */
    int head_dim;      /* embedding dim per attention head */
    int n_heads_q;     /* query heads */
    int n_heads_kv;    /* key/value heads (GQA: may be < n_heads_q) */
    int block_tokens;  /* positions per block, a power of two; 0: KV_CACHE_DEFAULT_BLOCK */
} kv_cache_config_t;

#define KV_CACHE_DEFAULT_BLOCK 256

/* Create + map + pin the cache in shm_path. Returns NULL on error (logs to stderr). */
inference_kv_cache_t *kv_cache_create(const char *shm_path, const kv_cache_config_t *cfg);

/* Unregister, munmap and close. No-op on NULL. */
void kv_cache_free(inference_kv_cache_t *cache);

int    kv_cache_n_layers(const inference_kv_cache_t *c);
int    kv_cache_n_ctx(const inference_kv_cache_t *c);
int    kv_cache_d_kv(const inference_kv_cache_t *c);   /* __half values per token row = n_heads_kv*head_dim */
size_t kv_cache_total_bytes(const inference_kv_cache_t *c);

/* Block geometry (see the layout comment): positions per block, its log2, bytes per block (all layers' K and V), and
   the block stride in __half units for kernels (block_bytes / 2) */
int    kv_cache_block_tokens(const inference_kv_cache_t *c);
int    kv_cache_block_shift(const inference_kv_cache_t *c);
size_t kv_cache_block_bytes(const inference_kv_cache_t *c);
long long kv_cache_block_stride(const inference_kv_cache_t *c);
/* Host pointer to the whole mapping (blocks in position order, see file-top comment). */
void  *kv_cache_host_base(const inference_kv_cache_t *c);

/* Device pointer of the same buffer; equals the host base when zero-copy pinning succeeded. */
void  *kv_cache_device_ptr(const inference_kv_cache_t *c);

/* True when hipHostGetDevicePointer returned the host address (UMA identity holds here). */
bool   kv_cache_zero_copy_ok(const inference_kv_cache_t *c);

/* Row p of layer l: d_kv contiguous __half values inside the mapped file. NULL if out of range. */
__half        *kv_cache_k_row(inference_kv_cache_t *c, int layer, int pos);
const __half  *kv_cache_k_row_const(const inference_kv_cache_t *c, int layer, int pos);
__half        *kv_cache_v_row(inference_kv_cache_t *c, int layer, int pos);
const __half  *kv_cache_v_row_const(const inference_kv_cache_t *c, int layer, int pos);

/* Wipe every byte of the mapping (fresh context / full reset of stored data). */
void kv_cache_zero(inference_kv_cache_t *c);

/* Position bookkeeping for sequential generation: returns next position index (post-increment) or -1 if full. */
int  kv_cache_advance(inference_kv_cache_t *c);
int  kv_cache_n_stored(const inference_kv_cache_t *c);
void kv_cache_reset_positions(inference_kv_cache_t *c);   /* n_stored = 0; does not wipe data */

} /* extern "C" */

#endif /* KV_CACHE_H */
