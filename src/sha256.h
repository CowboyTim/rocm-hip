// sha256.h -- SHA-256 (FIPS 180-4), for naming KV block-store files by the chained hash of their tokens.

#ifndef SHA256_H
#define SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t h[8];
    uint64_t len;       // bytes hashed so far
    uint8_t buf[64];
    size_t n;           // bytes in buf
} sha256_ctx;

void sha256_init(sha256_ctx* c);
void sha256_update(sha256_ctx* c, const void* data, size_t len);
void sha256_final(sha256_ctx* c, uint8_t out[32]);

#ifdef __cplusplus
}
#endif

#endif // SHA256_H
