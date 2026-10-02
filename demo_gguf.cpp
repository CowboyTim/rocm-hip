/* ---------------------------------------------------------------------------
 * demo_gguf -- exercise the gguf_mmap API (Phase 1)
 *
 * Usage: demo_gguf <model.gguf> [n_ctx]
 *
 * Demonstrates: load_model, tensor introspection, KV queries, n_ctx config
 * --------------------------------------------------------------------------- */

#include "gguf_mmap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* memset */

static const char *type_name(int t) {
    switch (t) {
        case 0: return "F32";     case 1: return "F16";   case 2: return "Q4_0";
        case 3: return "Q4_1";    case 6: return "Q5_0";  case 7: return "Q5_1";
        case 8: return "Q8_0";    case 9: return "Q8_1";  case 10:return "Q2_K";
        case 11:return "Q3_K";    case 12:return "Q4_K";  case 13:return "Q5_K";
        case 14:return "Q6_K";    case 15:return "Q8_K";  case 39:return "MXFP4";
        case 40:return "NVFP4";
        default: return "?";
    }
}

static void print_tensor_info(const gguf_model_t *m, int64_t idx) {
    printf("  [%3lld] %-50s %6s ne=", (long long)idx,
           gguf_tensor_name(m, idx), type_name(gguf_tensor_type(m, idx)));

    uint8_t nd = gguf_tensor_dims(m, idx);
    int64_t *ne = gguf_tensor_ne(m, idx);
    for (uint8_t d = 0; d < nd; ++d) {
        if (d > 0) printf("x");
        printf("%lld", (long long)ne[d]);
    }
    printf(" bs=%zu ts=%zu offset=%llu nbytes=%zu\n",
           (unsigned long)gguf_tensor_block_size(m, idx),
           (unsigned long)gguf_tensor_type_size(m, idx),
           (unsigned long long)gguf_tensor_offset(m, idx),
           (unsigned long)gguf_tensor_nbytes(m, idx));
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_ctx]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int64_t n_ctx = 512; /* default */

    if (argc >= 3) n_ctx = atoll(argv[2]);

    printf("=== GGUF mmap reader demo ===\n\n");

    /* Load model via mmap() -- no fread(), no copy */
    printf("[1] Loading '%s' via mmap()...\n", path);
    gguf_model_t *model = gguf_mmap_open(path);
    if (!model) {
        fprintf(stderr, "Failed to load '%s'\n", path);
        return 1;
    }

    /* Print basic metadata */
    printf("\n[2] Model metadata:\n");
    printf("  Version:            %u\n",       gguf_version(model));
    printf("  Alignment:          %zu bytes\n",  gguf_alignment(model));
    printf("  Data blob offset:   %zu bytes\n",  gguf_data_offset(model));
    printf("  Tensors:            %lld\n",       (long long)gguf_n_tensors(model));
    printf("  KV pairs:           %lld\n",       (long long)gguf_n_kv(model));

    /* Print KV metadata */
    printf("\n[3] Key-value metadata (%lld pairs):\n", (long long)gguf_n_kv(model));
    int64_t nkv = gguf_n_kv(model);
    for (int64_t i = 0; i < nkv && i < 80; ++i) {
        const char *key = gguf_kv_key(model, i);
        if (strstr(key, "general.") || strstr(key, "attention.") ||
            strstr(key, "tokenizer.") || strstr(key, "embedding") ||
            strstr(key, "context_length")) {
            uint32_t val;
            if (gguf_kv_get_u32(model, i, &val) == 0) {
                printf("  %-40s = %u\n", key, val);
            } else {
                printf("  %-40s = [non-scalar or array]\n", key);
            }
        }
    }

    /* Model-level queries */
    printf("\n[4] Model-level queries:\n");
    printf("  arch:              %s\n",       gguf_arch(model));
    printf("  vocab_name:        %s\n",       gguf_vocab_name(model));
    printf("  vocab_size:        %lld\n",     (long long)gguf_vocab_size(model));
    printf("  head_count:        %lld\n",     (long long)gguf_head_count(model));
    printf("  head_count_kv:     %lld\n",     (long long)gguf_head_count_kv(model));
    printf("  context_length:    %lld\n",     (long long)gguf_context_length(model));

    /* Print all tensors */
    int64_t nt = gguf_n_tensors(model);
    printf("\n[5] Tensor listing (%lld total):\n", (long long)nt);

    if (nt > 100) {
        /* Show summary by type */
        int counts[43]; memset(counts, 0, sizeof(counts));
        for (int64_t i = 0; i < nt; ++i) {
            int t = gguf_tensor_type(model, i);
            if (t >= 0 && t < 43) counts[t]++;
        }
        printf("  Type breakdown:\n");
        for (int t = 0; t < 43; ++t) {
            if (counts[t] > 0) {
                printf("    %-8s (%2d): %d tensors, ~%.1f GB data\n",
                       type_name(t), t, counts[t],
                       (double)(counts[t]) * 1e-9); /* rough estimate */
            }
        }
        printf("\n  First 10 tensors:\n");
        for (int64_t i = 0; i < 10 && i < nt; ++i) print_tensor_info(model, i);
        printf("  ... (%lld more)\n", (long long)(nt - 10));
    } else {
        for (int64_t i = 0; i < nt; ++i) print_tensor_info(model, i);
    }

    /* Configure KV cache context length */
    printf("\n[6] KV cache configuration:\n");
    printf("  Default n_ctx:       %lld\n", (long long)gguf_get_n_ctx(model));

    gguf_set_n_ctx(model, n_ctx);
    printf("  Set n_ctx to:      %lld\n", (long long)n_ctx);

    int64_t stored_ctx = gguf_get_n_ctx(model);
    printf("  Verified n_ctx:    %lld %s\n", (long long)stored_ctx,
           stored_ctx == n_ctx ? "OK" : "MISMATCH!");

    /* Free model -- unmaps the file and frees all metadata */
    printf("\n[7] Freeing model...\n");
    gguf_mmap_close(model);
    printf("Done. File unmapped, all resources released.\n");

    return 0;
}
