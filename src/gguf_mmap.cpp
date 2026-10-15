/* ---------------------------------------------------------------------------
 * gguf_mmap -- standalone mmap-based GGUF reader (Phase 1)
 * Loads a GGUF file via mmap(), parses header + KV pairs + tensor metadata.
 * --------------------------------------------------------------------------- */

#include "gguf_mmap.h"
#include "quant_types.h"

#include <stdio.h>      /* fprintf, stderr */
#include <stdlib.h>     /* calloc, free, malloc */
#include <string.h>     /* memcpy, strcmp, strlen */
#include <fcntl.h>      /* open, O_RDONLY */
#include <unistd.h>     /* close, munmap */
#include <sys/mman.h>   /* mmap, MAP_FAILED */
#include <sys/stat.h>   /* ::stat, struct stat */
#include <cerrno>       /* errno */

static const char  GGUF_MAGIC[]       = "GGUF";
static const uint32_t GGUF_VERSION    = 3;
static const size_t  GGUF_ALIGN       = 32;

static inline size_t align_up(size_t x, size_t n) { return ((x + n - 1) & ~(n - 1)); }

typedef uint16_t ggml_half;

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* Block sizes (elements per block for quantized types) */
/* Block geometry comes from the single per-type table in quant_types (Phase 20 Step 3). */
static inline int64_t get_block_size(int type) {
    const qt_traits *t = qt_get_traits(type);
    return t ? t->blck_size : -1;
}

static inline size_t get_type_size(int type) {
    const qt_traits *t = qt_get_traits(type);
    return t ? (size_t)t->type_size : (size_t)-1;
}

/* Check if a tensor type stores scales internally within each block.
 * Returns 1 for self-contained types (Q4_0, Q4_1, Q8_0, TYPE100).
 * Returns 0 for types requiring external .scales companion tensors. */

int gguf_type_is_self_contained(int type) {
    switch (type) {
        /* Standard llama.cpp quantization types - all self-contained */
        case 2:  /* Q4_0: {half scale, uint8[16]} per 32 values */
        case 3:  /* Q4_1: {half scale, half min, uint8[16]} per 32 values */
        case 7:  /* Q5_0/Q8_0 variants depending on GGUF version */
        case 8:  /* Q8_0: {half scale, int8[32]} per 32 values */
            return 1;

        /* Extended/custom quantization types */
        case 13: /* Qwen3.5 custom (TYPE13): block_size=256, UE4M3 scales per-16 elements */
        case 14: /* Qwen3.5 variant of Q4_K/Q4_1 equivalent — self-contained block layout */
        case 100:/* TYPE100 = ROCmFP4 Codebook10: UE4M3 scale + nibbles per block */
            return 1;

        default:
            /* For unknown types, treat as self-contained by default.
             * Quantized types store scales internally; only FP16/F32/etc. need no scales at all.
             * If a type truly requires external .scales, it will be added to this list explicitly. */
            fprintf(stderr, "[gguf_type_is_self_contained] unknown type %d, assuming self-contained\n", type);
            return 1;
    }
}

/* ---------------------------------------------------------------------------
 * Internal structures
 * --------------------------------------------------------------------------- */

struct tensor_meta {
    char         name[64];
    uint8_t      n_dims;       /* actual dimensions count (1-4) */
    int64_t      ne[4];        /* dimension sizes; ne[i>=n_dims]==1 */
    int          type;         /* ggml_type enum value */
    uint64_t     data_offset;  /* byte offset from start of data blob */
};

/* KV scalar value with explicit type tag for correct retrieval */
struct kv_scalar {
    uint8_t type;   /* original gguf_type */
    union {
        uint8_t  v_u8;  int8_t   v_i8;  uint16_t v_u16; int16_t  v_i16;
        uint32_t v_u32; int32_t  v_i32; float      v_f32; bool     v_bool;
        uint64_t v_u64; int64_t  v_i64; double     v_f64;
    };
};

struct kv_entry {
    char            *key;         /* allocated string, owned by us */
    uint8_t          is_array : 1;/* 0=scalar, 1=array */
    uint8_t          arr_type : 7;/* element type (only when is_array) */
    struct kv_scalar scalar;      /* used only when !is_array */
    uint64_t         arr_n;       /* number of elements (only when is_array) */
    char             *str_val;    /* allocated string (for string-type scalars) */
    char            **arr_strings;/* for string arrays: array of strings */
    size_t           arr_str_n;   /* number of strings in arr_strings */
};

/* Opaque handle matching the public API declaration */
typedef struct gguf_model gguf_model_t;

struct gguf_model {
    void           *map_ptr;         /* base pointer from mmap() */
    size_t          map_size;        /* file size we mapped */
    uint32_t        version;         /* GGUF version */
    size_t          alignment;       /* effective alignment (default 32) */
    size_t          data_off;        /* data blob start offset from file begin */
    int64_t         n_tensors;       /* total tensors in file */
    struct tensor_meta *tensors;     /* heap-allocated array[n_tensors] */
    int64_t         n_kv;            /* total KV pairs */
    struct kv_entry *kvs;            /* heap-allocated array[n_kv] */
    int64_t         n_ctx;           /* requested context length, default 512 */
};

/* ---------------------------------------------------------------------------
 * Helpers: read values at current position in the mapped buffer
 * --------------------------------------------------------------------------- */

static char *read_string(const uint8_t **pp) {
    const uint8_t *p = *pp;
    uint64_t len; memcpy(&len, p, 8); p += 8;
    if (len > (1ULL << 30)) return NULL;
    char *s = (char *)malloc(len + 1); if (!s) return NULL;
    memcpy(s, p, len); s[len] = '\0'; p += len;
    *pp = p; return s;
}

/* ---------------------------------------------------------------------------
 * Core: load and parse a single GGUF file via mmap()
 * --------------------------------------------------------------------------- */

static gguf_model_t *load_gguf(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "gguf_mmap: cannot stat '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "gguf_mmap: cannot open '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    void *mapped = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "gguf_mmap: mmap failed for '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    const uint8_t *base = (const uint8_t *)mapped;

    /* Verify magic + version */
    if (memcmp(base, GGUF_MAGIC, 4) != 0 || *(const uint32_t *)(base + 4) != GGUF_VERSION) {
        fprintf(stderr, "gguf_mmap: invalid GGUF in '%s' (bad magic/version)\n", path);
        munmap(mapped, st.st_size); return NULL;
    }

    gguf_model_t *model = (gguf_model_t *)calloc(1, sizeof(gguf_model_t));
    if (!model) { munmap(mapped, st.st_size); return NULL; }

    model->map_ptr   = mapped;
    model->map_size  = st.st_size;
    model->version   = *(const uint32_t *)(base + 4);
    model->alignment = GGUF_ALIGN;

    const uint8_t *p = base + 8; /* skip past magic+version header */
    int64_t n_tensors, n_kv;
    memcpy(&n_tensors, p, 8); p += 8;
    memcpy(&n_kv,      p, 8); p += 8;

    model->n_tensors = n_tensors;
    model->n_kv      = n_kv;

    size_t nt_alloc = (size_t)n_tensors > 0 ? (size_t)n_tensors : 1;
    size_t nk_alloc = (size_t)n_kv     > 0 ? (size_t)n_kv     : 1;
    model->tensors = (struct tensor_meta *)calloc(nt_alloc, sizeof(struct tensor_meta));
    model->kvs     = (struct kv_entry *)calloc(nk_alloc, sizeof(struct kv_entry));

    if (!model->tensors || !model->kvs) goto fail;

    /* --- Parse KV pairs --- */
    for (int64_t i = 0; i < n_kv; ++i) {
        struct kv_entry *kv = &model->kvs[i];

        char *key = read_string(&p); if (!key) goto fail;
        kv->key = key;

        int32_t type_id; memcpy(&type_id, p, 4); p += 4;

        if (type_id == GGUF_TYPE_STRING) {
            /* String value (general.architecture, tokenizer.ggml.model/pre, ...) */
            kv->str_val = read_string(&p);
            if (!kv->str_val) goto fail;
            kv->scalar.type = GGUF_TYPE_STRING;
        } else if (type_id == GGUF_TYPE_ARRAY) {
            int32_t arr_type; uint64_t arr_n;
            memcpy(&arr_type, p, 4); p += 4;
            memcpy(&arr_n, p, 8); p += 8;

            /* String arrays: store strings for tokenizer metadata */
            if ((enum gguf_type)arr_type == GGUF_TYPE_STRING) {
                kv->arr_strings = (char**)calloc(arr_n, sizeof(char*));
                if (!kv->arr_strings) goto fail;
                for (uint64_t j = 0; j < arr_n; ++j) {
                    char *s = read_string(&p);
                    if (!s) goto fail;
                    kv->arr_strings[j] = s;
                }
                kv->arr_str_n = (size_t)arr_n;
            } else {
                size_t elem_sz = 0; switch ((enum gguf_type)arr_type) {
                    case GGUF_TYPE_UINT8:   elem_sz=1; break; case GGUF_TYPE_INT8:    elem_sz=1; break;
                    case GGUF_TYPE_UINT16:  elem_sz=2; break; case GGUF_TYPE_INT16:   elem_sz=2; break;
                    case GGUF_TYPE_UINT32:  elem_sz=4; break; case GGUF_TYPE_INT32:   elem_sz=4; break;
                    case GGUF_TYPE_FLOAT32: elem_sz=4; break; case GGUF_TYPE_BOOL:    elem_sz=1; break;
                    case GGUF_TYPE_UINT64:  elem_sz=8; break; case GGUF_TYPE_INT64:   elem_sz=8; break;
                    case GGUF_TYPE_FLOAT64: elem_sz=8; break; default: goto fail;
                }
                p += arr_n * elem_sz;
            }

            kv->is_array     = 1;
            kv->is_array     = 1;
            kv->arr_type     = (uint8_t)arr_type;
            kv->arr_n        = arr_n;
        } else {
            /* Scalar value -- read into union based on type */
            kv->is_array      = 0;
            kv->scalar.type   = (uint8_t)type_id;

            switch ((enum gguf_type)type_id) {
                case GGUF_TYPE_UINT8:   memcpy(&kv->scalar.v_u8, p, 1); ++p; break;
                case GGUF_TYPE_INT8:    memcpy(&kv->scalar.v_i8, p, 1); ++p; break;
                case GGUF_TYPE_UINT16:  memcpy(&kv->scalar.v_u16, p, 2); p+=2; break;
                case GGUF_TYPE_INT16:   memcpy(&kv->scalar.v_i16, p, 2); p+=2; break;
                case GGUF_TYPE_UINT32:  memcpy(&kv->scalar.v_u32, p, 4); p+=4; break;
                case GGUF_TYPE_INT32:   memcpy(&kv->scalar.v_i32, p, 4); p+=4; break;
                case GGUF_TYPE_FLOAT32: memcpy(&kv->scalar.v_f32, p, 4); p+=4; break;
                case GGUF_TYPE_BOOL:    memcpy(&kv->scalar.v_bool, p, 1); ++p; break;
                case GGUF_TYPE_STRING:  { char *s = read_string(&p); free(s); } break;
                case GGUF_TYPE_UINT64:  memcpy(&kv->scalar.v_u64, p, 8); p+=8; break;
                case GGUF_TYPE_INT64:   memcpy(&kv->scalar.v_i64, p, 8); p+=8; break;
                case GGUF_TYPE_FLOAT64: memcpy(&kv->scalar.v_f64, p, 8); p+=8; break;
                default: goto fail;
            }

            /* Track specific KV pairs we care about */
            if (strcmp(key, "general.alignment") == 0) {
                model->alignment = kv->is_array ? 32 : (size_t)(uint64_t)kv->scalar.v_u32;
            }
        }
    }

    /* --- Parse tensor metadata --- */
    for (int64_t i = 0; i < n_tensors; ++i) {
        struct tensor_meta *tm = &model->tensors[i];

        char *name = read_string(&p); if (!name) goto fail;
        size_t nl = strlen(name);
        if (nl >= sizeof(tm->name)) nl = sizeof(tm->name) - 1;
        memcpy(tm->name, name, nl); tm->name[nl] = '\0'; free(name);

        uint32_t nd; memcpy(&nd, p, 4); p += 4;
        if (nd < 1 || nd > 4) goto fail;
        tm->n_dims = (uint8_t)nd;

        for (uint8_t d = 0; d < nd; ++d) { int64_t dim; memcpy(&dim, p, 8); p += 8; tm->ne[d] = dim; }

        int32_t tid; memcpy(&tid, p, 4); p += 4;
        tm->type = tid;

        uint64_t doff; memcpy(&doff, p, 8); p += 8;
        tm->data_offset = doff;
    }

    /* Data blob starts at aligned offset after all tensor info */
    model->data_off   = align_up((size_t)(p - base), model->alignment);
    return model;

fail:
    fprintf(stderr, "gguf_mmap: corrupted structure in '%s'\n", path);
    munmap(mapped, st.st_size);
    for (int64_t i = 0; i < n_kv; ++i) {
        free(model->kvs[i].key);
        if (model->kvs[i].arr_strings) {
            for (size_t j = 0; j < model->kvs[i].arr_str_n; ++j) free(model->kvs[i].arr_strings[j]);
            free(model->kvs[i].arr_strings);
        }
    }
    free(model->tensors); free(model->kvs); free(model);
    return NULL;
}

/* ---------------------------------------------------------------------------
 * Public API implementation
 * --------------------------------------------------------------------------- */

gguf_model_t *inference_load_model(const char *path) {
    if (!path) return NULL;
    return load_gguf(path);
}

void inference_free_model(gguf_model_t *model) {
    if (!model) return;
    for (int64_t i = 0; i < model->n_kv; ++i) {
        free(model->kvs[i].str_val);
        if (model->kvs[i].arr_strings) {
            for (size_t j = 0; j < model->kvs[i].arr_str_n; ++j) free(model->kvs[i].arr_strings[j]);
            free(model->kvs[i].arr_strings);
        }
        free(model->kvs[i].key);
    }
    free(model->tensors); free(model->kvs);
    if (model->map_ptr) munmap(model->map_ptr, model->map_size);
    free(model);
}

/* --- Tensor introspection --- */

int64_t      gguf_n_tensors(const gguf_model_t *m)   { return m ? m->n_tensors : 0; }

const char  *gguf_tensor_name(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return "(null)";
    return m->tensors[idx].name;
}

int          gguf_tensor_type(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return -1;
    return m->tensors[idx].type;
}

int64_t     *gguf_tensor_ne(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return NULL;
    return m->tensors[idx].ne;
}

uint8_t      gguf_tensor_dims(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return 0;
    return m->tensors[idx].n_dims;
}

size_t       gguf_tensor_block_size(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return (size_t)-1;
    return get_block_size(m->tensors[idx].type);
}

size_t       gguf_tensor_type_size(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return (size_t)-1;
    return get_type_size(m->tensors[idx].type);
}

size_t gguf_tensor_row_size(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return (size_t)-1;
    const struct tensor_meta *tm = &m->tensors[idx];
    int64_t bs = get_block_size(tm->type); size_t ts = get_type_size(tm->type);
    if (bs <= 0 || ts == (size_t)-1) return (size_t)-1;
    int64_t n_blocks = (tm->ne[0] + bs - 1) / bs;
    return align_up(n_blocks * ts, GGUF_ALIGN); /* aligned to block boundary */
}

uint64_t gguf_tensor_offset(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return UINT64_MAX;
    return m->tensors[idx].data_offset;
}

size_t gguf_tensor_nbytes(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_tensors) return (size_t)-1;
    const struct tensor_meta *tm = &m->tensors[idx];
    int64_t bs  = get_block_size(tm->type); size_t ts = get_type_size(tm->type);
    if (bs <= 0 || ts == (size_t)-1) return (size_t)-1;
    int64_t nelem = 1; for (int d = 0; d < tm->n_dims; ++d) nelem *= tm->ne[d];
    return (size_t)(nelem / bs) * ts;
}

size_t gguf_tensor_nbytes_pad(const gguf_model_t *m, int64_t idx) {
    /* In GGUF the data blob is contiguous and aligned at start. */
    return gguf_tensor_nbytes(m, idx);
}

/* --- KV metadata --- */

int64_t      gguf_n_kv(const gguf_model_t *m)   { return m ? m->n_kv : 0; }

const char  *gguf_kv_key(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_kv) return "(null)";
    return m->kvs[idx].key;
}

/* Read scalar KV values by index. Returns 0 on success, -1 on error/type mismatch. */
int32_t gguf_kv_get_u8(const gguf_model_t *m, int64_t idx, uint8_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_UINT8:  *out=kv->scalar.v_u8; break;
        case GGUF_TYPE_INT8:   *out=(uint8_t)kv->scalar.v_i8; break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_i8(const gguf_model_t *m, int64_t idx, int8_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_INT8:   *out=kv->scalar.v_i8; break;
        case GGUF_TYPE_UINT8:  *out=(int8_t)kv->scalar.v_u8; break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_u16(const gguf_model_t *m, int64_t idx, uint16_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_UINT8:  *out=kv->scalar.v_u8; break;
        case GGUF_TYPE_INT8:   *out=(uint16_t)kv->scalar.v_i8; break;
        case GGUF_TYPE_UINT16: *out=kv->scalar.v_u16; break;
        case GGUF_TYPE_INT16:  *out=(uint16_t)kv->scalar.v_i16; break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_i16(const gguf_model_t *m, int64_t idx, int16_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_INT8:   *out=kv->scalar.v_i8; break;
        case GGUF_TYPE_UINT8:  *out=(int16_t)kv->scalar.v_u8; break;
        case GGUF_TYPE_INT16:  *out=kv->scalar.v_i16; break;
        case GGUF_TYPE_UINT16: *out=(int16_t)kv->scalar.v_u16; break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_u32(const gguf_model_t *m, int64_t idx, uint32_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_UINT8:  *out=kv->scalar.v_u8; break;
        case GGUF_TYPE_INT8:   *out=(uint32_t)kv->scalar.v_i8; break;
        case GGUF_TYPE_UINT16: *out=kv->scalar.v_u16; break;
        case GGUF_TYPE_INT16:  *out=(uint32_t)kv->scalar.v_i16; break;
        case GGUF_TYPE_UINT32: *out=kv->scalar.v_u32; break;
        case GGUF_TYPE_INT32:  *out=(uint32_t)kv->scalar.v_i32; break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_i32(const gguf_model_t *m, int64_t idx, int32_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_UINT8:  *out=kv->scalar.v_u8; break;
        case GGUF_TYPE_INT8:   *out=kv->scalar.v_i8; break;
        case GGUF_TYPE_UINT16: *out=kv->scalar.v_u16; break;
        case GGUF_TYPE_INT16:  *out=kv->scalar.v_i16; break;
        case GGUF_TYPE_UINT32: *out=(int32_t)kv->scalar.v_u32; break;
        case GGUF_TYPE_INT32:  *out=kv->scalar.v_i32; break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_f32(const gguf_model_t *m, int64_t idx, float *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_FLOAT32:{ *out=kv->scalar.v_f32; } break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_u64(const gguf_model_t *m, int64_t idx, uint64_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_UINT8:  *out=kv->scalar.v_u8; break;
        case GGUF_TYPE_INT8:   *out=(uint64_t)(uint8_t)kv->scalar.v_i8; break;
        case GGUF_TYPE_UINT16: *out=kv->scalar.v_u16; break;
        case GGUF_TYPE_INT16:  *out=(uint64_t)(int16_t)kv->scalar.v_i16; break;
        case GGUF_TYPE_UINT32: *out=kv->scalar.v_u32; break;
        case GGUF_TYPE_INT32:  *out=(uint64_t)(int32_t)kv->scalar.v_i32; break;
        case GGUF_TYPE_UINT64: *out=kv->scalar.v_u64; break;
        default: return -1;
    }
    return 0;
}

int32_t gguf_kv_get_i64(const gguf_model_t *m, int64_t idx, int64_t *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_UINT8:  *out=kv->scalar.v_u8; break;
        case GGUF_TYPE_INT8:   *out=kv->scalar.v_i8; break;
        case GGUF_TYPE_UINT16: *out=kv->scalar.v_u16; break;
        case GGUF_TYPE_INT16:  *out=kv->scalar.v_i16; break;
        case GGUF_TYPE_UINT32: *out=(int64_t)kv->scalar.v_u32; break;
        case GGUF_TYPE_INT32:  *out=kv->scalar.v_i32; break;
        case GGUF_TYPE_UINT64: *out=(int64_t)kv->scalar.v_u64; break;
        case GGUF_TYPE_INT64:  *out=kv->scalar.v_i64; break;
        default: return -1; /* type mismatch */
    }
    return 0;
}

int32_t gguf_kv_get_f64(const gguf_model_t *m, int64_t idx, double *out) {
    if (!m || !out || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return -1;
    switch (kv->scalar.type) {
        case GGUF_TYPE_FLOAT32:{ double v = (double)kv->scalar.v_f32; *out=v; } break;
        case GGUF_TYPE_FLOAT64:{ double v = kv->scalar.v_f64; *out=v; } break;
        default: return -1;
    }
    return 0;
}

/* Read scalar KV value as string. Returns NULL if not a string or key not found. */
const char *gguf_kv_str_val(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_kv) return NULL;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array == 0 && kv->scalar.type != GGUF_TYPE_STRING) return NULL;
    return kv->str_val ? kv->str_val : "";
}

/* Get KV value type by index. Returns GGUF type enum, or -1 on error. */
int gguf_kv_type(const gguf_model_t *m, int64_t idx) {
    if (!m || idx<0 || idx>=m->n_kv) return -1;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return kv->arr_type;
    return kv->scalar.type;
}

/* --- Model-level queries --- */

static int64_t find_kv_idx(const gguf_model_t *m, const char *key) {
    if (!m || !key) return -1;
    for (int64_t i = 0; i < m->n_kv; ++i) {
        if (strcmp(m->kvs[i].key, key) == 0) return i;
    }
    return -1;
}

static int64_t kv_as_int64(const gguf_model_t *m, int64_t idx) {
    if (!m || idx < 0) return 0;
    const struct kv_entry *kv = &m->kvs[idx];
    if (kv->is_array) return 0;
    switch (kv->scalar.type) {
        case GGUF_TYPE_UINT8:  return (int64_t)kv->scalar.v_u8;
        case GGUF_TYPE_INT8:   return (int64_t)kv->scalar.v_i8;
        case GGUF_TYPE_UINT16: return (int64_t)kv->scalar.v_u16;
        case GGUF_TYPE_INT16:  return (int64_t)kv->scalar.v_i16;
        case GGUF_TYPE_UINT32: return (int64_t)kv->scalar.v_u32;
        case GGUF_TYPE_INT32:  return (int64_t)kv->scalar.v_i32;
        case GGUF_TYPE_UINT64: return (int64_t)kv->scalar.v_u64;
        case GGUF_TYPE_INT64:  return kv->scalar.v_i64;
        default:               return 0;
    }
}

static int64_t find_kv_or(const gguf_model_t *m, const char *const *keys, int n_keys) {
    for (int i = 0; i < n_keys; ++i) {
        int64_t idx = find_kv_idx(m, keys[i]);
        if (idx >= 0) return idx;
    }
    return -1;
}

/* Look up "<general.architecture>.<suffix>" first (e.g. "qwen35moe.block_count"), then the legacy keys. */
static int64_t find_arch_kv(const gguf_model_t *m, const char *suffix, const char *const *legacy, int n_legacy) {
    int64_t idx = find_kv_idx(m, "general.architecture");
    if (idx >= 0 && m->kvs[idx].str_val && m->kvs[idx].str_val[0]) {
        char key[256];
        snprintf(key, sizeof(key), "%s.%s", m->kvs[idx].str_val, suffix);
        int64_t a = find_kv_idx(m, key);
        if (a >= 0) return a;
    }
    return find_kv_or(m, legacy, n_legacy);
}

int64_t gguf_arch_int(const gguf_model_t *m, const char *suffix, int64_t def) {
    int64_t idx = find_arch_kv(m, suffix, NULL, 0);
    return idx >= 0 ? kv_as_int64(m, idx) : def;
}

float gguf_arch_float(const gguf_model_t *m, const char *suffix, float def) {
    int64_t idx = find_arch_kv(m, suffix, NULL, 0);
    if (idx < 0 || m->kvs[idx].is_array) return def;
    switch (m->kvs[idx].scalar.type) {
        case GGUF_TYPE_FLOAT32: return m->kvs[idx].scalar.v_f32;
        case GGUF_TYPE_FLOAT64: return (float)m->kvs[idx].scalar.v_f64;
        default:                return (float)kv_as_int64(m, idx);
    }
}

const char *gguf_arch(const gguf_model_t *m) {
    static char arch_buf[128] = {0};
    /* Architecture key varies by model family */
    int64_t idx = find_kv_idx(m, "general.architecture");
    if (idx >= 0 && m->kvs[idx].str_val && m->kvs[idx].str_val[0]) {
        strncpy(arch_buf, m->kvs[idx].str_val, sizeof(arch_buf)-1);
        return arch_buf;
    }
    /* Check common architecture KV keys */
    const char *cand[] = {
        "tokenizer.ggml.model", "llmark", "backpack"
    };
    idx = find_kv_or(m, cand, 3);
    if (idx >= 0 && m->kvs[idx].str_val && m->kvs[idx].str_val[0]) {
        strncpy(arch_buf, m->kvs[idx].str_val, sizeof(arch_buf)-1);
        return arch_buf;
    }
    return "(unknown)";
}

const char *gguf_vocab_name(const gguf_model_t *m) {
    static char vname_buf[128] = {0};
    int64_t idx = find_kv_idx(m, "tokenizer.ggml.model");
    if (idx >= 0 && m->kvs[idx].str_val && m->kvs[idx].str_val[0]) {
        strncpy(vname_buf, m->kvs[idx].str_val, sizeof(vname_buf)-1);
        return vname_buf;
    }
    return "";
}

const char *gguf_tokenizer_type(const gguf_model_t *m) {
    /* Tokenizer type is the model name (e.g. "bpe", "gpt2") */
    return gguf_vocab_name(m);
}

const char *gguf_pre_type(const gguf_model_t *m) {
    static char pre_buf[128] = {0};
    int64_t idx = find_kv_idx(m, "tokenizer.ggml.pre");
    if (idx < 0) idx = find_kv_idx(m, "preprocessing");
    if (idx >= 0 && m->kvs[idx].str_val && m->kvs[idx].str_val[0]) {
        strncpy(pre_buf, m->kvs[idx].str_val, sizeof(pre_buf)-1);
        return pre_buf;
    }
    return "";
}

int64_t gguf_vocab_size(const gguf_model_t *m) {
    /* Try multiple vocab size keys */
    const char *keys[] = {
        "tokenizer.ggml.vocab_size", "tokenizer_vocab_size", "vocab_size"
    };
    int64_t idx = find_kv_or(m, keys, 3);
    if (idx < 0) return 0;
    return kv_as_int64(m, idx);
}

int64_t gguf_head_count(const gguf_model_t *m) {
    const char *keys[] = {
        "attention.head_count", "qwen35.attention.head_count",
        "llama.attention.heads"
    };
    int64_t idx = find_arch_kv(m, "attention.head_count", keys, 3);
    if (idx < 0) return 0;
    return kv_as_int64(m, idx);
}

int64_t gguf_head_count_kv(const gguf_model_t *m) {
    const char *keys[] = {
        "attention.head_count_kv", "qwen35.attention.head_count_kv",
        "llama.attention.heads_kv", "attention.kv_heads"
    };
    int64_t idx = find_arch_kv(m, "attention.head_count_kv", keys, 4);
    if (idx < 0) return 0;
    return kv_as_int64(m, idx);
}

int64_t gguf_context_length(const gguf_model_t *m) {
    const char *keys[] = {
        "general.context_length", "qwen35.context_length", "qwen2.context_length",
        "context_length", "llama.context_length"
    };
    int64_t idx = find_arch_kv(m, "context_length", keys, 5);
    if (idx < 0) return 0;
    return kv_as_int64(m, idx);
}

int64_t gguf_embedding_length(const gguf_model_t *m) {
    const char *keys[] = {
        "embedding_length", "qwen35.embedding_length", "qwen2.embedding_length",
        "llama.embedding_length", "model_dim"
    };
    int64_t idx = find_arch_kv(m, "embedding_length", keys, 5);
    if (idx < 0) return 0;
    return kv_as_int64(m, idx);
}

int64_t gguf_block_count(const gguf_model_t *m) {
    const char *keys[] = {
        "block_count", "qwen35.block_count", "qwen2.block_count",
        "llama.n_layer", "n_layer"
    };
    int64_t idx = find_arch_kv(m, "block_count", keys, 5);
    if (idx < 0) return 0;
    return kv_as_int64(m, idx);
}

/* --- KV cache configuration (bridge to Phase 3) --- */

void      gguf_set_n_ctx(gguf_model_t *m, int64_t n_ctx) {
    if (m && n_ctx > 0) m->n_ctx = n_ctx;
}

int64_t   gguf_get_n_ctx(const gguf_model_t *m)         {
    return m ? m->n_ctx : 512;
}

/* Get string array from KV by key. Returns 0 on success, -1 if not found or not a string array. */
int32_t gguf_kv_get_string_array(const gguf_model_t *model, const char* key, const char*** out_strings, size_t* out_count) {
    if (!model || !key || !out_strings || !out_count) return -1;
    int64_t idx = find_kv_idx(model, key);
    if (idx < 0) return -1;
    const struct kv_entry *kv = &model->kvs[idx];
    if (!kv->is_array || kv->arr_type != GGUF_TYPE_STRING) return -1;
    *out_strings = (const char**)kv->arr_strings;
    *out_count = kv->arr_str_n;
    return 0;
}

void gguf_free_string_array(const char** strings, size_t count) {
    (void)strings; (void)count; /* strings are owned by the model; no separate free needed */
}

/* Get added_tokens array from KV by key. Returns 0 on success, -1 if not found or not a string array. */
int32_t gguf_kv_get_added_tokens(const gguf_model_t *model, const char* key, const char*** out_strings, size_t* out_count) {
    if (!model || !key || !out_strings || !out_count) return -1;
    int64_t idx = find_kv_idx(model, key);
    if (idx < 0) return -1;
    const struct kv_entry *kv = &model->kvs[idx];
    if (!kv->is_array || kv->arr_type != GGUF_TYPE_STRING) return -1;
    *out_strings = (const char**)kv->arr_strings;
    *out_count = kv->arr_str_n;
    return 0;
}

size_t    gguf_alignment(const gguf_model_t *m)     { return m ? m->alignment : GGUF_ALIGN; }
size_t    gguf_data_offset(const gguf_model_t *m)   { return m ? m->data_off : 0; }
uint32_t  gguf_version(const gguf_model_t *m)       { return m ? m->version : 0; }

/* Get tensor data pointer by name */
const void* gguf_get_tensor_ptr(const gguf_model_t *m, const char* name, size_t* out_bytes) {
    if (!m || !name || !out_bytes) return nullptr;

    struct gguf_model* model = (struct gguf_model*)m;
    size_t n_tensors = model->n_tensors;
    for (size_t i = 0; i < n_tensors; ++i) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            const struct tensor_meta *tm = &model->tensors[i];

            /* Calculate total elements */
            int64_t total_elems = tm->ne[0];
            for (uint8_t d = 1; d < tm->n_dims; ++d) total_elems *= tm->ne[d];

            /* For quantized types, compute based on blocks */
            int64_t bs = get_block_size(tm->type);
            size_t ts = get_type_size(tm->type);

            if (bs > 0 && ts != (size_t)-1) {
                /* Quantized: ceil(total_elements / block_size) × bytes_per_block */
                int64_t total_blocks = total_elems / bs;
                if (total_elems % bs != 0) ++total_blocks;
                *out_bytes = (size_t)total_blocks * ts;
            } else {
                /* Unquantized: total_elements × element_size */
                *out_bytes = (size_t)total_elems * ts;
            }

            return (const char*)model->map_ptr + model->data_off + tm->data_offset;
        }
    }
    return nullptr;
}

/* --- Low-level mmap access (for inference engine) --- */

gguf_model_t* gguf_mmap_open(const char* path) {
    return load_gguf(path);
}

void gguf_mmap_close(gguf_model_t* model) {
    if (!model) return;
    if (model->map_ptr) munmap(model->map_ptr, model->map_size);
    if (model->tensors) free(model->tensors);
    if (model->kvs) {
        for (int64_t i = 0; i < model->n_kv; ++i) free(model->kvs[i].key);
        free(model->kvs);
    }
    free(model);
}

const void* gguf_mmap_get_tensor(const gguf_model_t* model, const char* name, size_t* out_bytes) {
    return gguf_get_tensor_ptr(model, name, out_bytes);
}


const void* gguf_map_base(const gguf_model_t *m, size_t* out_size) {
    if (!m) return NULL;
    if (out_size) *out_size = m->map_size;
    return m->map_ptr;
}
