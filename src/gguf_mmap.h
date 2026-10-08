#ifndef GGUF_MMAP_H
#define GGUF_MMAP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque model handle for GGUF mmap layer */
typedef struct gguf_model gguf_model_t;

/* --- Low-level mmap access (for inference engine) --- */
gguf_model_t*      gguf_mmap_open(const char* path);
void               gguf_mmap_close(gguf_model_t* model);
const void*        gguf_mmap_get_tensor(const gguf_model_t* model, const char* name, size_t* out_bytes);

/* --- Tensor introspection --- */
int64_t      gguf_n_tensors(const gguf_model_t *model);
const char  *gguf_tensor_name(const gguf_model_t *model, int64_t idx);
int          gguf_tensor_type(const gguf_model_t *model, int64_t idx);
int64_t     *gguf_tensor_ne(const gguf_model_t *model, int64_t idx); /* array of up to 4 dimensions */
uint8_t      gguf_tensor_dims(const gguf_model_t *model, int64_t idx); /* n_dims (1-4) */
size_t       gguf_tensor_block_size(const gguf_model_t *model, int64_t idx);
size_t       gguf_tensor_type_size(const gguf_model_t *model, int64_t idx); /* bytes per block */
size_t       gguf_tensor_row_size(const gguf_model_t *model, int64_t idx);   /* bytes for ne[0] elements */
uint64_t     gguf_tensor_offset(const gguf_model_t *model, int64_t idx);     /* data blob offset */
size_t       gguf_tensor_nbytes(const gguf_model_t *model, int64_t idx);     /* unpadded byte count */
size_t       gguf_tensor_nbytes_pad(const gguf_model_t *model, int64_t idx); /* padded byte count */

/* --- KV metadata --- */
int64_t      gguf_n_kv(const gguf_model_t *model);
const char  *gguf_kv_key(const gguf_model_t *model, int64_t idx);
int          gguf_kv_type(const gguf_model_t *model, int64_t idx);

/* Read scalar KV values by index. Returns 0 on success, -1 on error/type mismatch. */
int32_t      gguf_kv_get_u8(const gguf_model_t *model, int64_t idx, uint8_t *out);
int32_t      gguf_kv_get_i8(const gguf_model_t *model, int64_t idx, int8_t *out);
int32_t      gguf_kv_get_u16(const gguf_model_t *model, int64_t idx, uint16_t *out);
int32_t      gguf_kv_get_i16(const gguf_model_t *model, int64_t idx, int16_t *out);
int32_t      gguf_kv_get_u32(const gguf_model_t *model, int64_t idx, uint32_t *out);
int32_t      gguf_kv_get_i32(const gguf_model_t *model, int64_t idx, int32_t *out);
int32_t      gguf_kv_get_f32(const gguf_model_t *model, int64_t idx, float *out);
int32_t      gguf_kv_get_u64(const gguf_model_t *model, int64_t idx, uint64_t *out);
int32_t      gguf_kv_get_i64(const gguf_model_t *model, int64_t idx, int64_t *out);
int32_t      gguf_kv_get_f64(const gguf_model_t *model, int64_t idx, double *out);

/* --- Model-level queries --- */
size_t    gguf_alignment(const gguf_model_t *model);
size_t    gguf_data_offset(const gguf_model_t *model);
uint32_t  gguf_version(const gguf_model_t *model);

/* Get model metadata keys useful for KV config decisions */
const char  *gguf_arch(const gguf_model_t *model);         /* "qwen2", "llama", etc. (string) */
const char  *gguf_vocab_name(const gguf_model_t *model);   /* tokenizer type name (string) */
const char  *gguf_tokenizer_type(const gguf_model_t *model); /* "gpt2", "bpe", etc. (string) */
const char  *gguf_pre_type(const gguf_model_t *model);     /* tokenizer pre-processing type (string) */
int64_t      gguf_vocab_size(const gguf_model_t *model);   /* vocab size from GGUF */
int64_t      gguf_head_count(const gguf_model_t *model);   /* attention heads */
int64_t      gguf_head_count_kv(const gguf_model_t *model); /* RoPE dimensions hint */
int64_t      gguf_context_length(const gguf_model_t *model); /* stored context_len if present */
int64_t      gguf_embedding_length(const gguf_model_t *model); /* embedding/d_model dimension */
int64_t      gguf_block_count(const gguf_model_t *model);  /* number of transformer layers */

/* Read scalar KV value as string. Returns NULL if not a string or key not found. */
const char  *gguf_kv_str_val(const gguf_model_t *model, int64_t idx);

/* Get string array from KV by key. Returns 0 on success, -1 if not found or not an array of strings.
 * The caller must free the returned array and its strings with gguf_free_string_array(). */
int32_t gguf_kv_get_string_array(const gguf_model_t *model, const char* key, const char*** out_strings, size_t* out_count);
void gguf_free_string_array(const char** strings, size_t count);

/* Get added_tokens array (array of strings) from KV by key. Returns 0 on success, -1 if not found or not a string array. */
int32_t gguf_kv_get_added_tokens(const gguf_model_t *model, const char* key, const char*** out_strings, size_t* out_count);

/* Get tensor data pointer by name. Returns NULL if not found.
 * out_bytes receives the tensor size in bytes. */
const void* gguf_get_tensor_ptr(const gguf_model_t *model, const char* name, size_t* out_bytes);

/* --- Quantization type helpers --- */

/* Check if a tensor type stores scales internally (self-contained).
 * Returns 1 for Q4_0, Q4_1, Q8_0, TYPE100 etc. where scales are
 * embedded within each block structure. Returns 0 for external-scales types. */
int          gguf_type_is_self_contained(int type);

#ifdef __cplusplus
}
#endif

#endif /* GGUF_MMAP_H */