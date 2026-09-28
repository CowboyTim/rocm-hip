#ifndef GGUF_MMAP_H
#define GGUF_MMAP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque model handle */
typedef struct inference_model inference_model_t;

/* Load model from GGUF file path via mmap(). Returns NULL on error. */
inference_model_t *inference_load_model(const char *path);

/* Free all resources associated with the model. */
void               inference_free_model(inference_model_t *model);

/* --- Tensor introspection (Phase 1 deliverable) --- */

int64_t      inference_n_tensors(const inference_model_t *model);
const char  *inference_tensor_name(const inference_model_t *model, int64_t idx);
int          inference_tensor_type(const inference_model_t *model, int64_t idx);
int64_t     *inference_tensor_ne(const inference_model_t *model, int64_t idx); /* array of up to 4 dimensions */
uint8_t      inference_tensor_dims(const inference_model_t *model, int64_t idx); /* n_dims (1-4) */
size_t       inference_tensor_block_size(const inference_model_t *model, int64_t idx);
size_t       inference_tensor_type_size(const inference_model_t *model, int64_t idx); /* bytes per block */
size_t       inference_tensor_row_size(const inference_model_t *model, int64_t idx);   /* bytes for ne[0] elements */
uint64_t     inference_tensor_offset(const inference_model_t *model, int64_t idx);     /* data blob offset */
size_t       inference_tensor_nbytes(const inference_model_t *model, int64_t idx);     /* unpadded byte count */
size_t       inference_tensor_nbytes_pad(const inference_model_t *model, int64_t idx); /* padded byte count */

/* --- KV metadata --- */

int64_t      inference_n_kv(const inference_model_t *model);
const char  *inference_kv_key(const inference_model_t *model, int64_t idx);
int          inference_kv_type(const inference_model_t *model, int64_t idx);

/* Read scalar KV values by index. Returns 0 on success, -1 on error/type mismatch. */
int32_t      inference_kv_get_u8(const inference_model_t *model, int64_t idx, uint8_t *out);
int32_t      inference_kv_get_i8(const inference_model_t *model, int64_t idx, int8_t *out);
int32_t      inference_kv_get_u16(const inference_model_t *model, int64_t idx, uint16_t *out);
int32_t      inference_kv_get_i16(const inference_model_t *model, int64_t idx, int16_t *out);
int32_t      inference_kv_get_u32(const inference_model_t *model, int64_t idx, uint32_t *out);
int32_t      inference_kv_get_i32(const inference_model_t *model, int64_t idx, int32_t *out);
int32_t      inference_kv_get_f32(const inference_model_t *model, int64_t idx, float *out);
int32_t      inference_kv_get_u64(const inference_model_t *model, int64_t idx, uint64_t *out);
int32_t      inference_kv_get_i64(const inference_model_t *model, int64_t idx, int64_t *out);
int32_t      inference_kv_get_f64(const inference_model_t *model, int64_t idx, double *out);

/* --- Model-level queries --- */

size_t    inference_alignment(const inference_model_t *model);
size_t    inference_data_offset(const inference_model_t *model);
uint32_t  inference_version(const inference_model_t *model);

/* Get model metadata keys useful for KV config decisions */
const char  *inference_arch(const inference_model_t *model);         /* "qwen2", "llama", etc. (string) */
const char  *inference_vocab_name(const inference_model_t *model);   /* tokenizer type name (string) */
const char  *inference_tokenizer_type(const inference_model_t *model); /* "gpt2", "bpe", etc. (string) */
const char  *inference_pre_type(const inference_model_t *model);     /* tokenizer pre-processing type (string) */
int64_t      inference_vocab_size(const inference_model_t *model);   /* vocab size from GGUF */
int64_t      inference_head_count(const inference_model_t *model);   /* attention heads */
int64_t      inference_head_count_kv(const inference_model_t *model); /* RoPE dimensions hint */
int64_t      inference_context_length(const inference_model_t *model); /* stored context_len if present */
int64_t      inference_embedding_length(const inference_model_t *model); /* embedding/d_model dimension */
int64_t      inference_block_count(const inference_model_t *model);  /* number of transformer layers */

/* Read scalar KV value as string. Returns NULL if not a string or key not found. */
const char  *inference_kv_str_val(const inference_model_t *model, int64_t idx);

/* --- KV cache configuration (Phase 1/3 bridge) --- */

/* Set context length for KV cache allocation at runtime. Default: 512. */
void         inference_set_n_ctx(inference_model_t *model, int64_t n_ctx);
int64_t      inference_get_n_ctx(const inference_model_t *model);

#ifdef __cplusplus
}
#endif

#endif /* GGUF_MMAP_H */
