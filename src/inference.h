// inference.h -- Phase 4/10/11: C API for zero-copy ROCmFP4 inference engine on Strix Halo (gfx1151).
// All functions are extern "C" with opaque handles. No C++ or HIP types exposed.
// Designed for eventual Perl XS CPAN module consumption.

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stddef.h>
#include <stdint.h>
#include "tokenizer.h"
#include "sampler.h"

#ifdef __cplusplus
extern "C" {
#endif

// Error codes (0 = success, negative = error)
typedef enum {
    INFERENCE_OK = 0,
    INFERENCE_ERR_INVALID_ARG = -1,
    INFERENCE_ERR_OUT_OF_MEMORY = -2,
    INFERENCE_ERR_MODEL_LOAD = -3,
    INFERENCE_ERR_KV_CACHE = -4,
    INFERENCE_ERR_TOKENIZER = -5,
    INFERENCE_ERR_KERNEL_LAUNCH = -6,
    INFERENCE_ERR_INVALID_STATE = -7,
    INFERENCE_ERR_NOT_IMPLEMENTED = -8,
    INFERENCE_ERR_CONTEXT_FULL = -9,
} inference_error_t;

// Convert error code to human readable string.
const char* inference_strerror(inference_error_t err);

// Opaque handle for the inference engine model
typedef struct inference_model inference_model_t;

// Model loading: mmap() the GGUF file, parse metadata, pin weight pages via hipHostRegister.
// Returns NULL on failure (check inference_get_last_error()).
inference_model_t* inference_load_model(const char* gguf_path);

// Load options. stage_dir: copy the tensors the GPU reads into one stage file in this directory (a tmpfs, best one
// mounted huge=always: /dev/shm) and map them from there; what only the CPU reads (the token embeddings when the LM
// head is separate, Qwen3.8's PLE table) stays mapped from the GGUF on disk. The stage file
// (rocm-inference-stage-<hash>.bin, the hash of the source files' identity and the staged tensors) is kept: a later
// load of the same files maps it again without copying. Not staged (the GGUF mapped directly, as without the option)
// when there is not enough room in the directory or memory: inference_stage_info says what happened. NULL options or
// stage_dir: no staging.
typedef struct inference_load_opts {
    const char* stage_dir;
} inference_load_opts;
inference_model_t* inference_load_model_ex(const char* gguf_path, const inference_load_opts* opts);
// One line about staging for this model ("staged 31234 MB in <path>", "reused <path> (31234 MB)", "not staged: ..."),
// or "" when it was not asked for. Valid while the model is.
const char* inference_stage_info(const inference_model_t* model);

// Reference counting for shared ownership
void inference_model_retain(inference_model_t* model);
void inference_model_release(inference_model_t* model);

// Free model: unpin all pages, unmap GGUF file, free metadata structures.
void inference_free_model(inference_model_t* model);

// Tokenization (CPU-side, uses tokenizer from GGUF metadata)
// Returns allocated int32 array of token IDs (length written to out_len), or NULL on failure.
// Caller must free with inference_free_tokens().
int32_t* inference_tokenize(inference_model_t* model, const char* text, int* out_len);
void inference_free_tokens(int32_t* tokens, int len);

// KV cache initialization: allocates /dev/shm buffer for all layers, pins with hipHostRegister.
// n_ctx: maximum sequence length (context window).
// Returns 0 on success, negative error code on failure.
int inference_kv_cache_init(inference_model_t* model, int n_ctx);
void inference_kv_cache_free(inference_model_t* model);

// Inference control
// Seed for inference_generate's sampler (default 0).
void inference_set_seed(inference_model_t* model, unsigned int seed);

// Opt-in: use an int8 (Q8_0, as llama-quantize writes it) copy of a float (F32 / F16 / BF16) LM head, built
// on the GPU the first time (~0.25 s; half the head's size in memory). On Qwen3.6-35B-A3B (1 GB BF16 head)
// generation is ~20% faster; the logits differ slightly from the BF16 head's, so greedy output can change where
// the top two tokens are close. Off by default; may be called before or after inference_kv_cache_init.
// Returns 0, 1 if the head is already quantized (nothing changes), negative on error.
int inference_set_lm_head_q8(inference_model_t* model, int enable);

// Opt-in: an MTP draft head for speculative decoding (Qwen3.8-Flash-Next / qwen4exp: a nextn sidecar GGUF such as
// mtp-Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST.gguf). The head proposes the next token, the model verifies it in the
// same pass as the current one, so an accepted draft gives two tokens per pass; the output is what generation
// without it gives (a draft is only kept when it is the token sampled anyway). path NULL unloads it. May be
// called before or after inference_kv_cache_init. Returns 0, negative on error (the file is not a matching
// head).
int inference_set_draft_model(inference_model_t* model, const char* path);
// Tokens per prefill micro-batch (16 .. 8192; 0: the default, 2048 for MoE models, 1024 for dense ones). Bigger
// batches read each MoE expert's weights once for more tokens: with 4096, long prompts on MoE models are up to ~10% faster (Qwen3.8-Flash-Next
// 7,281 tokens 725 -> 795 tok/s, Qwen3.5-122B-A10B 2,427 tokens 608 -> 674, Qwen3.6-35B-A3B +4-7%); dense models
// gain nothing. The scratch memory grows with it (~1 MB per token on Qwen3.8: ~1 GB at 1024, ~4 GB at 4096). The
// output does not depend on it beyond the float summation order of the WMMA kernels (exactly the same with
// QMATVEC_NO_WMMA).
// Must be called before inference_kv_cache_init (the scratch is allocated there). Returns 0, negative on error.
int inference_set_prefill_batch(inference_model_t* model, int n_tokens);

// Models with sliding-window attention layers (Gemma-4) keep those layers' K / V in a ring of their last positions
// instead of the whole context: at 128k positions 3 GB instead of 31 GB for Gemma-4 26B-A4B. positions: -1 (the
// default) a ring of the smallest power of two >= window + prefill batch; N > 0 at least N positions (more history:
// prefix reuse continues a sequence only where the ring still holds the window before the reused prefix, else
// the prompt is processed from the start); 0 no ring, every layer keeps its full history (needed for a block
// store, inference_store_open). No ring either when it would not be smaller than the context. Must be called
// before inference_kv_cache_init. Returns 0, negative on error. inference_swa_ring: the ring's positions after
// inference_kv_cache_init (0: no ring).
int inference_set_swa_ring(inference_model_t* model, int positions);
int inference_swa_ring(const inference_model_t* model);

// The KV cache's element type: INFERENCE_KV_F16 (the default) or INFERENCE_KV_Q8_0 (int8 with an fp16 scale per 32
// values, as llama.cpp's -ctk/-ctv q8_0: 1.06 bytes per value instead of 2, so about half the memory and the
// attention reads of long contexts; the cache rows must hold a multiple of 256 values). Must be called before
// inference_kv_cache_init. Returns 0, negative on error.
#define INFERENCE_KV_F16 0
#define INFERENCE_KV_Q8_0 1
int inference_set_kv_type(inference_model_t* model, int kv_type);

// Drafts per pass with a draft head: the head chains them (each from its own output), the model verifies them
// with the current token in one pass. More drafts give more tokens per pass when they are accepted, at the cost
// of a longer pass. 1..3: that many; 0 (default): 2 or 3, chosen per pass from the recent acceptance of each
// draft position and the measured pass times (3 on code and chat replies, typically 2 on prose). Returns 0,
// negative on error.
int inference_set_draft_tokens(inference_model_t* model, int n);

// The model holds one sequence: tokens are decoded at positions 0, 1, 2, ... (n_past counts them).
// inference_reset starts a new sequence (position 0, which also clears the DeltaNet recurrent state).
void inference_reset(inference_model_t* model);
int inference_n_past(const inference_model_t* model);

// Prefill: n tokens of the current sequence at positions n_past .. n_past + n - 1, in batches (much faster
// than n inference_decode_token calls, same results). Afterwards the logits are those of the last token.
// Returns 0 on success, INFERENCE_ERR_CONTEXT_FULL if they don't fit, negative on error.
int inference_prefill(inference_model_t* model, const int32_t* tokens, int n);

// Single-token decode step: runs one forward pass for token_id at position n_past, appends it to the
// KV cache and recurrent state, and returns the greedy next token. Use inference_get_logits (or
// qwen_model_logits) to sample differently.
// Returns 0 on success, INFERENCE_ERR_CONTEXT_FULL when n_past reached n_ctx, negative on error.
int inference_decode_token(inference_model_t* model, int token_id, int* next_token_out);

// Batched forward pass (prefill or decode) for a batch of sequences.
// tokens: array of token IDs of length batch_size * seq_len (row-major).
// positions: array of sequence start positions in KV cache (length batch_size).
// seq_lens: array of actual sequence lengths (length batch_size).
// logits_out: optional output buffer for logits, size batch_size * vocab_size floats. If NULL, logits are not returned.
// Returns 0 on success, negative on error.
int inference_forward_batch(inference_model_t* model,
                            const int32_t* tokens,
                            const int* positions,
                            const int* seq_lens,
                            int batch_size,
                            int seq_len,
                            float* logits_out);

// Generation: starts a new sequence, prefills prompt_ids, then samples up to max_new_tokens tokens with
// the sampler chain of sampler.h. Stops early at EOS (not written to out_ids) or when the context is full.
// out_ids: buffer of max_new_tokens; out_len: number of tokens generated.
// Returns 0 on success, negative on error.
int inference_generate_ex(inference_model_t* model, const int32_t* prompt_ids, int prompt_len, int max_new_tokens,
                          const sampler_params* params, int32_t* out_ids, int* out_len);

// ---- Streaming generation (the API for applications and the Perl module)

// Why inference_generate_stream stopped
typedef enum {
    INFERENCE_STOP_NONE = 0,
    INFERENCE_STOP_EOG = 1,           // the model produced an end-of-generation token (EOS, end of turn, ...)
    INFERENCE_STOP_MAX_TOKENS = 2,    // max_new_tokens reached
    INFERENCE_STOP_STRING = 3,        // the text reached one of the stop strings
    INFERENCE_STOP_CALLBACK = 4,      // on_token returned nonzero
    INFERENCE_STOP_CONTEXT_FULL = 5,  // the sequence filled the context
} inference_stop_reason;

// Called once per generated token with the text that is ready: UTF-8, possibly empty, because a piece that
// ends inside a multi-byte character or could still become a stop string is held back and delivered with a
// later token. A final call with token = -1 delivers any held-back text when generation ends (not the text
// from a stop string on). Return nonzero to stop after this token.
typedef int (*inference_token_cb)(void* user, int32_t token, const char* text, int text_len);

typedef struct inference_gen_options {
    sampler_params sampler;           // sampler.h chain; sampler.temperature <= 0 is greedy
    int max_new_tokens;               // default 256
    const char* const* stop_strings;  // optional; generation stops before the first occurrence (not delivered)
    int n_stop_strings;
    int ignore_eog;                   // 1: do not stop at end-of-generation tokens (default 0)
    int reuse_prefix;                 // 1: prompt_ids is the whole sequence (e.g. a re-rendered chat), not an
                                      // addition: the common prefix with the current sequence is reused
    int checkpoint_tail;              // recurrent models: save the state this many tokens (at most half the
                                      // prefilled part) before the end of the prompt, where a re-rendered chat
                                      // usually diverges (default 16; 0: none)
    inference_token_cb on_token;      // optional
    void* user;                       // passed to on_token
    int prefill_chunk;                // inference_stream_begin only: prefill the prompt in steps of at most this
                                      // many tokens (0: all at begin), so that other sequences' steps go on
                                      // between them; a step that prefills emits no token (-1, no text)
} inference_gen_options;

// Defaults: sampler_default_params(), max_new_tokens 256, no stop strings, stop at EOG, no callback, prompt
// appended (reuse_prefix 0), checkpoint_tail 16, prefill_chunk 0.
inference_gen_options inference_gen_default_options(void);

// Continues the current sequence (inference_reset starts a new one): appends the n_prompt prompt tokens
// (n_prompt may be 0 to continue from the last token), then generates. With options->reuse_prefix the prompt is
// the whole sequence instead: the longest prefix it shares with the current sequence is kept and only the rest is
// prefilled (attention-only models rewind anywhere; models with recurrent layers rewind to the latest saved
// checkpoint inside the shared prefix, else start over). See inference_last_prompt_stats. Every emitted token is decoded into the
// sequence, so a later call continues exactly after it; the end-of-generation token that stops it is neither
// emitted nor decoded. out_ids (optional, max_out entries) receives the emitted tokens; *n_out counts them
// (all of them, even beyond max_out); *reason why it stopped. Returns 0 on success (a full context during
// generation is a stop reason), INFERENCE_ERR_CONTEXT_FULL if the prompt does not fit, negative on error.
int inference_generate_stream(inference_model_t* model, const int32_t* prompt_ids, int n_prompt,
                              const inference_gen_options* options, int32_t* out_ids, int max_out, int* n_out,
                              inference_stop_reason* reason);

// ---- Sequences: several conversations on one model, each its own KV cache (as big as inference_kv_cache_init's),
// recurrent state, history and prefix-reuse checkpoints. Everything above works on the current sequence (0, the
// default, until another is selected); selecting is a swap of pointers, nothing is copied. inference_seq_new: a new
// empty sequence (after inference_kv_cache_init; not with a draft model), its number, or -1 (not enough memory);
// inference_seq_free: free one (not 0). 0 / negative on error.
int inference_seq_new(inference_model_t* model);
int inference_seq_select(inference_model_t* model, int seq);
int inference_seq_current(const inference_model_t* model);
int inference_seq_free(inference_model_t* model, int seq);

// ---- Step-wise generation: inference_generate_stream one token per call, so that a caller can interleave
// sequences. begin handles the prompt on the current sequence (as inference_generate_stream: options' reuse_prefix,
// checkpoint_tail, ...; on_token is not used) and returns a stream bound to that sequence, NULL on error. step
// selects the stream's sequence when another is current, samples a token, decodes it into the sequence and returns
// 1 while generation goes on, 0 when it has stopped (*reason; the last step may still emit a token), negative on
// error. *token: the emitted token (-1: none); text / text_len: the text ready to deliver with it (a stop string's
// start or a split UTF-8 character is held back until it is resolved), valid until the next call. No speculative
// decoding.
typedef struct inference_stream inference_stream_t;
inference_stream_t* inference_stream_begin(inference_model_t* model, const int32_t* prompt_ids, int n_prompt,
                                           const inference_gen_options* options);
int inference_stream_step(inference_stream_t* stream, int32_t* token, const char** text, int* text_len,
                          inference_stop_reason* reason);
int inference_stream_tokens(const inference_stream_t* stream);  // tokens emitted so far
int inference_stream_seq(const inference_stream_t* stream);     // its sequence
void inference_stream_free(inference_stream_t* stream);
// One step of each of n streams (of different sequences of one model) with one forward pass for all (batched
// decoding: the matmuls read each weight once for all sequences), as inference_stream_step on each. results[i] is
// streams[i]'s step: token, text, reason, and rc (as inference_stream_step returns: 1 going, 0 stopped, < 0 its
// error). Returns 0, or < 0 when the batch failed. Without batched decoding (a draft head)
// the streams are decoded one by one, with the same results.
typedef struct {
    int32_t token;  // the token emitted (-1: none)
    const char* text;  // the text ready with it (valid until the stream's next step)
    int text_len;
    inference_stop_reason reason;
    int rc;
} inference_step_result;
int inference_stream_step_batch(inference_stream_t* const* streams, int n, inference_step_result* results);

// Embedding models (EmbeddingGemma, GGUF architecture gemma-embedding). inference_embedding_dim: the embedding's
// length (0: not an embedding model). inference_embed: the embedding of one input, n tokens as inference_tokenize
// gives them (with BOS / EOS), after inference_kv_cache_init (n <= its length, at most 2048): the model's pooling
// (mean for EmbeddingGemma) of its last hidden states into out[inference_embedding_dim]; normalize 1: scaled to
// length 1 (as llama-server's /v1/embeddings). Leaves the model's sequences alone. 0, or < 0
// (INFERENCE_ERR_CONTEXT_FULL: the input is too long).
int inference_embedding_dim(const inference_model_t* model);
int inference_embed(inference_model_t* model, const int32_t* tokens, int n, float* out, int normalize);

// How the last inference_generate_stream prompt was handled: tokens reused from the sequence, tokens prefilled.
int inference_last_prompt_stats(const inference_model_t* model, int* reused, int* prefilled);

// ---- KV block store (Phase 22b): prompts shared across sessions, processes and restarts.
// With a store open, every full KV block (inference_store_get_stats: block_tokens positions, all layers) is
// published as a file <chain hash>.kvb in dir, and the recurrent state (hybrid models) is saved as <chain hash>.rst
// at chosen block boundaries: every snapshot_every-th one (0: none) and the last one of each
// inference_generate_stream prompt. The chain hash covers the model, the layout and every token up to the end of
// the block, so equal prefixes give equal names. A reuse_prefix prompt then starts from the deepest block chain in
// the store that ends in a snapshot, when that reaches further than the current sequence: the blocks are mapped
// in place (no copy), the snapshot is read back. Store blocks are never written; the files can be deleted at
// any time (LRU eviction is the caller's: files are touched when used) and moved to and from other storage.
// dir must support O_TMPFILE (tmpfs such as /dev/shm, ext4, xfs, btrfs); the KV cache's own block files move
// there. Returns 0, negative on error.
int inference_store_open(inference_model_t* model, const char* dir, int snapshot_every);
void inference_store_close(inference_model_t* model);

typedef struct inference_store_stats {
    long long blocks_published;    // blocks this model wrote to the store
    long long blocks_attached;     // store blocks mapped into the sequence
    long long snapshots_saved, snapshots_restored;
    long long tokens_from_store;   // prompt tokens taken from the store instead of prefilled
    int block_tokens;              // positions per block
    long long block_bytes;         // size of a .kvb file
    long long snapshot_bytes;      // size of a .rst file (0: no recurrent layers)
} inference_store_stats;

int inference_store_get_stats(const inference_model_t* model, inference_store_stats* stats);

// The store file names (<hash>.kvb, one per line, in order) of the full blocks of a token list: e.g. to keep a
// system prompt's blocks from eviction. A block's .rst snapshot, if any, has the same name with .rst. Writes at
// most max_len - 1 bytes plus a NUL; returns the full length, negative on error.
int inference_store_block_names(const inference_model_t* model, const int32_t* tokens, int n, char* out, int max_len);

// ---- Chat. Templates are rendered outside this library (e.g. with a Jinja engine, as the Perl module does with
// MiniJinja): render the conversation to text, tokenize it (special tokens in the text, such as <|im_start|>,
// become their ids), then generate with reuse_prefix so each turn only prefills what changed.

// The GGUF's chat template (tokenizer.chat_template), or a named one (tokenizer.chat_template.<name>) when name
// is not NULL; NULL if absent. Valid while the model lives.
const char* inference_chat_template(inference_model_t* model, const char* name);
// BOS / EOS token ids (-1 if none); their text (the bos_token / eos_token template variables) is
// inference_detokenize of the id.
int inference_bos_id(const inference_model_t* model);
int inference_eos_id(const inference_model_t* model);

// 1 if the token ends generation (llama.cpp's rule: the GGUF's EOS / EOT / EOM ids and end-of-turn markers).
int inference_token_is_eog(const inference_model_t* model, int32_t token);

// Text of n tokens (UTF-8; control tokens render as their text). Writes at most max_len - 1 bytes plus a NUL;
// returns the full length (call again with a bigger buffer if it is >= max_len), negative on error.
int inference_detokenize(const inference_model_t* model, const int32_t* ids, int n, char* out, int max_len);

// Same with temperature / top_k / top_p only (min_p and repeat penalty off) and the inference_set_seed seed.
// temperature <= 0 is greedy.
int inference_generate(inference_model_t* model,
                       const int32_t* prompt_ids,
                       int prompt_len,
                       int max_new_tokens,
                       float temperature,
                       int top_k,
                       float top_p,
                       int32_t* out_ids,
                       int* out_len);

// Forward declaration of Qwen model handle (defined in qwen_model.h)
struct QwenModel;
typedef struct QwenModel qwen_model_t;

// Get the internal Qwen model handle (created during kv_cache_init)
// Use this instead of creating a new qwen model — ensures proper initialization
qwen_model_t* inference_get_qwen_model(inference_model_t* model);

// Copy of the logits of the last decoded token (no forward pass), for custom sampling.
// vocab_ids: pre-allocated int32[vocab_size], filled with 0..vocab_size-1
// logits: pre-allocated float[vocab_size]
// Returns 0 on success, negative on error (e.g. nothing decoded yet).
int inference_get_logits(inference_model_t* model, int* vocab_ids, float* logits);

// Model introspection
int inference_n_ctx(const inference_model_t* model);
int inference_batch_size(const inference_model_t* model);  // always 1 for now
const char* inference_model_name(const inference_model_t* model);
int inference_vocab_size(const inference_model_t* model);

// Direct tensor access for debugging/analysis
// Returns pointer to tensor data in mapped memory (valid while model is alive)
// out_bytes: receives tensor size in bytes
const void* inference_get_tensor_ptr(const inference_model_t* model, const char* tensor_name, size_t* out_bytes);

// Tokenizer access for decoding tokens back to text
// Returns opaque tokenizer handle (valid while model is alive)
// Can be passed to tokenizer_decode() for text output
tokenizer_t* inference_get_tokenizer(const inference_model_t* model);

// Error handling: returns last error message (thread-local static buffer), or NULL if none.
const char* inference_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H