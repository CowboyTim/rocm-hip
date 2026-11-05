# Project Plan: rocm-hip (Strix Halo, gfx1151)

> What is open: next steps, planned work, leads. How the engine is built and what it runs: [ARCHITECTURE.md](ARCHITECTURE.md).
> Finished work, with its measurements, is in the git history (this file was pruned of it on 2026-11-05).

## ▶ Next (week of 2026-11-09; user, 2026-11-05: "we will work on it next week")

1. **Qwen3.8-Flash-Next from BF16** (section below): the model's own MTP layer as its draft head; a batch's PLE rows
   read from disk in parallel.
2. **Concurrent runs** in rocm-inference-server (section below).
3. **Server logging** through an engine event hook, formatted in Perl (section below, planned, not started).

Later: a conversion / quantization tool in this project; expert pin / unpin for MoE models larger than memory.

## Status in brief

A standalone engine (no llama.cpp linked; `llama.cpp/` = the ROCmFPX fork and `llama.cpp-stock/` = upstream are
reference oracles only): GGUF files mapped and pinned in place (zero-copy on the UMA), HIP kernels per quant type
(int8 / WMMA), a C API (`libinference.so`), the Perl XS module ROCm::Inference with chat templates (Minijinja) and
reply parsing, and rocm-inference-server (OpenAI-compatible, SSE, tools, embeddings, `--models-dir` with models
loaded per request, each by default at its trained context within `--kv-budget`). Architectures: qwen2, qwen3,
qwen3moe, qwen35 / qwen35moe, qwen4exp (Qwen3.8-Flash-Next), gemma4, gemma-embedding, nemotron_h_moe, llama. KV
caches F16 / Q8_0 / Q4_0 (Hadamard-rotated when quantized), a KV block store, batched decoding, MTP speculative
decoding. Regression suite: `make regress` / `make regress-full` (tests/regress.tsv: perplexity per model / quant
type / KV type / context, pass keys to 128k, forward_verify; perplexity agrees with llama.cpp within ~0.5%, the
K-quants ~1%). Deployment: the user's llama.cpp-rocm-gfx1151-docker (branch rocm-inference) builds rocm-hip from
GitHub `main` and serves the engine at /rocm/v1 next to llama-server (no CPAN release planned yet).

## Qwen3.8-Flash-Next from BF16

/models/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST-PLE_Q8_0.gguf (122.8 GB): upstream's convert_hf_to_gguf.py, then the
ROCmFPX llama-quantize (local worktree /data/rocmfpx-strix, patched to read `recurrent_layers` as upstream's bool
flags; `--tensor-type per_layer_token_embd=q8_0`, Q4_0_ROCMFP4_FAST). ~68 GB pinned, the 54 GB PLE table read from
disk. ppl 2.1927 (wikitext, 10 chunks); 2k prompt 850 tok/s prefill, 39.3 tok/s generation with the PLE rows cached.
- **The model's own MTP layer as its draft head.** The base-model GGUF carries it (blk.48: nextn.eh_proj / enorm /
  hnorm, nextn.hc_head_down / norm / up — the head's own hyper-connection mix, new — and attention with QSA
  compress ratio 4); the release came with a separate head file (/data/mtp, set_draft_model: ~38 -> ~50 tok/s).
  Load the nextn block of the same file as the draft, dense attention there (exact below top_k 2048 x ratio 4
  positions; above, only which drafts are accepted can differ). The server then needs no separate draft file.
- **PLE rows in parallel.** A text whose rows are not cached prefills at 391 tok/s: ~32k random 4 KB reads of the
  table one at a time (~85 us each). Fetch a batch's rows concurrently (madvise WILLNEED / readahead, or a few
  threads: NVMe serves deep queues) so the table can stay on disk at prefill speed.
- Not tried yet: the server pairing a separate draft-head file from `--models-dir` with the real Flash.
- The fork's qwen4exp is behind upstream's (recurrent_layers as indices vs upstream's bool flags; llama-server
  rejects these files): worth porting in the fork (the user's).
- No XNACK on gfx1151 (rocminfo "XNACK enabled: NO", amdgpu noretry=1; the driver can switch it, unstable by the
  user's experience): the GPU cannot fault pages in, so whatever it reads must be pinned; only CPU-read tensors
  (the PLE table) can be left to the page cache.

## Concurrent runs (to investigate; user, 2026-11-05)

The server serves one request at a time unless `--parallel N`, which batches N conversations of one model (not
with a draft head); with `--models-dir` each loaded model has its own sequences and the loop steps them in turn. A
request that arrives during another, or during a model load (2 s from the page cache, minutes from a cold disk),
waits: a test request waited 7 min 22 s, then ran at 38 tok/s. To look at: `--parallel` by default, and with
speculative decoding (a draft per sequence, or verify batches); several models generating at once (one forward pass
each per round; GPU sharing); long prefills in chunks between other requests' tokens (prefill_chunk exists); loads
off the request path (between rounds, or in a second process); fairness and queue limits per user (with the
logging's user and conversation id); llama-server on the same GPU.

## Server logging: events from the engine, formatted in Perl (planned, not started)

User: "proper logging: prefill stage, % of, prefill done, tg generation, streamed batch etc.. make a plan so that I
can follow what it does (PID, user, timestamp, coloring) ... via a hook to perl, so we can perlify the logging *in
perl*, not in C". Today the library prints nothing (INFERENCE_VERBOSE: unstructured debug text), the only hook into
Perl is on_token, a prompt's prefill is one blocking C call, and the server logs Web::Route's line per request.
1. **C: an event hook, no formatting.** `inference_set_event_cb(model, cb, user)`: a small struct (type, sequence,
   counts, ms) at prefill_begin (total, reused from the sequence, from the KV store, to prefill), prefill_progress
   (after each batch / chunk), prefill_end, checkpoint (DeltaNet) and store (blocks resumed / published), draft
   (proposed / accepted per pass), context_full. Nothing printed in C; no cost without a hook. Generation speed:
   Perl times on_token.
2. **XS:** `set_event_handler($m, sub { my ($event, $info) = @_ })`; a die in the handler becomes a warning.
3. **Perl: ROCm::Inference::Log.** A line: timestamp (ms), PID, level, user, client IP, conversation id, request
   id, model, event, message. Colours per kind (request bold, prefill cyan, generation green, model load / unload
   magenta, warn yellow, error red); auto / always (docker logs) / never, NO_COLOR honoured. Progress at most every
   10% or 2 s. Text or JSON lines. Levels error / warn / info / debug (debug: --parallel step lines).
4. **Server:** per request: received (client, user, model, messages, tools, stream), queue wait, model load /
   unload (time, memory), prefill progress and result (reused / store / new, tok/s), time to first token,
   generation every few seconds, done (finish reason, tokens, times, tok/s), client gone, errors; Web::Route's
   request line through the same logger.
5. **Who:** the user (nginx X-Remote-User from $remote_user; later the subject of a JWT: the user may add Keycloak
   login with a JWT as API key), the client IP (X-Real-IP) and opencode's conversation id (conv_id / slot header
   from the fixups). Docker repo: proxy_set_header X-Remote-User in /rocm/.
6. Options `--log-level`, `--log-format text|json`, `--log-color auto|always|never`.
7. Tests: the C events' order (progress increasing to the total); the formatter without a GPU; log lines in
   t/60-server.t.

Example (text, one opencode turn):

    2026-11-05 14:03:12.481 [41231] INFO  alice 10.0.0.7 conv-17 chatcmpl-6f3a Qwen3.6-35B  > POST /v1/chat/completions  stream, 14 msgs, 9 tools
    2026-11-05 14:03:12.483 [41231] INFO  ... prefill 31,204 tok: 28,672 reused, 2,532 new
    2026-11-05 14:03:12.890 [41231] INFO  ... prefill  41%  1,040/2,532  2,555 tok/s
    2026-11-05 14:03:13.471 [41231] INFO  ... prefill done 2,532 tok in 0.99 s (2,558 tok/s)
    2026-11-05 14:03:13.520 [41231] INFO  ... first token after 1.04 s
    2026-11-05 14:03:15.522 [41231] INFO  ... generating  122 tok  61.0 tok/s
    2026-11-05 14:03:17.640 [41231] INFO  ... done tool_calls  251 tok in 4.12 s (60.9 tok/s), total 5.16 s

## Later

- **A conversion / quantization tool in this project** (user: "the conversion belongs with this project sort of").
  What the engine runs is specialized (ROCmFP4 FAST, the PLE table on disk, per-tensor recipes) and depends on the
  fork's converter and quantizer, whose conventions can disagree with upstream's. Scope: HF safetensors -> GGUF
  for the architectures here, metadata both the engine and llama.cpp read; quantization to the engine's types
  with per-tensor recipes (PLE / token embeddings / output, attention K/V), streamed tensor by tensor (no
  full-model RAM: the BF16 Flash is 359 GB); an importance matrix collected on the GPU by the engine from an FP4
  model; checks with dequant_verify / forward_verify against ggml's quantizers. C/C++ next to quant_types.cpp, a
  Perl front end.
- **Expert pin / unpin for MoE models larger than memory** (user: "for MoE models with a limited set of activated
  weights, dense will always be too small"). Each GGUF file is one mapping, pinned (hipHostRegister) in ranges
  around the CPU-only tensors; registration works on any range, so: after the router, make the chosen experts'
  ranges resident and registered, with a cache of recently used experts; dense layers stay pinned. Only for MoE
  models that do not fit even at FP4.

## Open leads

Speed:
- Long-context prefill attention runs at ~25% of the FP16 WMMA peak (Qwen3.8-Flash 22% of a 32k prefill, more
  beyond; already ahead of llama.cpp's: 5.5 vs 7.7 s on Qwen3.6-35B at 32k): the largest lever for long prompts.
- Qwen3.8-27B (dense ROCmFP4_FAST) is compute-bound: matmul_wmma_q4_0_rocmfp4_fast ~26 of ~59 TOPS (70% of a 32k
  prefill); its epilogue (exact per-block scaling, ~19%) would need FP16 WMMA on pre-scaled weights (twice the
  LDS). Tabled.
- MTP verify passes: read experts shared by the verified tokens once (~3.8 of 10 shared by two tokens: ~19% of a
  2-token pass's expert bytes); draft cost 2.7 ms, mostly the head's LM head (a vocabulary cut hurts other
  scripts; a shortlist from the main model's top logits of the previous row, or an int8 copy).
- Decode leftovers (Flash): moe_router 48 x 14 us, hc_norm 98 x 7.6 us, deltanet_rule 36 x 19 us, hc_mix 97 x
  2.9 us, ~930 launches x ~1.7 us of gaps; each well under 5%.
- Prefill: deltanet_conv (~100 GB/s); the quantizing producers (swiglu, add_rmsnorm_rows) compute both Q8_0 and
  Q8_K though each consumer reads one; the CPU embedding row decode; attn_k / attn_v as two launches (~1%).
- Small F16 matrices (Qwen3.5-0.8B, 1,024-2,048 rows) latency-bound at 55-76 GB/s; MoE experts in F16 / BF16 run
  matmul_f_id (no model here has them); BF16 FFN down (K 17408) the slowest shape (23 TFLOPS).

Quality / correctness:
- K-quants dot Q8_K activations (one scale per 256), as ggml's CPU backend; ggml's GPU backend uses q8_1 (per 32):
  ~1% better perplexity at Q2_K. An option for the K-quant kernels.
- A quantized KV cache needs n_kv * D % 256 == 0 (Qwen2.5-0.5B, 2 x 64, cannot have one).
- Qwen3.6-35B Q4_0 forward_verify drifts to cos 0.989 at layer 21 (smooth, next token = llama.cpp's): its suite
  row uses min cos 0.985.
- One forward_verify run on the 35B once hit a GPU memory fault in matvec_q8_id_q5_K, never reproduced; likely
  the test's buffer overrun fixed on 2026-11-04 (5e6e9e6). Close after a few clean runs.

Tests and API:
- Fold the remaining verifiers into the regression suite: generate_verify, store_verify.sh, mtp_verify (needs
  the draft head), the Perl t/ suite (Minijinja and Web::Route on PERL5LIB).
- Untested: Mistral (llama architecture, SentencePiece tokenizer), Llama-3.1-8B (no files); Llama 3.1's built-in
  tools (<|python_tag|> brave_search.call(...)) not parsed; Nemotron's reply format (check against its template).
- The XS module lacks inference_prefill_logits, model_name and set_seed (add when a Perl user needs them); a C
  API channel flag for reply kinds only if a non-Perl user needs it.

## Tabled

- Older architectures: gpt-oss, deepseek2, glm4, phi3, gemma2 / gemma3, Mistral (user, 2026-11-04).
- The ROCmFP4_FAST WMMA kernel work (user, 2026-11-04: "move on, table this for now").
- A CPAN release of ROCm::Inference, Minijinja and Web::Route (user, 2026-11-05: the Docker image builds them
  from source).

## Tried and rejected (measured; do not repeat without a new idea)

- HIP graphs for decode launch gaps: ~0.3% (1.96 vs 1.88 us per dependent kernel; per-token parameter updates).
- Small-kernel merges (moe_combine into add_rmsnorm: slower, every workgroup recomputes the expert sum; others
  assessed not worth it, 2026-10-21).
- WMMA operand pipelining in the FAST kernel: 10% slower (173 VGPRs); bigger dense WMMA tiles, 2 tiles per wave,
  32 weight rows per wave: no gain.
- Decode attention on a Q8_0 cache: 16-byte K pieces, 4 V dims per thread, 128-position chunks: no gain.
- Flash prefill prefetch of the next K/V block at D 256: spills (13 VGPRs), -5%; kept for D <= 128 only.
- MTP vocabulary shortlist: +2-5% on English, -18% on Chinese (tokens spread over the vocabulary).
- moe_router: per-wave argmax top-k instead of the bitonic sort (15.3 vs 14.3 us); 8 waves per router row.
- Unrolling the FAST matvec quad loop by 2; fewer lanes per row for the Q8_0 matvec (32 is best).
- IQ WMMA staging prefetch; a K rotation per workgroup in the expert kernels.
- Loads inside branches or `c ? load : 0` in kernels: they serialize (keep loads unconditional, select after).
