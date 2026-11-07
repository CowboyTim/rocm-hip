# Project Plan: rocm-hip (Strix Halo, gfx1151)

> What is open: next steps, planned work, leads. How the engine is built and what it runs: [ARCHITECTURE.md](ARCHITECTURE.md).
> Finished work, with its measurements, is in the git history (this file was pruned of it on 2026-11-05 and 2026-11-06).

## ▶ Next (week of 2026-11-09; user, 2026-11-05: "we will work on it next week")

1. **Qwen3.8-Flash-Next from BF16** (section below): the model's own MTP layer as its draft head; a batch's PLE rows
   read from disk in parallel.
2. **Concurrent runs** in rocm-inference-server (section below).
3. **Server logging** (section below): implemented; the server test and the Flash soak run are open.
4. **GLM-5.3-Flash (`glm5_next`) as an architecture** (section below): the gate is memory capacity, not kernels -
   95% of its 320B params are expert weights, and none of FP8 / FP4 / Q4_0 fits 128 GB, so nothing runs until the
   expert pin / unpin decision below is made. Start with the tiny GGUF generator, not the kernels.

Later: a conversion / quantization tool in this project; expert pin / unpin for MoE models larger than memory.

## Status in brief

The engine runs qwen2, qwen3, qwen3moe, qwen35 / qwen35moe, qwen4exp (Qwen3.8-Flash-Next), gemma4, gemma-embedding,
nemotron_h_moe and llama from zero-copy GGUF mappings on the UMA. Surfaces: the C API (`libinference.so`), the Perl
XS module ROCm::Inference (Minijinja templates), rocm-inference-server (OpenAI-compatible, `--models-dir`). Regression
suite: `make regress` / `make regress-full`. Deployed by the user's llama.cpp-rocm-gfx1151-docker (branch
rocm-inference), which builds rocm-hip from GitHub `main`. Design: ARCHITECTURE.md; measurements: git history.

## Qwen3.8-Flash-Next from BF16

Quantized: /models/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST-PLE_Q8_0.gguf (PLE table Q8_0 on disk). Open:
- **The model's own MTP layer as its draft head.** The base-model GGUF carries it (blk.48: nextn.eh_proj / enorm /
  hnorm, nextn.hc_head_down / norm / up — the head's own hyper-connection mix, new — and attention with QSA
  compress ratio 4); the release came with a separate head file (/data/mtp, set_draft_model: ~38 -> ~50 tok/s).
  Load the nextn block of the same file as the draft, dense attention there (exact below top_k 2048 x ratio 4
  positions; above, only which drafts are accepted can differ). The server then needs no separate draft file.
- **PLE rows in parallel.** A text whose rows are not cached prefills at 391 tok/s: ~32k random 4 KB reads of the
  table one at a time (~85 us each). Fetch a batch's rows concurrently (madvise WILLNEED / readahead, or a few
  threads: NVMe serves deep queues) so the table can stay on disk at prefill speed.
- The fork's qwen4exp is behind upstream's (recurrent_layers as indices vs upstream's bool flags; llama-server
  rejects these files): worth porting in the fork (the user's).

## GLM-5.3-Flash (`glm5_next`): what has to change, and in what order (planned, not started; user, 2026-11-06)

Config read from huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json on 2026-11-06; the numbers below are
computed from it, not copied from its model card. 320B total / 18B active, 45 decoder layers, hidden 4096, vocab
154,880, 1M context. 34 linear-attention layers (KDA) + 11 sparse-attention layers (MLA with a top-2048 index over
a k=4-pooled compressed index) at layers 3, 7, 11 ... 43; 42 MoE layers (the first 3 layers are dense FFN), 288
routed experts + 1 shared, top-8, sigmoid routing with `mlp.gate.e_score_correction_bias` and
`routed_scaling_factor` 2.5; `mhc` with `hc_mult` 4 and `hc_sinkhorn_iters` 20; `swiglu_limit` 10.0;
`num_nextn_predict_layers` 1. Two readings are inference, not config: `kda_layers` as Kimi Delta Attention
(gated-deltanet family) and `index_kpool_compress` as key pooling by 4 with APE to restore the lost position
information. Both set buffer sizes (`conv_dim = (2*n_k + n_v) * 128`), so check them against the checkpoint's
tensor shapes before sizing anything.

1. **Gate: capacity, not kernels.** 42 layers x 289 experts x 3 x 4096 x 2048 = **305.5B of the 320B params
   (95%) are expert weights**; the non-expert part (attention, embed, LM head, norms) is 14.5B. As released (FP8
   e4m3) that is 320 GB; at ROCmFP4_FAST 4.25 bpw it is 170 GB; Q4_0 is 180 GB. None fit 128 GB, and ~2.6 bpw is
   the first quantization that does. Pinning decides the speed: the non-experts cost 7.7 GB pinned at FP4, leaving
   120 GB -> 8,996 of the 12,096 experts resident (**74%**), and the other 26% of expert reads are NVMe traffic at
   3-5 GB/s, ~50x slower than the 220-256 GB/s UMA. Per token, 8 experts x 42 layers = 8.46B params are touched:
   all of it off NVMe at 3-5 GB/s is 0.7-1.1 tok/s, 26% cold is 2.6-4.3 tok/s (the NVMe reads only, no compute), and the all-resident
   bandwidth ceiling is 26-30 tok/s at FP8 (8.46 GB/token) and 49-57 at FP4 (4.5 GB/token) at 220-256 GB/s. The earlier
   12-14 / 23-27 were estimates with no stated efficiency; measure them before quoting. So "can we run it" is a routing-trace question, not a kernel question:
   route ~10k tokens through the router alone (small enough to run on CPU from a quantized copy), count the
   distinct experts hit per layer, and settle the target bpw + the pinned fraction **before** writing a kernel.
   This is the "expert pin / unpin" item under **Later** below, and for this model it is a prerequisite, not a
   later nicety.
2. **The converter and the loader, no kernels.** `llama.cpp/conversion/` has base, command_r, deepseek,
   diffusion_gemma, laguna, llama, mellum, muse_glimmer, qwen4exp and zaya - **no `glm5_next` mapping** - so the
   HF -> GGUF conversion is itself a deliverable. The loader dispatches on `general.architecture` against a
   hardcoded list at `src/inference.cpp:856`, so an unknown arch fails with "architecture %s is not supported"
   before anything runs. Tensor name -> slot map from the checkpoint's own `modules_to_not_convert` (a literal
   tensor-name list): `self_attn.{q,k,v,qkv_proj,o,f_a,f_b,g_a,g_b,b}_proj*`, `{q,k,v}_conv1d`, `A_log`, `dt_bias`,
   `o_norm`, `fused_qkvbfg_a_proj`, `indexer.{wq_b,wk,k_norm,weights_proj,index_kpool_compress_ape,
   index_kpool_compress_gate}`, `hyper_connection`, `mapping_proj`, `mlp.gate`, `mlp.gate.e_score_correction_bias`.
   Keys to carry through the loader: `index_topk` (2048), `index_kpool` (4), `index_kpool_compress`,
   `index_share_for_mtp_iteration`, `hc_mult` (4), `hc_sinkhorn_iters` (20), `hc_eps`, `kv_lora_rank` (512),
   `q_lora_rank` (1536), `qk_head_dim` (256), `qk_rope_head_dim` (0), `mla_use_nope`, `swiglu_limit` (10.0),
   `first_k_dense_replace` (3), `num_nextn_predict_layers` (1). De-risk first: extend
   `tools/gen_qwen4exp_tiny.py` into a `gen_glm5next_tiny.py` that writes a 4-layer tiny GGUF (3 dense + 1 DSA,
   8 experts, tiny hidden) from these names, so phases 1-2 below can run on a CPU host. Hermes has ROCm 10.1 access (user, 2026-11-06), as the work host
   does (`/opt/rocm`, hipcc, the gfx1151 GPU); phases 3-4 need the GPU, so they run on the work host.
3. **CPU reference path against the oracle.** The vendored fork already has `llama.cpp/src/models/glm-dsa.cpp`
   implementing MLA + the DSA indexer (`ATTENTION_KEY_LENGTH_MLA`, `ATTENTION_VALUE_LENGTH_MLA`,
   `INDEXER_HEAD_COUNT`, `INDEXER_KEY_LENGTH`, `INDEXER_TOP_K`, `wq_b / wv_b / wo / indexer_k_norm /
   indexer_proj`), so `tools/ref_oracle/ref_dump` is a working oracle for the sparse path from day one. Check
   whether that file also has mHC and the KDA per-channel decay; if not, the reference for those does not exist
   and has to be written from HF's modeling file. CPU ops in the `qwen_ops.cpp` style the forward pass and
   `tests/forward_verify.cpp` already share: the KDA step (per-channel decay), indexer score -> kpool compress ->
   top-k -> gather, the mHC Sinkhorn mixer, the MoE router **with sigmoid scoring + `e_score_correction_bias`**
   (this router does softmax - a real difference, not a detail: sigmoid + bias, then top-8, then renormalize, x2.5)
   and the `swiglu_limit` clamp at 10.0.
   **The #1 correctness risk is here, not in the kernels:** the indexer's top-k is a *selection*, not a summation.
   Exactness by construction (batched kernels reproducing the single-token summation order bit for bit under
   `QMATVEC_NO_WMMA=1`) does not cover selection order - a prefill of N tokens must select exactly the entries N
   single-token steps would have, or the sparse attention reads different positions and the logits diverge
   silently. Pin the tie-break down (ties to the lower index, as `moe_router` already does) and test it as a
   property, not as a perplexity number.
4. **GPU kernels.** KDA's decay is per-**channel** diagonal, not this engine's per-head scalar gate: `g_out` grows
   from `[T][n_v]` to `[T][n_v*128]` and the state update becomes `S *= exp(g_c)` per channel, in
   `deltanet_pre`/`deltanet_rule` (`DELTANET_HEAD_SIZE` is already 128 and KDA's head_dim is 128, so the state
   shape carries over unchanged). The indexer's top-2048 selection over the k=4-pooled index is the same problem
   shape as the bitonic top-k in `moe_router` (32 index heads x 128 in, an index out) - reuse that machinery.
   Prefill stays dense FP16-WMMA flash attention (with the whole prefix there is nothing to select); decode gathers
   the selected 2048 positions, replacing the split-K-over-256-position-chunks decode for those 11 layers only.
5. **KV cache layout + quantization, both gates.** MLA stores one latent of `kv_lora_rank` 512 (+ a 64 rope slot;
   `qk_rope_head_dim` is 0, so NoPE) per position, not per-head K/V: `kv_bytes_per_token` is computed at load from
   the `attn_k` / `attn_v` tensor widths in `src/inference.cpp` and must become a latent-cache path or the KV budget
   is wrong by a large factor. It is a big win: 576 floats x 11 layers = **12 KB/position** in F16 against 704
   KB/position for a 64-head x 256-dim layout, and that is what makes 1M context addressable. The block store's
   SHA-hashed `<hash>.kvb` slots must also key on the **compressed index**, or a resumed sequence has no index for
   its reused prefix. On quantization: the release is FP8 with dynamic scaling and gfx1151 has no FP8/BF8/FP6/FP4
   instructions (ARCHITECTURE.md s4), so either an e4m3 -> F16 LUT decode feeding the F16 WMMA path (256-entry LUT
   = 512 B, fits LDS) or offline requantization to ROCmFP4_FAST; the `v_perm` codebook trick that the int8 path
   uses for 4-bit codes does not reach 8-bit codes (3-bit indices only), so there is no free version of the first.
   Decide before the kernels, because it sets the resident set and therefore the tok/s in item 1.

What breaks that the new-kernel list does not show: `index_share_for_mtp_iteration: true` means the draft head
shares the main model's index, so the verify path's `deltanet_mid_states` / `qwen_model_rollback(keep)` has to roll
the index back too or rebuild it per pass; the `n_hc == 0` dispatch (`src/inference_internal.h:95`) branches the
whole forward pass's residual handling, and mHC replaces *every* norm, so the decode profile's `hc_norm` 98 x 7.6 us
and `hc_mix` 97 x 2.9 us items stay and grow (a per-token 4x4 doubly-stochastic matrix, cheap per layer but on the
critical path of all 45); and layer kinds are classified from the tensors present, never by index, so the
classifier's predicate must learn KDA's `A_log`/`dt_bias`/`b_proj`/`f_a_proj` names or every KDA layer silently
falls through to the wrong kind and the model runs wrong-but-silently. Vision (`glm5_next_vision`: 24 layers,
hidden 1024, 16 heads, patch 14, 448 px, `spatial_merge_size` 2, `temporal_patch_size` 2, `projection_intermediate_size`
10240 -> `out_hidden_size` 4096) is an independent track that can go last: the engine has no encoder and the C API
is text-in/text-out, so it needs an encoder path, a patch + merger projection to `n_embd`, the image/video token
interleave (image_token_id 154854, image_start/end 154830/154831, video_start/end 154832/154833) and the Perl
template's thinking channel - the engine streams every emitted token, so `reasoning_effort` (low/high/max, default
max) and `clear_thinking` (default false) need the reply-kind channel flag currently listed as "only if a non-Perl
user needs it". A GLM tokenizer (vocab 154,880) is a GGUF-metadata problem for `tokenizer.cpp`, not a new tokenizer.

Ordering: the gate comes first because its answer decides whether the rest is worth doing on this hardware. The
phases after it are cheap, CPU-only and oracle-backed, and are where the indexer's selection-order risk resolves;
GPU time on KDA / indexer / sparse attention belongs after those. An optional cheaper first target: a smaller DSA
sibling, which the oracle already knows as `glm-dsa`, exercises KDA + indexer + MLA + MoE with none of the capacity
problem, then scale up to the full model.

## Concurrent runs (to investigate; user, 2026-11-05)

Found in the opencode config (opencode_fn.json, not in the repo): each agent is a slot (`X-LLamaCPP-Id-slot`: plan 6,
build 5, explore1 4, explore2 2, verifier 3), `max_concurrency` 2, and compaction summarizes with a separate request.
The server keeps ONE sequence, so every switch between slots re-prefills the conversation: that is the 12k-token
reuse in the logs (the shared system prompt and tools), and a 59k prefill at 263 tok/s (227 s).

Decision (user, 2026-11-07): KV persists, and concurrent slots reuse without clearing each other's cache. Switching
is measured fast enough that waiting is not needed: the 27B model's KV is 69,632 bytes per token (F16), so a 12k
prefix is 0.84 GB and a 60k prefix 4.2 GB; /data reads 6.5 GB/s (writes 5.1 GB/s), so restoring 60k takes under a
second against 227 s of re-prefill. Plan:
1. Per-slot sequences: a request carries its slot; the server keeps one sequence per slot (up to --parallel N, LRU
   beyond that) and prefers the slot's own sequence for its prefix.
2. Persistence through the block store (`--store`, already resumes blocks and recurrent snapshots): on by default
   per model, so an evicted slot and a restart resume from disk.
3. Measure the restore on the real model before choosing the default N (KV budget: 5 slots at 131072 context is about
   45 GB of F16 KV for this model; the store makes N a latency choice, not a memory one).


The server serves one request at a time unless `--parallel N`, which batches N conversations of one model (not
with a draft head); with `--models-dir` each loaded model has its own sequences and the loop steps them in turn. A
request that arrives during another, or during a model load (2 s from the page cache, minutes from a cold disk),
waits: a test request waited 7 min 22 s, then ran at 38 tok/s. To look at: `--parallel` by default, and with
speculative decoding (a draft per sequence, or verify batches); several models generating at once (one forward pass
each per round; GPU sharing); long prefills in chunks between other requests' tokens (prefill_chunk exists); loads
off the request path (between rounds, or in a second process); fairness and queue limits per user (with the
logging's user and conversation id); llama-server on the same GPU.

## Server logging (implemented, not yet run end to end)

The engine's event hook (C), its Perl binding (`set_event_handler`), the formatter `ROCm::Inference::Log`, the server's
lines (start, done, prefill progress, loads, access) and the options `--log-level`, `--log-format`, `--log-color` are
in. Unit tests: `t/05-log.t` (formatter), `t/15-events.t` (GPU, a real generate). Not yet run:
- **Server test** (`t/60-server.t`, to extend): the server on a tiny model with `--log-level debug --log-format json`,
  one chat request; the captured lines parsed as JSON, in order: prefill_begin, prefill_progress (up to the total),
  prefill_end, the first token, done. The first token has no line yet: the done line carries the first-token time.
- **Soak on the Flash model**: one request with a long prompt; the prefill ticks visible in real time, no per-token
  lines. Needs a model location that pins (see the draft-head item under Open leads).
Identity: the user (nginx X-Remote-User; a Keycloak JWT later), the client address (X-Real-IP), and the conversation
id (the X-LLamaCPP-Id-slot header of the opencode fixups). The docker repo's nginx needs `proxy_set_header
X-Remote-User` on /rocm/ (not done here).

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
- **Qwen3.8-Flash-Next beyond 262,144 tokens (YaRN)** (HF card, 2026-11-06). The checkpoint is 262,144 natively
  (`max_position_embeddings`, `rope_type` default); the card reaches 1M by a YaRN override (`factor` 4.0, original
  262144, with an attention scale). Nothing is dropped by the converter: the GGUF has no rope-scaling keys, and the
  engine has no YaRN. To do: YaRN frequencies and the attention scale in the RoPE, a `--rope-factor` option, and a
  quality check at long context (not only that it runs). The 16 GB KV budget caps it at about 600k positions at F16
  even then. Keep in mind; not started.

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
- Chunked prefill is not split-invariant on the WMMA path: a split at any point other than the engine's micro-batch
  changes logits by up to ~0.7 (0.6B, one 1128-token prompt); exact with `QMATVEC_NO_WMMA=1`. Streaming with
  `prefill_chunk`, and a store's block splits, inherit it. Decide whether the WMMA path should be exact.
- Draft events are not verified: `events_test MODEL DRAFT` is written but not run. The Flash model does not pin on
  the work host (ext4 /models, a 67.7 GB range: invalid argument), and there is no small model with a draft head
  (tools/gen_qwen4exp_tiny.py writes none). Needs a huge-page location for the Flash model, or a generator for a tiny head.
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
