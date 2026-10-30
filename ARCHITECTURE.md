# Architecture

How the inference engine in this repository is built: a standalone, zero-copy LLM engine for **AMD Strix Halo
(gfx1151, RDNA3.5 APU, 128 GB unified memory)**, a C library (`libinference.so`) with a Perl XS binding
(`ROCm::Inference`). This file describes the design as it is; [PLAN.md](PLAN.md) holds the status, the
measurements behind each decision and the history (phases).

## 1. Principles

- **Zero copy.** Model files are `mmap()`ed and registered with `hipHostRegister`; on the UMA APU the host
  pointer is the device pointer, so kernels read the weights in place, in their native GGUF block format. No
  `hipMemcpy` of weights or activations, no repacking.
- **GPU first, CPU too.** Everything per layer runs on the GPU as one queue per forward pass with a single sync;
  the CPU does what is cheap and irregular there: token-embedding rows, PLE hash rows, tokenization, sampling.
- **Bandwidth is the budget.** Generation reads every active weight once per token (~190–230 GB/s of the
  ~256 GB/s peak), so kernels are judged by bytes streamed and launches; prefill reads each weight once per
  tile of tokens.
- **Exactness by construction.** Every batched kernel sums each token in the order of its single-token kernel,
  so a prompt fed at once, token by token, split, restored from a checkpoint or verified speculatively gives
  the same bits (with `QMATVEC_NO_WMMA=1`; see section 5).
- **Verified against a reference.** llama.cpp (the ROCmFPX fork in `llama.cpp/`, CPU build) is an oracle only:
  `tools/ref_oracle/ref_dump` dumps its activations, the tests compare with them. Nothing links it.
- **Thin bindings.** The C API is architecture neutral; the Perl module maps it function by function (handle
  first), with policy (chat templates, store eviction) in plain Perl.

## 2. Repository layout

| Path | Role |
|---|---|
| `src/gguf_mmap.{h,cpp}` | GGUF parser on a read-only mapping; split models (`-0000N-of-0000M.gguf`) as one model |
| `src/inference.{h,cpp}`, `inference_internal.h` | C API, model loading (hyperparameters, per-layer binding), sequence state, generation loop, prefix reuse, KV block store, draft head |
| `src/qwen_model.{h,cpp}` | The forward pass (`QwenModel`): scratch, recurrent state, layer kinds, MoE, hyper-connections, PLE, LM head, MTP, verify / rollback |
| `src/matvec_q.hip`, `qmatvec.{h,cpp}` | All weight matmuls: float, int8 (single / few / many tokens), WMMA tiles, expert-indexed variants |
| `src/gpu_ops.{hip,h,cpp}` | Small vector kernels: RMSNorm (+ residual), SwiGLU, MoE router / combine / grouping, HC norm / mix, PLE, MTP concat |
| `src/attention.{hip,h,cpp}` | Gated full attention: q/k norm, RoPE, KV write, split-K decode attention, FP16-WMMA flash attention for prefill |
| `src/deltanet.{hip,h,cpp}` | Gated DeltaNet: conv1d with state, L2 norms, gates, the delta rule, gated RMSNorm |
| `src/q8_device.h` | Device-side Q8_0 / Q8_K activation quantization (bit-identical to ggml's CPU quantizers) |
| `src/kv_cache.{h,cpp}` | KV cache: block-major layout, one file (slot) per block, attach / publish |
| `src/quant_types.{h,cpp}` | CPU decoders of every GGUF type used (bit-exact vs ggml) |
| `src/tokenizer.*`, `sampler.*`, `sha256.*` | Byte-level BPE from the GGUF, sampling, store hashing |
| `src/kernel_loader.*`, `kernels_embed.S`, `libinference.map` | Code objects embedded in the library; exported symbols |
| `src/demo_qwen.cpp` | `qwen_demo`: generation from the command line |
| `tests/` | Oracle and self-consistency tests (section 9); Phase 3 component tests |
| `tools/` | `ref_oracle/ref_dump` (reference dumps), `gguf_info.py`, `gguf_split.py`, `gen_qwen4exp_tiny.py`, `prof_summary.py`, `matvec_bench` |
| `perl/ROCm-Inference/` | The CPAN module (XS + Perl + tests) |
| other `src/*.hip` / `*_main*.cpp` | The original proof-of-concept demos and Phase 3 components (README.md) |

## 3. Memory

All GPU-visible memory is host memory registered with HIP; there are no device allocations.

| Region | Backing | Registration | Notes |
|---|---|---|---|
| Model weights | the GGUF file(s), `mmap(MAP_PRIVATE, PROT_READ)` | read-only, per file | Pages of CPU-only tensors (token embeddings when the LM head is not tied, the qwen4exp PLE table) are left out: they stay demand-paged, so a 94 GB model pins ~64 GB and its 27 GB PLE table can live on disk. A file on disk pins page-cache pages. |
| Scratch (activations) | one unlinked `/dev/shm` file | coarse-grained | Carved into per-micro-batch arrays (up to 1,024 tokens); the GPU caches it, coherent at kernel boundaries |
| Recurrent state | `/dev/shm` file | coarse-grained | DeltaNet conv + delta-rule state per linear layer (thread-ordered for coalescing), PLE conv state and token history; checkpoints are GPU copies of the whole region; `rs_mid` holds the state after the first token of a verify pass |
| KV cache | one `O_TMPFILE` per 256-position block | per slot | Block-major `[K_0 V_0 ... K_L V_L]` per block; a slot is replaced by `MAP_FIXED` remapping (store attach), published by `linkat` |
| Draft head | its own GGUF mapping + one-layer KV cache | as above | Loaded on demand |

## 4. Weights and matmuls

Weights stay in their GGUF blocks; each type has a CPU decoder (tests) and GPU kernels.

- **Float path** (`matvec_<type>`, `matmul_f_<type>`): decode and FMA; any type, any K; used where no int8
  kernel applies (e.g. Q8_0, F16).
- **Int8 path** (`matvec_q8_<type>`): the activation is quantized once (by the producing kernel, `q8_device.h`)
  to ggml's activation format for the weight type (Q8_0 for ROCmFP4, Q8_K for K-quants); weights are decoded
  four values at a time into packed int8 (`v_perm` codebook lookup for ROCmFP4) and multiplied with
  `v_dot4` (`__builtin_amdgcn_sudot4`); float scales per block. ROCmFP4 needs K % 64 (FAST: % 128), K-quants
  K % 256. `L` lanes per row (4..32) by K and the launch's row count; *virtual lanes* keep the 32-lane batched
  summation order for every L.
- **Few tokens** (`matvec_q8t_<type>`, 2–4 tokens): weights decoded once, applied to each token with the
  single-token order (speculative verify passes, the 2-row LM head).
- **Grouped** (`qmatvec_launch_q8_multi[_t]`): up to four matrices on the same input in one launch (DeltaNet
  qkv/z/alpha/beta, attention q/k/v, HC down + inject).
- **Batched** (`matmul_q8_<type>`, tiles of 16 tokens) and **WMMA** (`matmul_wmma_<type>`, int8 WMMA tiles of 64
  tokens × 128 rows for ROCmFP4 / FAST, × 64 rows for K-quants, from 32 tokens): prefill.
- **Experts** (`matvec_q8_id_*`, `matmul_q8_id_*`, `matmul_wmma_id_*`): expert-indexed; prefill batches of 8+
  tokens are grouped by expert (`moe_group`) into tiles (up to 64 entries on the WMMA path, so a 1,024-token
  batch reads each expert's weights about once; 16 otherwise); an id of -1 selects an extra matrix (the shared
  expert as one more slot when generating).
- **LM head**: on the last row only (two rows in a verify pass); opt-in int8 copy of a float head
  (`inference_set_lm_head_q8`).

Supported weight types: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K, MXFP4, NVFP4, ROCmFP4 and
ROCmFP4_FAST; int8 kernels (single token, batched, WMMA dense and expert) for all but the float types. Types that
share a layout share the kernels through a small policy struct: the 36-byte pair family (ROCmFP4, Q4_0, NVFP4), the
17-byte quad family (ROCmFP4_FAST, MXFP4), a block per lane (Q4_1, Q5_0, Q5_1) and the K-quants. NVFP4 from
ModelOpt / compressed-tensors checkpoints carries a global scale per tensor (per expert), `<weight>.scale`: the
loader registers it with qmatvec (`qmatvec_set_out_scale`), and every qmatvec launch on that matrix multiplies its
outputs by it afterwards (`scale_out*` kernels: dense rows, expert slots by id, expert-grouped tiles), as llama.cpp's
`ggml_mul` after the matmul. F32, F16 and BF16 have dedicated 16-byte-load float kernels (F16 / BF16
round the activations to their type first, as ggml). The CPU decoders are bit-exact against the reference's on real
tensors (`tests/dequant_verify`).

**ROCmFP4** (as the reference's `rocmfp4.h` defines it): `Q4_0_ROCMFP4` (type 100) blocks of 32 values in 18
bytes, `qs[16] e[2]`: values 0..15 are the low nibbles (scale `e[0]`), 16..31 the high nibbles (`e[1]`); 4.5
bits per weight. `Q4_0_ROCMFP4_FAST` (101): 17 bytes, `qs[16] e`, one scale; 4.25 bits per weight. A nibble is
a sign bit and an index into `{0, 1, 2, 3, 4, 6, 8, 10}`; a scale byte is UE4M3 at half scale (`ue4m3_half`;
0x7f and above decode to 0). The int8 kernels turn four nibbles into four int8 codes with two `v_perm` lookups
(a positive and a negative 8-entry table, the sign choosing; mask bytes kept ≤ 7, whose behaviour on gfx1151 was
measured, PLAN.md) and multiply them with `v_dot4` against the Q8_0 activation, which is quantized exactly as
ggml's CPU does, so each block's integer sum equals the reference's. Large ROCmFP4 / FAST matvecs run at
185–230 GB/s (peak ~256).

gfx1151 has no FP8 / BF8 / FP6 / BF6 / FP4 instructions at all: AMD's RDNA 3.5 ISA (the machine-readable spec
from GPUOpen, 1,517 instructions) has none, and the ROCm assembler (`llvm-mc -mcpu=gfx1151`) rejects them
(checked 2026-10-19). FP8 / BF8 conversions exist on CDNA 3 (gfx942), RDNA 4 (gfx1200), CDNA 4 (gfx950) and
gfx1250; FP4 / FP6 scaled conversions only on CDNA 4 (gfx950) and gfx1250. So OCP FP4 / FP6 / FP8 weights would
need software decodes here; `v_perm` takes 3-bit indices, so the table lookup used for ROCmFP4 does not extend
to 6-bit codes. No model used here needs them.

## 5. Exactness and WMMA

The batched kernels (`matmul_q8_*`, `matvec_q8t_*`, attention below 16 tokens, the per-token MoE path below 8
tokens) reproduce the single-token sums bit for bit. The WMMA kernels (int8 matmuls from 32 tokens, the
expert-grouped matmuls, FP16 flash attention from 16 tokens) sum in another order: prefill then differs from
token-by-token decoding in the last bits, which routing near-ties can amplify, but not in accuracy against the
reference (also the F32 router logits in prefill: `matmul_f32_tile`). Their integer dots are exact; ROCmFP4_FAST
(one scale per 32 values) adds a block's two 16-value WMMA dots in int32 and scales once, the same per-block term as
the dot4 kernels, so only the order of the float sum over blocks differs (within ~5e-6 relative of the dot4
kernels). `QMATVEC_NO_WMMA=1` (all), `QMATVEC_NO_WMMA_MM=1` (matmuls), `ATTN_NO_FLASH=1` (attention) switch them
off; the exactness tests run with `QMATVEC_NO_WMMA=1`.

## 6. Models

Layer kinds are classified from the tensors present (never from the index): full attention (`attn_q`) or Gated
DeltaNet (`ssm_alpha`); MoE (`ffn_gate_inp`) or dense FFN.

| Architecture | Models | Blocks |
|---|---|---|
| `qwen3` | Qwen3-0.6B | plain attention only (head_dim 128, no output gate), dense SwiGLU; no recurrent state |
| `qwen35` | Qwen3.5-0.8B, Qwen3.5-4B | 3 DeltaNet : 1 gated attention (head_dim 256), dense SwiGLU, RMSNorm residual |
| `qwen35moe` | Qwen3.6-35B-A3B, Qwen3.5-122B-A10B | the same with a 256-expert top-8 MoE + gated shared expert |
| `qwen4exp` | Qwen3.8-Flash-Next (177B) | + hyper-connections, PLE, 512 experts top-10, sigmoid DeltaNet gate, MTP draft head |

Attention kernels are compiled for head_dim 128 and 256 (`attn_*_128` / `_256`); `attn_gated` (from `attn_q`'s
shape) selects the [Q | gate] layout and the output gate. Prefix reuse and the KV block store resume an
attention-only model at any position / block (no recurrent state, no snapshots).

**Forward pass** (`QwenModel::forward`, T tokens at positions pos0..): the CPU decodes the embedding rows into
the residual, then each layer: norm → token mixer (attention with the KV cache, or DeltaNet with the recurrent
state) → residual add + norm → FFN / MoE → residual add fused with the next norm (`add_rmsnorm`); the LM head
on the last token and one sync. Prefill runs this in micro-batches of up to 1,024 tokens
(`inference_set_prefill_batch`: 16 to 8,192, up to ~10% faster long prompts on MoE models at 4,096, for ~1 MB of
scratch per token on Qwen3.8).

- **Attention**: gated Q (`[Q | gate]` per head), q/k RMSNorm, partial NeoX / IMRoPE, GQA, sigmoid output gate.
  Decode: split-K partial attention over 256-position chunks + combine; prefill: FP16 WMMA flash attention.
- **Gated DeltaNet**: conv1d (state of the last 3 inputs) + SiLU, L2-normed q/k, decay `exp(softplus(alpha +
  dt) * A)` and `sigmoid(beta)`, the delta rule over a 128×128 state per value head (8 lanes per column, shuffle
  reductions), gated RMSNorm (`SiLU(z)`; `sigmoid(z)` for qwen4exp).
- **MoE**: F32 router → softmax, top-k by bitonic sort (ties to the lower index), renormalized weights and the
  sigmoid shared-expert gate (`moe_router`) → gate/up, SwiGLU, down per selected expert → weighted sum + gated
  shared expert (`moe_combine`).
- **Hyper-connections (qwen4exp)**: the residual is 4 streams of `n_embd`. Before each mixer and FFN a *mixer*
  replaces the norm: per-stream RMSNorm with a gamma over all streams (`hc_norm`, also combining the previous
  block's output into each stream with weights `2 sigmoid(inject / 4)`), a rank-320 gate `sigmoid(up(SiLU(down
  / 4)))`, and the mean of the gated streams (`hc_mix`). The final mixer replaces `output_norm`. The up
  projection (K = 320) runs from a zero-padded K = 384 copy so the int8 / WMMA kernels apply.
- **PLE (qwen4exp, layer 1)**: per token, 16 rows of a 320M-row table selected by hashing the 2- and 3-gram of
  token ids (CPU; EOS cuts the n-gram), projected to a key (gating each stream) and a value, plus a dilated
  causal depthwise conv with state (`ple_gate`, `ple_conv`).

## 7. Speculative decoding (MTP)

A nextn sidecar GGUF (`inference_set_draft_model`) holds one more full-attention MoE block with its own
embeddings, final mixer and LM head. Its row at position p combines the token at p and the main model's residual
streams at p − 1 (`eh_proj` per stream); every main forward runs it on its rows, so its one-layer KV cache
follows the sequence. The head drafts k tokens (`inference_set_draft_tokens`: 1–3, or by default 2 or 3 per
pass, whichever gives the most tokens per second by the recent acceptance of each draft position and the
measured pass times): its argmax,
then the head again on that draft with its own output streams in place of the model's (`qwen_model_mtp_drafts`).
Generation decodes `[token, drafts]` in one pass of k + 1 tokens (`qwen_model_forward_verify`), exactly as
single-token passes, keeping the recurrent state after each of the first k tokens (DeltaNet conv / delta-rule
state, PLE conv state and history; `deltanet_mid_states`). Each next token is sampled from its row's logits as
always: while it equals the next draft, that draft is already decoded; at the first that differs
`qwen_model_rollback(keep)` restores the state after the kept tokens. A pass never spans a KV-store snapshot
boundary. Output equals plain decoding for any sampler. On chat replies (acceptance ~0.97): 2.0 / 2.9 / 3.8
tokens per pass with 1 / 2 / 3 drafts (51 / 60 / 65 tok/s vs 34 plain); on prose and Chinese the head guesses
less well and 2 is best; the adaptive default is within ~1% of the best fixed count on each.

## 8. The library layer

- **Sequence**: the model holds one sequence (`n_past`, the token history); `inference_generate_stream`
  continues it: the prompt is appended (or, with `reuse_prefix`, is the whole sequence and the longest common
  prefix is reused), then tokens are sampled, streamed (UTF-8- and stop-string-safe) and decoded until EOG, a
  stop string, the callback or `max_new_tokens`. Every emitted token is decoded, so calls chain.
- **Prefix reuse**: KV rows are just overwritten; the recurrent state cannot be rewound, so prompts leave GPU
  checkpoints of it (`checkpoint_tail` tokens before their end) to restart from.
- **KV block store** (`inference_store_open`): every full 256-token block of a sequence is published as a file
  named by the SHA-256 chain hash of its tokens (`<hash>.kvb`, the root hash covers the model and the layout),
  with recurrent-state snapshots (`.rst`) every N blocks and at each prompt's last boundary. A new sequence maps
  the longest stored prefix in place (copy on write before writing into shared blocks) and resumes from its
  deepest snapshot; any process sharing the directory benefits. Eviction is deleting files (policy in
  `ROCm::Inference::Store`).
- **Shared library**: `libinference.so.1` contains the whole engine with the code objects embedded; it exports
  only `inference_*`, `tokenizer_*`, `sampler_*`. `make install PREFIX=…` installs it with headers and a
  `rocm-inference.pc`.

## 9. Verification

| Test | Checks |
|---|---|
| `forward_verify` | each engine piece fed the reference's inputs vs the reference outputs; the whole model per layer (4B, 35B) |
| `qwen4exp_verify` | per-layer streams, single layers by teacher forcing, logits; token-by-token / split / restored = prefill |
| `generate_verify` | streaming, stop strings, chat turns with prefix reuse, LM-head option |
| `mtp_verify` | draft acceptance, speculative = plain (greedy, sampled, library) |
| `store_verify.sh` | the KV block store across processes (optionally with a draft head) |
| `matvec_verify`, `dequant_verify`, `kv_verify`, `tokenizer_*`, `sampler_verify`, `capi_smoke` | kernels, decoders, tokenizer, C API |
| `perl/ROCm-Inference/t/*.t` | the Perl binding (GPU tests with `ROCM_INFERENCE_TEST_*`) |

Models without room for their weights are verified on small random-weight replicas with the real tensor
layout (`tools/gen_qwen4exp_tiny.py`).

## 10. Performance (gfx1151, measured 2026-10-20)

`qwen_demo`, greedy, second run (weights in the page cache); prefill of a 2,427-token prompt; generation of 256
tokens after a short prompt. Runs vary by about ±5%.

| Model | Quantization (size) | Generation | Prefill |
|---|---|---|---|
| Qwen3-0.6B | Q8_0 (0.6 GB) | 184 tok/s | 7,200 tok/s |
| Qwen3.5-0.8B | UD-Q8_K_XL (1.2 GB): Q8_0 / F16 matrices | 120 tok/s | 4,110 tok/s |
| Qwen3.5-4B | ROCmFP4 (2.9 GB) | 59 tok/s | 1,850 tok/s |
| Qwen3.6-35B-A3B | ROCmFP4 (20 GB), BF16 head | 62 tok/s (more with the int8 head option) | 2,075 tok/s |
| Qwen3.8-27B (dense) | Q4_0_ROCMFP4_FAST (14.6 GB) | 13.8 tok/s (~200 GB/s of weights) | 296 tok/s |
| Qwen3.5-122B-A10B | UD-Q4_K_XL (77 GB): Q4_K / Q5_K experts, Q8_0 dense | 22.6 tok/s | 620 tok/s |
| Qwen3.8-Flash-Next (177B) | Q4_0_ROCmFP4_FAST (94 GB) | 34–38 tok/s; 48–55 with the MTP head | 826 tok/s |

Qwen3 MoE models (`qwen3moe`: no shared expert) are not supported yet.

Profiling: `QWEN_PROFILE=1` (per-section times), `rocprofv3 --kernel-trace` + `tools/prof_summary.py` (per kernel).
