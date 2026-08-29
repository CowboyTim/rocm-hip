# gfx1151 native MFMA/WMMA/FP4 + KV‑cache fast‑path – 13‑week roadmap
*(All items keep the original 13‑week calendar; the KV‑cache / prefill work is parallelised with the kernel development because it touches the same files and dispatch points.)*

---

## 0️⃣  Quick‑look – Where KV‑cache / prefill lives today
| Component | File(s) | Current path |
|-----------|---------|--------------|
| KV‑cache allocation / layout | `src/llama-kv-cache.cpp`, `src/llama-kv-cache‑iswa.cpp`, `src/llama-kv-cache-dsv4.cpp` | Linear buffer, 2‑D (layers × heads) with stride‑based indexing. |
| Prefill (prompt processing) | `llama.cpp → llama_encode → llama_eval → ggml_graph_compute → ggml_cuda_mul_mat_q / ggml_cuda_flash_attn` | Calls the same GEMM / Flash‑Attention kernels used for decode. |
| Cache‑update (decode step) | `llama_decode → ggml_cuda_mul_mat_q` (K‑proj, V‑proj) + `ggml_cuda_flash_attn` | Same kernels as prefill, but one token at a time. |

Because **prefill = one large GEMM + Flash‑Attn** and **decode = many tiny GEMM + Flash‑Attn**, any kernel‑level speed‑up automatically benefits both. The extra work is to *avoid redundant memory movement / layout changes* when the prompt is long and the cache is filled in one shot.

---

## 1️⃣  Extended Discovery (Weeks 1‑2) – KV‑cache / prefill audit
| New task | Owner | Deliverable |
|----------|-------|-------------|
| Map current KV‑cache layout – draw exact `ggml_tensor` shapes for `K` and `V` per layer/head, note strides, padding, contiguity. | KV‑cache engineer | Diagram + table (layer, head, `K` shape, `V` shape, stride). |
| Profile prefill vs. decode – run a 4k‑token prompt on gfx1151, capture `rocprof` traces for prefill GEMM, first decode step, steady‑state decode. | Perf engineer | CSV + rocpdf screenshots. |
| Identify “copy‑on‑write” – does the current code copy the prompt’s K/V into the cache, or write directly into the pre‑allocated cache buffer? | KV‑cache engineer | Boolean + code pointers. |
| Add a “prefill‑only” benchmark – `llama-cli -p "<4k‑token prompt>" -n 0` (no decode) to isolate prefill cost. | Perf engineer | Baseline numbers (tokens/s, latency, LDS/reg usage). |

*All of this lives in the same Week 1‑2 window as the original discovery (baseline ROCm‑FP4 kernel audit, ISA gap analysis, baseline token‑gen numbers).*

---

## 2️⃣  Build‑system flag – add a **KV‑cache fast‑path switch** (Week 2‑3)

| New CMake option | Location | Reason |
|------------------|----------|--------|
| `GGML_USE_GFX1151_KV_FAST` (default **OFF**) | `ggml/CMakeLists.txt` → propagated to `ggml/src/ggml-hip/CMakeLists.txt` | Turns on the *in‑place KV‑cache write* + *prefill‑only Flash‑Attn* kernels without affecting the generic path. |
| Propagate to `build_llama.cpp.sh` (`-DGGML_USE_GFX1151_KV_FAST=ON`). | Build script | One‑line enable for Strix‑Halo. |

*No code changes yet – just the switch.*

---

## 3️⃣  Core kernel work – **MFMA / WMMA / VOP3P** (Weeks 3‑8)  
*Exactly the same kernels as the original plan.*  
**Additional requirement:** make the new MFMA/WMMA kernels **KV‑cache aware** (see §4).

| Phase | Kernel | KV‑cache / prefill hook |
|-------|--------|--------------------------|
| **3.1 MFMA GEMM primitive** | `ggml/src/ggml‑hip/mfma_gfx1151.cu` (new) | **Accept an optional `dst_base + dst_stride`** that points directly into the pre‑allocated KV buffer; if `dst_stride == 0` fall back to temporary buffer (current behaviour). |
| **3.2 WMMA fallback** | `wmma_gfx1151.cu` (new) | Same stride‑aware signature. |
| **3.3 VOP3P dot‑product** | `dot_vop3p_gfx1151.cu` (new) | No change – used by both prefill & decode. |
| **3.4 Flash‑Attn gfx1151** | `fattn_gfx1151.cu` (new) | **Two entry points**: <br>• `flash_attn_prefill` – takes the whole prompt K/V tensors, writes **directly into the KV cache** (no intermediate copy). <br>• `flash_attn_decode` – unchanged (single‑token). |
| **3.5 mmq / mmvq dispatch** | `mmq.cu`, `mmvq.cu` | When `src0->type == GGML_TYPE_Q4_0_ROCMFP4` **and** `GGML_USE_GFX1151_KV_FAST` → call the stride‑aware MFMA GEMM that writes straight into the KV cache. |
| **3.6 Copy / pack** | unchanged | Verify `cpy_q_q_block` still optimal for the new layout (it already copies packed FP4 bytes). |

**Acceptance criteria** (same as original + KV‑cache specific):

* Prefill of 4 k tokens writes K/V **directly** into the cache (zero extra `hipMemcpyAsync`).  
* Decode step sees the same layout – no extra transpose.  
* End‑to‑end token/s ↑ ≥ 1.8× **and** prefill latency ↓ ≥ 30 % (because the copy is gone).

---

## 4️⃣  KV‑cache / Prefill Integration (Weeks 4‑6)

| Week | Activity | Owner | Exit criteria |
|------|----------|-------|---------------|
| 4 | **In‑place KV write path** – modify `llama_kv_cache_update` to call the new stride‑aware MFMA GEMM instead of `ggml_cuda_mul_mat_q` + `hipMemcpyAsync`. | KV‑cache engineer | Unit test: `llama_kv_cache_update` writes exactly the same bytes as before (memcmp). |
| 5 | **Prefill‑only Flash‑Attn entry** – add `flash_attn_prefill` that receives the whole prompt K/V and writes into the KV cache in a single kernel launch. | Flash‑Attn engineer | Prefill latency ↓ ≥ 30 % on 4k‑token prompt (measured with `rocprof`). |
| 5‑6 | **Decode path unchanged** – ensure existing decode path still works (no regression). | QA engineer | All existing decode tests pass. |
| 6 | **Benchmark suite** – run `llama-bench -m model.fp4.gguf -p <4k> -n 0` (prefill only) and `llama-bench -m model.fp4.gguf -p <4k> -n 128` (prefill + decode). | Perf engineer | Prefill latency ↓ ≥ 30 %; overall tokens/s ≥ 1.8× baseline. |

*All of these weeks overlap with the original kernel development (Weeks 3‑8) – you are just adding a few extra hooks.*

---

## 5️⃣  Integration & Dispatch (Weeks 8‑9) – same as original + KV flag

| Task | Owner | Note |
|------|-------|------|
| Add `if (GGML_USE_GFX1151_KV_FAST && ggml_hip_same_mapped_region(...))` in `ggml_cuda_cpy` to **skip the copy** when source & destination are the same mapped region (already done for zero‑copy). | Copy engineer | Re‑use existing helper. |
| Extend `ggml_cuda_mul_mat_q` / `ggml_cuda_flash_attn` with `#if defined(GGML_USE_GFX1151_KV_FAST)` guards that call the new *prefill* / *KV‑aware* kernels. | Dispatch engineer | Same guard pattern as `GGML_USE_GFX1151_MFMA`. |
| Expose both flags in `build_llama.cpp.sh` (`-DGGML_USE_GFX1151_MFMA=ON -DGGML_USE_GFX1151_KV_FAST=ON`). | Build engineer | One‑line enable. |

---

## 🔟  Validation, Testing & Performance (Weeks 10‑12) – **add KV‑cache specific tests**

| Test | Metric | Target |
|------|--------|--------|
| **Unit test – KV‑cache in‑place write** | `memcmp(cache_before, cache_after) == 0` after prefill | ✅ |
| **Prefill‑only latency** (4k tokens) | `rocprof` wall‑time | ≤ 0.7× original |
| **Decode steady‑state throughput** | tokens/s | ≥ 1.8× baseline |
| **End‑to‑end token generation (prefill + decode)** | tokens/s, latency | ≥ 1.8× baseline, latency ↓ ≥ 20 % |
| **Memory traffic** (`rocprof --stats`) | LDS / VRAM bytes moved | ↓ ≥ 30 % for prefill |
| **Numerical parity** (logits vs. FP32 CPU) | max‑abs‑diff < 1e‑4 | ✅ |
| **Long‑run stress (1 h)** | No crashes, no drift | ✅ |

---

## 🔟 Documentation, Release & Handoff (Week 13) – add KV‑cache sections

| Doc update | Owner |
|------------|-------|
| `docs/backend/ROCm.md` – new section **“gfx1151 KV‑cache fast‑path”** (flags, expected speed‑up). | Docs owner |
| `docs/build.md` – list both flags (`GGML_USE_GFX1151_MFMA`, `GGML_USE_GFX1151_KV_FAST`). | Docs owner |
| Release notes – “gfx1151 native MFMA/WMMA + zero‑copy KV‑cache prefill”. | PM |
| Upstream PR – single PR containing both flag sets. | Repo maintainer |

---

## 📋  Updated 13‑week Gantt (condensed)

| Week | Main focus (original) | **Extra KV / prefill work** |
|------|----------------------|----------------------------|
| 1‑2 | Discovery, baseline | **KV‑cache audit, prefill profiling** |
| 2‑3 | Build‑system flags (`GGML_USE_GFX1151_MFMA`) | **Add `GGML_USE_GFX1151_KV_FAST`** |
| 3‑5 | MFMA GEMM primitive | **Stride‑aware MFMA API** |
| 5‑6 | WMMA + VOP3P | **No extra** |
| 6‑7 | Flash‑Attn gfx1151 | **Add `flash_attn_prefill`** |
| 8‑9 | Dispatch + build flag | **Add KV‑fast dispatch** |
| 4‑6 (parallel) | – | **In‑place KV write, prefill kernel** |
| 9‑11 | Validation | **Add KV‑cache benchmarks** |
| 12 | Power / stress | **Same** |
| 13 | Docs / release | **Add KV‑cache docs** |

*Total calendar stays **13 weeks**; the KV‑cache work is **parallelised** with the kernel development because it touches the same files and dispatch points.*

---

## 🎯  Bottom line

| What you gain | How |
|---------------|-----|
| **2×‑wide MFMA / WMMA** for FP4 GEMM | MFMA kernels (Week 3‑5) |
| **Native 4‑bit matrix‑multiply** (no de‑quant) | Fused de‑quant inside MFMA (Week 3) |
| **VOP3P dot‑products** for attention scores | Dot kernels (Week 5) |
| **Zero‑copy KV‑cache prefill** (no extra `hipMemcpy`) | In‑place MFMA write + `flash_attn_prefill` (Weeks 4‑6) |
| **Overall token‑generation ↑ ≈ 2×**, prefill latency ↓ ≥ 30 % | End‑to‑end bench (Week 10‑12) |

All of this lives under **two orthogonal CMake flags** (`GGML_USE_GFX1151_MFMA` and `GGML_USE_GFX1151_KV_FAST`) so you can enable them independently on any gfx1151 system. The plan re‑uses the same code paths you are already building for the MFMA path, keeping the engineering effort modest while delivering a **large, measurable win for both prompt processing and steady‑state decoding** on Strix Halo (gfx1151) APUs.

---

## 9️⃣  Quick‑start Checklist for the First Engineer

```text
[ ] Clone the rocmfp4‑strix‑upstream‑merged branch
[ ] Add GGML_USE_GFX1151_MFMA and GGML_USE_GFX1151_KV_FAST to ggml/CMakeLists.txt
[ ] Create ggml/src/ggml-hip/mfma_gfx1151.cu/.cuh
[ ] Implement a 64×64×64 FP4‑aware MFMA GEMM (inline asm)
[ ] Add unit test (GoogleTest) – compare to reference GEMM
[ ] Hook the kernel into mmq.cu via GGML_USE_GFX1151_MFMA guard
[ ] Add stride‑aware signature for KV‑cache write
[ ] Implement flash_attn_prefill in fattn_gfx1151.cu
[ ] Wire GGML_USE_GFX1151_KV_FAST dispatch in ggml_cuda_mul_mat_q / ggml_cuda_flash_attn
[ ] Run ./build_llama.cpp.sh -DGGML_USE_GFX1151_MFMA=ON -DGGML_USE_GFX1151_KV_FAST=ON
[ ] Verify token/s on a 7B FP4 model (≥1.8× speed‑up)
[ ] Run prefill‑only benchmark (llama-bench -p <4k> -n 0) – latency ↓ ≥ 30 %
```

---

## 📂  How to persist this plan

Save the file above as **`PLAN.md`** in the repository root (or any location you like).

```bash
cat <<'EOF' > /home/tim/llama.cpp.git/PLAN.md
# <paste the whole markdown content above>
EOF
```

When you open the project next time, the file will be there and you can continue exactly where you left off.

*If you prefer a different location, just change the path in the `cat` command.*

---

*That’s the full, merge‑ready plan – you can now `git add PLAN.md && git commit -m "Add 13‑week gfx1151 MFMA + KV‑cache plan"` and push it so the whole team sees it.*