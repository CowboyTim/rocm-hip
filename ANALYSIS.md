# Qwen3.8-Flash-DGX Local Build Analysis

## Architecture Summary

This local build (`Qwen3.8-27B`) is **NOT** the official Qwen3.8-Flash-Next model. It appears to be a smaller, distilled variant from the Qwen3.5/Qwen3 series.

### Key Differences

| | **Local Model** | **Official Qwen3.8-Flash-Next (HF/DGX)** |
|---|---|---|
| Param Count | ~27B total | ~180B total (~6B active) |
| Type | **Dense** (single FFN per layer) | **MoE** (512 experts, top-10 routed) |
| Layers | 64 decoder layers | 48 decoder layers |
| Attention | Alternating linear_attention / full_attention (every 4th) | Hybrid GDN + QSA sparse attention + multi-hyperconnections |
| N-gram (PLE) | **No** n-gram embedding table or caching | **Yes** — ~51B PLE n-gram embedding tables injected into tokens |
| Mamba SSM | Yes (GDN uses mamba_ssm_dtype float32) | Yes (same lineage) |
| Multimodal | Has vision_config but no image/video tokens in local files | Text + Image + Video (vision_start/end, audio tokens) |
| Tokenizer | BPE (`Qwen2Tokenizer`), vocab 248,320 | Same base tokenizer |

### Architecture Details

**Local Model:**
- Dense transformer with alternating linear and full attention layers
- No expert routing, no MoE parameters
- No n-gram caching infrastructure
- bf16/fp16 precision (no NVFP4 quantization)
- Uses `Qwen3_5ForConditionalGeneration` architecture

**Official Qwen3.8-Flash-Next:**
- Mixture-of-Experts: 512 experts per layer, top-10 active (~6B params activated)
- N-gram embedding tables (~51B params for PLE lookup)
- NVFP4 W4A4 quantization via NVIDIA Model Optimizer
- Hybrid GDN + QSA sparse attention + multi-hyperconnection streams
- Full multimodal support (text, image, video, audio)

## Conclusion

This local build is a dense variant from the Qwen3.5/Qwen3 series, not the Flash-Next MoE+n-gram model described on GitHub or HuggingFace. It lacks the key innovations of the official Flash-Next model:
1. **No MoE routing** — single feed-forward network per layer
2. **No n-gram caching** — no PLE (Paged LLM Embedding) infrastructure
3. **Smaller scale** — ~27B vs ~180B total parameters (~6B active in Flash-Next)
