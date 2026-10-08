// ==================== Embedding Lookup ====================
extern "C" __global__ void embedding_lookup(const int* token_ids, const __half* embed_table,
                                            __half* output, int batch_size, int hidden_size, int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int token = token_ids[idx];
    if (token < 0) token = 0;
    if (token >= vocab_size) token = vocab_size - 1;

    const __half* src = embed_table + (size_t)token * hidden_size;
    __half* dst = output + (size_t)idx * hidden_size;

    // Simple element-by-element copy (avoids potential uint4 alignment issues on AMD)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

        }
    }
}

// Position ID generation for RoPE (CPU-side convenience, but can be GPU if needed)
extern "C" __global__ void build_position_ids(int* position_ids, int start_pos, int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < seq_len) {
        position_ids[idx] = start_pos + idx;
    }
}