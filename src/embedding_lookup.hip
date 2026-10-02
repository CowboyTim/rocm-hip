// embedding_lookup.hip -- Phase 4: embedding table lookup for Qwen 3 token IDs.
// Input: token_ids [batch_size] int32, embedding_table [vocab_size x hidden_size] FP16
// Output: embeddings [batch_size x hidden_size] FP16

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void embedding_lookup(const int* token_ids, const __half* embedding_table,
                                            __half* output, int batch_size, int hidden_size, int vocab_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    int token = token_ids[idx];
    if (token < 0) token = 0;
    if (token >= vocab_size) token = vocab_size - 1;

    const __half* src = embedding_table + (size_t)token * hidden_size;
    __half* dst = output + (size_t)idx * hidden_size;

    // Vectorized copy: 8 elements per thread (128-bit)
    for (int i = 0; i < hidden_size; i += blockDim.x * 8) {
        int pos = i + threadIdx.x * 8;
        if (pos + 7 < hidden_size) {
            // Load 8 FP16 elements (128-bit)
            uint4 chunk = *reinterpret_cast<const uint4*>(src + pos);
            *reinterpret_cast<uint4*>(dst + pos) = chunk;
        } else {
            // Tail handling
            for (int j = 0; j < 8 && pos + j < hidden_size; ++j) {
                dst[pos + j] = src[pos + j];
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