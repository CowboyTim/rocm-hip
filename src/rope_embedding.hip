// rope_embedding.hip -- Phase 3: rotary position embedding (rotate-half scheme) on unified memory.
//
//   For each row r and i in [0, half_dim):
//     y[r][i]      = x[r][i]      * ctab[r][i] - x[r][i+H] * stab[r][i]
//     y[r][i+H]    = x[r][i+H]    * ctab[r][i] + x[r][i]   * stab[r][i]
//
// where H = half_dim = D/2 and ctab/stab are FP32 cosine/sine tables PRECOMPUTED ON THE HOST into
// mapped buffers from CPU-provided position IDs (matches how production engines build the RoPE table;
// keeps this kernel pure algebra with no device-libm divergence). Positions arrive as int32 rows in a
// CPU-mapped buffer per the Phase-3 design ("position IDs loaded from CPU-mapped buffer").
//
// x is [n_rows x D] FP16 (rows are token-head pairs flattened), y the same shape. One 32-thread block
// per row; arbitrary even D works (threads beyond half_dim idle on small widths).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void rope_rotate_half(const __half* x, const float* ctab, const float* stab,
                                           __half* y, int n_rows, int half_dim)
{
    const int r = blockIdx.x;                       // row (token-head pair) index
    if (r >= n_rows) return;                        // uniform per block

    const __half* xr = x + (size_t)r * half_dim * 2;
    __half* yr      = y + (size_t)r * half_dim * 2;
    const float* cr = ctab + (size_t)r * half_dim;
    const float* sr = stab + (size_t)r * half_dim;

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        float lo = (float)xr[i];                    // first half of the pair
        float hi = (float)xr[i + half_dim];         // second half

        float c = cr[i], s = sr[i];
        yr[i]            = (__half)(lo * c - hi * s);
        yr[i + half_dim] = (__half)(hi * c + lo * s);
    }
}
