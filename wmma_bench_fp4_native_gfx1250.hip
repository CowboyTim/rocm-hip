#include <hip/hip_runtime.h>

// Native FP4 WMMA kernel for gfx1250+ (RDNA4 / AMD AI engines)
// Uses __builtin_amdgcn_wmma_f32_32x16x128_f4 intrinsic directly
//
// Fragment type signatures:
//   A (int16):  16 x int32 vector = 64 bytes = 128 FP4 nibbles packed
//   B (int8):   8 x int32 vector  = 32 bytes = 64 FP4 nibbles packed  
//   C (float16): 16 x float32 vector = 64 bytes = 32 FP32 output elements
//
// The intrinsic computes: C_32x16 = A_fp4_32x128 x B_fp4_16x128
// FP4 dequantization uses ROCmFP4 Codebook10 (E2M1-derived)
//
// Type signature from LLVM intrinsic:
//   __builtin_amdgcn_wmma_f32_32x16x128_f4(
//       int16 A,     // 16 x int32 = 64 bytes
//       int8 B,      // 8 x int32  = 32 bytes
//       uint16_t C_mod,  // 0 = no modifier
//       float16 C    // 16 x float32 = 64 bytes, matches A type for output
//   ) -> float16

extern "C" __global__ void wmma_bench_fp4_native_gfx1250(
    const unsigned char* A,       // M x K FP4, row-major
    const unsigned char* B,       // K x N FP4, row-major
    float* C,                     // M x N FP32, row-major
    int M, int N, int K,
    float scale)
{
    const int lane = threadIdx.x;  // 0..31

    // Each block: one 32x16 output tile
    int baseRow = blockIdx.x * 32;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    int row_end = min(baseRow + 32, M);
    int col_end = min(baseCol + 16, N);

    int realRow = baseRow + lane;
    if (realRow >= row_end) return;

    // Shared memory for K-strip accumulation (one value per thread = 32 values)
    __shared__ float accum[32];

    // Initialize: each thread accumulates for its output row
    int rowInTile = realRow - baseRow;
    accum[rowInTile] = 0.0f;
    __syncthreads();

    // Loop over K strips (each strip = 128 FP4 elements)
    // The FP4 WMMA 32x16x128 processes 128 K-dimension elements per call
    const int numStrips = (K + 127) / 128;

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * 128;

        // Pack FP4 data from A row into int16 fragment (16 x int32)
        // A[realRow][kStart..kStart+127] - 128 FP4 nibbles
        // Each int32 holds 8 nibbles (4 bytes * 8 bits / 4 bits per nibble = 8 nibbles)
        // 128 nibbles / 8 = 16 int32 values -> int16 vector

        using A_frag = int __attribute__((ext_vector_type(16)));
        A_frag a_frag = {};
        
        for (int i = 0; i < 128 && (kStart + i) < K; i++) {
            int byte_idx = i / 2;           // 0..63 -> maps to 16 int32 slots (4 bytes each)
            int nibble_pos = (i % 2) * 4;   // 0 or 4 (lower or upper nibble in byte)
            
            uint8_t raw = A[realRow * K + kStart + i];
            uint8_t nibble = (i % 2 == 0) ? (raw & 0x0F) : ((raw >> 4) & 0x0F);
            
            a_frag[byte_idx / 4] |= (int)nibble << (nibble_pos + (byte_idx % 4) * 8);
        }

        // Pack FP4 data from B column into int8 fragment (8 x int32)
        // B[kStart..kStart+63][baseCol + col_offset] - first 64 FP4 nibbles of column
        // The intrinsic processes all 128 K elements internally
        
        using B_frag = int __attribute__((ext_vector_type(8)));
        B_frag b_frag = {};

        int col_offset = baseCol;
        for (int i = 0; i < 64 && (kStart + i) < K; i++) {
            int byte_idx = i / 2;
            int nibble_pos = (i % 2) * 4;
            
            uint8_t raw = B[(kStart + i) * N + col_offset];
            uint8_t nibble = (i % 2 == 0) ? (raw & 0x0F) : ((raw >> 4) & 0x0F);
            
            b_frag[byte_idx / 4] |= (int)nibble << (nibble_pos + (byte_idx % 4) * 8);
        }

        // Call native FP4 WMMA intrinsic
        // __builtin_amdgcn_wmma_f32_32x16x128_f4(a_frag, b_frag, c_mod, c_frag)
        // 
        // a_frag: A[32x128] FP4 packed as int16 (16 x int32)
        // b_frag: B[16x128] FP4 packed as int8 (8 x int32)
        // c_mod:  0 = no modifier
        // c_frag: A[32x16] FP32 accumulator as float16 (16 x float32)
        //
        // Returns: C[32x16] FP32 matrix = A_fp4 x B_fp4 (with hardware FP4 dequant)
        
        using C_frag = float __attribute__((ext_vector_type(16)));
        C_frag c_frag = {};

        c_frag = __builtin_amdgcn_wmma_f32_32x16x128_f4(a_frag, b_frag, (uint16_t)0, c_frag);

        // Extract result for our output row
        // c_frag has 16 float32 elements (64 bytes)
        // c_frag[i] corresponds to output row i of the 32-row tile
        // Our thread handles row (realRow - baseRow) in the 32x16 output tile
        
        float val = c_frag[rowInTile] * scale;

        atomicAdd(&accum[rowInTile], val);

        __syncthreads();
    }

    // Write result to global memory
    C[realRow * N + baseCol] = accum[rowInTile];
}
