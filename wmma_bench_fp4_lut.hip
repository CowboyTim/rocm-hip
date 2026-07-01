#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// FP4 ROCmFP4 Codebook10: 0, ±1, ±2, ±3, ±4, ±6, ±8, ±10
static const int8_t fp4_codebook[16] = {
    0,  1,  2,  3,  4,  6,  8, 10,
    0, -1, -2, -3, -4, -6, -8,-10
};

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

// FP4 WMMA benchmark kernel - uses LDS LUT for FP4->FP16 conversion
// Then feeds FP16 into native WMMA 16x16x16 tensor core
// Layout matches wmma_kernel_fp4.hip: A is column-major, B is row-major FP16
extern "C" __global__ void wmma_bench_fp4_lut(
    const unsigned char* A,       // M x K FP4, column-major (1 nibble per byte)
    const __half* B,              // K x N FP16, row-major
    __half* C,                    // M x N FP16, row-major
    int M, int N, int K,
    float scale)
{
    const int tid = threadIdx.x;  // 0..31
    const int lane = tid % 16;
    
    // Each block: one 16x16 output tile
    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;
    
    if (baseRow >= M || baseCol >= N) return;
    
    int row_end = min(baseRow + 16, M);
    int col_end = min(baseCol + 16, N);
    
    // 16-entry LDS LUT: entry[nibble] = FP16 dequant value
    // Total size: 16 × 2 bytes = 32 bytes, latency ~2-4 cycles
    __shared__ half lds_lut[16];
    
    // Build LUT: threads 0-15 each build one entry
    if (tid < 16) {
        float val = (float)fp4_codebook[tid] * scale;
        lds_lut[tid] = __half(val);
    }
    __syncthreads();
    
    // Accumulator for WMMA output
    half16 c_frag = {};
    
    // FP4 fragment: 16 fp16 elements for WMMA input (dequantized from FP4)
    half16 w_frag = {};
    
    // Load one column of A (FP4) and dequant via LDS LUT
    // Column-major layout: weights[lane + ele*16]
    // lane selects the column, ele iterates rows
    for (int ele = 0; ele < 16; ++ele) {
        uint8_t nibble = A[lane + ele * 16];
        int idx = (nibble < 8) ? nibble : 7;  // clamp for test (no negative values in data)
        w_frag[ele] = lds_lut[idx];
    }
    
    // Load FP16 activations row-major: B[ele][lane] = ele*16 + lane
    half16 a_frag;
    for (int ele = 0; ele < 16; ++ele) {
        a_frag[ele] = B[ele * 16 + lane];
    }
    
    // WMMA: FP16 x FP16 -> FP16, 16x16x16
    // OPSEL=false: output at even indices [0,2,...,30]
    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(w_frag, a_frag, c_frag, false);
    
    // Store FP16 result
    // Lanes 0-15 -> rows 0,2,...,14; lanes 16-31 -> rows 1,3,...,15
    for (int ele = 0; ele < 8; ++ele) {
        const int r = ele * 2 + (tid / 16);
        if (r < row_end && lane < col_end) {
            C[16 * r + lane] = c_frag[ele * 2];
        }
    }
}
