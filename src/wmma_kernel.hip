#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Wave Matrix Multiply Accumulate (WMMA) intrinsics for gfx1151
// Computes a 16x16 fp16 matrix multiplication: C = A * B
// Uses __builtin_amdgcn_wmma_f16_16x16x16_f16_w32 intrinsic

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void wmma_matmul(__half* a, __half* b, __half* c)
{
    const int lIdx = threadIdx.x;

    // Fragment registers: 16 fp16 elements each in packed VGPR format
    half16 a_frag;
    half16 b_frag;
    half16 c_frag = {};

    // RDNA 3 replicates 16x16 WMMA matrices across all 32 lanes.
    // Lanes 0-15 and 16-31 each compute the same result but with different
    // OPSEL output selection. OPSEL=false puts results at even indices.
    const int lane = lIdx % 16;

    // Load one column of matrix A (16 elements)
    for (int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * lane + ele];
    }

    // Load one row of matrix B (16 elements)
    for (int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * ele + lane];
    }

    // WMMA intrinsic: OPSEL=false
    // Output is unpacked: c_frag[0], c_frag[2], ... c_frag[30] (16 values)
    // On RDNA3, all 32 lanes participate but only lanes 0-15 produce valid output
    c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

    // Store results from unpacked c_frag output.
    // OPSEL=false places 16 results at even indices [0,2,4,...,30] of c_frag.
    // We need 32 threads to cover all 16 rows of the 16x16 result matrix:
    //   threads 0-15: write result rows 0,2,4,...,14  (r = 2*ele + 0)
    //   threads 16-31: write result rows 1,3,5,...,15 (r = 2*ele + 1)
    for (int ele = 0; ele < 8; ++ele)
    {
        const int r = ele * 2 + (lIdx / 16);
        c[16 * r + lane] = c_frag[ele * 2];
    }
}
