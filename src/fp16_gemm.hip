// fp16_gemm.hip -- Phase 3: general FP16 x FP16 -> FP32 WMMA GEMM over unified memory.
//
// The attention workhorse: scores = Q @ K^T and out = P @ V are activation x activation products
// (no quantization), so unlike the Phase-2 kernels both operands are raw FP16 in mapped buffers.
// C[MxN] is written as FP32 row-major.
//
// Layout contract: A is M x K row-major; B is K x N row-major; C is M x N row-major. All three
// live in hipHostRegister'ed pages (GPU reads/writes them in place).
//
// Tiling matches the CPU-/hardware-verified conventions of wmma_kernel.hip / fp4_matmul_wmma.hip:
//   - one 16x16 output tile per 32-thread block; grid (ceil(M/16), ceil(N/16))
//   - lane L in 0..15 supplies row L of the A tile / column L of the B tile (lanes 16-31 duplicate)
//   - K processed in 16-wide strips; OPSEL=false c_frag flushed into an FP32 shared accumulator
//     after every strip so FP16 rounding stays bounded by one strip's magnitude
//   - partial tiles/strips zero-padded at the edges (no OOB reads, no garbage rows)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half16 __attribute__((ext_vector_type(16)));

extern "C" __global__ void fp16_gemm(const __half* A, const __half* B, float* C, int M, int N, int K)
{
    const int lIdx = threadIdx.x;             // 0..31
    const int lane = lIdx % 16;               // unique row/column index within the tile

    int baseRow = blockIdx.x * 16;
    int baseCol = blockIdx.y * 16;

    if (baseRow >= M || baseCol >= N) return;

    const int aRow = baseRow + lane;          // this thread's A row (duplicated across t/16 halves)
    const int bCol = baseCol + lane;          // this thread's B column

    __shared__ float accum[16][16];
    for (int i = lIdx; i < 256; i += 32) accum[i / 16][i % 16] = 0.0f;
    __syncthreads();

    half16 c_frag = {};                       // per-strip WMMA accumulator fragment

    const int numStrips = (K + 15) / 16;

    for (int s = 0; s < numStrips; s++) {
        int kStart = s * 16;

        half16 a_frag, b_frag;

        #pragma unroll
        for (int ele = 0; ele < 16; ++ele) {
            int k = kStart + ele;
            bool aOK = (aRow < M && k < K);   // zero-pad partial rows and strips
            bool bOK = (bCol < N && k < K);

            a_frag[ele] = aOK ? A[(size_t)aRow * K + k] : __half(0.0f);
            b_frag[ele] = bOK ? B[(size_t)k * N + bCol] : __half(0.0f);
        }

        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);

        // Flush this strip's contribution into FP32 accumulation and reset the fragment.
        #pragma unroll
        for (int ele = 0; ele < 8; ++ele) {
            accum[ele * 2 + (lIdx / 16)][lane] += (_Float16)c_frag[ele * 2];
        }
        c_frag = {};
        __syncthreads();   // flush complete before the next strip touches accum
    }

    // Store final results: thread t owns cells {row = 2*ele + (t/16), col = t%16}.
    #pragma unroll
    for (int ele = 0; ele < 8; ++ele) {
        int r = ele * 2 + (lIdx / 16);
        int tr = baseRow + r;
        int tc = baseCol + lane;
        if (tr < M && tc < N) C[(size_t)tr * N + tc] = accum[r][lane];
    }
}
