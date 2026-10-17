// qmatvec.h — host dispatch for matvec_q.hip (Phase 20 Step 3b).
//
// y[N] = W[N x K] x[K] where W is a GGUF tensor in its native block format, read in place from
// the pinned file mapping. Kernels are looked up by quant_types name: matvec_<qt_traits.name>.

#ifndef QMATVEC_H
#define QMATVEC_H

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load matvec_q.co (path to the code object). Returns 0 on success.
int qmatvec_init(const char *co_path);
void qmatvec_free(void);

// 1 if a matvec kernel exists for this ggml type id.
int qmatvec_supported(int type);

// Launch on the given stream (0 = default). W = first row, row_bytes = qt_row_size(type, K).
// All pointers must be device-accessible (pinned host memory on UMA). Returns hipSuccess on success.
hipError_t qmatvec_launch(int type, const void *W, size_t row_bytes, const float *x, float *y, int N, int K,
                          hipStream_t stream);

// ---------------------------------------------------------------- int8 path
// The activation vector is quantized once (qmatvec_quantize) into both formats ggml's CPU reference uses
// (Q8_0 for ROCmFP4, Q8_K for K-quants), then int8 matvecs use v_dot4. See matvec_q.hip. Types and shapes
// without an int8 kernel keep using qmatvec_launch on the float x.

typedef struct qmatvec_q8 {
    int8_t *q0;    // Q8_0 values [K]
    float *d0;     // Q8_0 scales [K / 32] (FP16-rounded, as ggml stores them)
    int8_t *qK;    // Q8_K values [K]
    float *dK;     // Q8_K scales [K / 256]
    int32_t *bsK;  // Q8_K sums of every 16 values [K / 16]
    int max_k;
} qmatvec_q8;

// The output pointers as kernels take them (Q8Out in q8_device.h): pass by value to kernels that quantize
// their own output. All NULL when q is NULL (the kernel then skips quantizing).
typedef struct qmatvec_q8_out {
    int8_t *q0;
    float *d0;
    int8_t *qK;
    float *dK;
    int32_t *bsK;
} qmatvec_q8_out;

static inline qmatvec_q8_out qmatvec_q8_outputs(const qmatvec_q8 *q) {
    qmatvec_q8_out o = {0, 0, 0, 0, 0};
    if (q) {
        o.q0 = q->q0;
        o.d0 = q->d0;
        o.qK = q->qK;
        o.dK = q->dK;
        o.bsK = q->bsK;
    }
    return o;
}

// Bytes of device-accessible memory a qmatvec_q8 for vectors of up to max_k values needs, and binding it
// to such memory (16-byte aligned). max_k must be a multiple of 256.
size_t qmatvec_q8_bytes(int max_k);
void qmatvec_q8_bind(qmatvec_q8 *q, void *mem, int max_k);

// 1 if an int8 kernel handles this type at this K (and the activation can be quantized: K % 256 == 0)
int qmatvec_q8_supported(int type, int K);

// Quantize x[K] into q (both formats). K % 256 == 0, K <= q->max_k.
hipError_t qmatvec_quantize(const float *x, int K, const qmatvec_q8 *q, hipStream_t stream);

// y = W x with x given by its quantized form q (from qmatvec_quantize of the same K)
hipError_t qmatvec_launch_q8(int type, const void *W, size_t row_bytes, const qmatvec_q8 *q, float *y, int N, int K,
                             hipStream_t stream);

// Two matvecs of the same type and K on the same input in one launch: y = W x (N rows), y2 = W2 x (N2 rows)
hipError_t qmatvec_launch_q8_pair(int type, const void *W, int N, float *y, const void *W2, int N2, float *y2,
                                  size_t row_bytes, const qmatvec_q8 *q, int K, hipStream_t stream);

// Batched: Y[t] = W X[t] for T tokens (prefill). q holds the T inputs quantized as one flattened [T * K]
// vector (qmatvec_quantize(X, T * K, q)); row t of the result goes to y + t * ldy. Same support as
// qmatvec_q8_supported; each weight block is decoded once per tile of 16 tokens.
hipError_t qmatvec_launch_q8_batch(int type, const void *W, size_t row_bytes, const qmatvec_q8 *q, float *y,
                                   long long ldy, int N, int K, int T, hipStream_t stream);

// Mixture-of-experts matvecs, with the expert choice read from GPU memory (no CPU round trip). W stacks every
// expert's N rows (expert e = rows e * N .. e * N + N - 1). For token t and slot j < n_used, expert
// e = ids[t * n_used + j] multiplies input row t (x_per_slot = 0) or row t * n_used + j (x_per_slot = 1) into
// row t * n_used + j of y (N values each). The int8 version reads the inputs from q (quantized as one flattened
// vector of all input rows); the float version reads x.
hipError_t qmatvec_launch_q8_id(int type, const void *W, size_t row_bytes, const int32_t *ids, int n_used,
                                int x_per_slot, const qmatvec_q8 *q, float *y, int N, int K, int T, hipStream_t stream);
hipError_t qmatvec_launch_id(int type, const void *W, size_t row_bytes, const int32_t *ids, int n_used, int x_per_slot,
                             const float *x, float *y, int N, int K, int T, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // QMATVEC_H
