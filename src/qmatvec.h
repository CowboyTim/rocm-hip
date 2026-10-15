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

#ifdef __cplusplus
}
#endif

#endif // QMATVEC_H
