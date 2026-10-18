// deltanet.h -- host launchers for deltanet.hip (Gated DeltaNet token mixer on the GPU, one token).
//
// All pointers must be device-accessible (pinned host memory on UMA). Both launches go on `stream`
// and need no sync between them; see deltanet.hip for the math and the state layout.

#ifndef DELTANET_H
#define DELTANET_H

#include <hip/hip_runtime.h>

#include "qmatvec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DELTANET_HEAD_SIZE 128  // the kernels are compiled for ssm.state_size = 128

// Load deltanet.co. Returns 0 on success.
int deltanet_init(const char* co_path);
void deltanet_free(void);

// Both take T consecutive tokens (rows below are per token) and process them in order. Each queues two
// launches (deltanet_pre one when T == 1).
//
// conv1d + state update + SiLU over conv_dim = (2 * n_k + n_v) * 128 channels of [q | k | v], L2 norm
// of the q and k heads, and the per-V-head decay gate and beta. conv_state: [conv_dim][d_conv - 1];
// conv_w: [conv_dim][d_conv]; qkv, conv_out: [T][conv_dim]; alpha, beta_in, g_out, beta_out: [T][n_v].
hipError_t deltanet_pre(const float* qkv, float* conv_state, const float* conv_w, int d_conv, float* conv_out,
                        int n_k, int n_v, const float* alpha, const float* beta_in, const float* dt_bias,
                        const float* ssm_a, float* g_out, float* beta_out, float eps, int T, hipStream_t stream);

// Where element S[i][j] (row i, column j) of V head h's 128 x 128 delta-rule state lives in the state buffer
// (n_v * 128 * 128 floats). The layout follows deltanet_rule's threads: workgroup (h, j / 16) has thread
// t = (j % 16) * 8 + i / 16 hold rows i / 16 * 16 .. + 15 of column j, and its r-th value is at
// ((h * 8 + j / 16) * 16 + r) * 128 + t, so each of the wave's state loads is 128 contiguous bytes.
// (8 lanes per column, 16 columns per workgroup: RULE_RP and RULE_CW in deltanet.hip.)
static inline size_t deltanet_state_index(int h, int i, int j) {
    return ((size_t)(h * 8 + j / 16) * 16 + (size_t)(i % 16)) * 128 + (size_t)((j % 16) * 8 + i / 16);
}

// Delta rule + gated RMSNorm for every V head. state: [n_v][128 * 128], laid out by deltanet_state_index.
// conv_out from deltanet_pre; z, out: [T][n_v * 128]; norm_w: ssm_norm [128]. T * n_v * 128 must be a
// multiple of 256. The output gate is SiLU(z) (qwen35), or sigmoid(z) with sigmoid_gate (qwen4exp). q8 != NULL:
// also quantize out for the int8 matvecs (all T rows, as one flattened vector).
hipError_t deltanet_rule(float* state, const float* conv_out, const float* g, const float* beta, const float* z,
                         const float* norm_w, float* out, int n_k, int n_v, float eps, int T, int sigmoid_gate,
                         const qmatvec_q8* q8, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // DELTANET_H
