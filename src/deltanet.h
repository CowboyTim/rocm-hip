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

// Delta rule + gated RMSNorm for every V head. state: [n_v][128][128], S[i][j] at i * 128 + j.
// conv_out from deltanet_pre; z, out: [T][n_v * 128]; norm_w: ssm_norm [128]. T * n_v * 128 must be a
// multiple of 256. q8 != NULL: also quantize out for the int8 matvecs (all T rows, as one flattened vector).
hipError_t deltanet_rule(float* state, const float* conv_out, const float* g, const float* beta, const float* z,
                         const float* norm_w, float* out, int n_k, int n_v, float eps, int T, const qmatvec_q8* q8,
                         hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // DELTANET_H
