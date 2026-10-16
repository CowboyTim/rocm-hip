// qwen_ops.cpp -- see qwen_ops.h.

#include "qwen_ops.h"

#include <cmath>
#include <vector>

void qwen_rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float inv = (float)(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

void qwen_rope_neox(float* v, int n_rot, int pos, float freq_base) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const double theta = pos * std::pow((double)freq_base, -2.0 * i / n_rot);
        const float c = (float)std::cos(theta), s = (float)std::sin(theta);
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

void qwen_attention(const float* q, const __half* K, const __half* V, int kv_stride, int n_kv,
                    int n_head, int n_head_kv, int head_dim, float scale, float* out) {
    const int grp = n_head / n_head_kv;
    std::vector<float> p(n_kv);
    for (int h = 0; h < n_head; ++h) {
        const float* qh = q + (size_t)h * head_dim;
        const int g = h / grp;
        float mx = -INFINITY;
        for (int t = 0; t < n_kv; ++t) {
            const __half* k = K + (size_t)t * kv_stride + (size_t)g * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; ++d) s += qh[d] * __half2float(k[d]);
            p[t] = s * scale;
            if (p[t] > mx) mx = p[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < n_kv; ++t) { p[t] = std::exp(p[t] - mx); sum += p[t]; }
        float* oh = out + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; ++d) oh[d] = 0.0f;
        for (int t = 0; t < n_kv; ++t) {
            const __half* v = V + (size_t)t * kv_stride + (size_t)g * head_dim;
            const float w = p[t] / sum;
            for (int d = 0; d < head_dim; ++d) oh[d] += w * __half2float(v[d]);
        }
    }
}
