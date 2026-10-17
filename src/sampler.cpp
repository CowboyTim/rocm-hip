// sampler.cpp -- see sampler.h.

#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <cstdint>
#include <random>
#include <vector>

namespace {

struct Cand {
    int id;
    float logit;
    float p;
};

} // namespace

struct sampler {
    sampler_params prm;
    std::mt19937_64 rng;
    std::deque<int> history;
    std::vector<Cand> cand;
    std::vector<uint8_t> marked;   // [n_vocab]: token is in the penalty window; only history entries are set

    explicit sampler(const sampler_params& p) : prm(p), rng(p.seed) {}

    // Uniform [0, 1) from the 53 high bits, so the draw does not depend on the standard library
    double uniform() { return (double)(rng() >> 11) * 0x1.0p-53; }

    bool pen = false;              // repetition penalty active for the current sample() call

    // Logit of token i after the repetition penalty
    float val(const float* logits, int i) const {
        const float l = logits[i];
        if (!pen || !marked[i]) return l;
        return l > 0.0f ? l / prm.repeat_penalty : l * prm.repeat_penalty;
    }

    int sample(const float* logits, int n) {
        pen = prm.repeat_penalty != 1.0f && !history.empty();
        if (pen) {
            marked.resize(n, 0);
            for (int t : history)
                if (t >= 0 && t < n) marked[t] = 1;
        }
        const int tok = pick(logits, n);
        if (pen)
            for (int t : history)
                if (t >= 0 && t < n) marked[t] = 0;
        return tok;
    }

    int pick(const float* logits, int n) {
        // Greedy: argmax after the penalty (lowest id on ties)
        if (prm.temperature <= 0.0f || prm.top_k == 1) {
            int best = 0;
            float bv = val(logits, 0);
            for (int i = 1; i < n; ++i) {
                const float v = val(logits, i);
                if (v > bv) { bv = v; best = i; }
            }
            return best;
        }

        auto by_logit = [](const Cand& a, const Cand& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); };
        cand.clear();
        bool sorted = false;
        double z_all = 0.0;  // softmax normalizer over the whole vocabulary, when top-p needs it (0: use cand)

        if (prm.top_k > 0) {
            // top-k in one pass: a min-heap of the k best (ids ascend, so ties keep the lower id)
            const int k = std::min(prm.top_k, n);
            auto worse = [](const Cand& a, const Cand& b) { return a.logit > b.logit; };  // heap top = worst kept
            for (int i = 0; i < n; ++i) {
                const float v = val(logits, i);
                if ((int)cand.size() < k) {
                    cand.push_back({i, v, 0.0f});
                    std::push_heap(cand.begin(), cand.end(), worse);
                } else if (v > cand.front().logit) {
                    std::pop_heap(cand.begin(), cand.end(), worse);
                    cand.back() = {i, v, 0.0f};
                    std::push_heap(cand.begin(), cand.end(), worse);
                }
            }
            std::sort(cand.begin(), cand.end(), by_logit);
            sorted = true;
        } else {
            float mx = -INFINITY;
            for (int i = 0; i < n; ++i) mx = std::max(mx, val(logits, i));
            if (prm.top_p < 1.0f) {
                // top-p without top-k (llama.cpp's bucket method): one pass puts each token's probability
                // mass into a bucket by its distance below the max logit. The nucleus lies in the buckets
                // up to where the cumulative mass reaches top_p; only those tokens are collected and sorted.
                // Tokens more than RANGE below the max (p < e^-RANGE of the top) share the last bucket.
                const int NB = 1024;
                const float RANGE = 40.0f, scale = NB / RANGE;
                double mass[NB] = {};
                for (int i = 0; i < n; ++i) {
                    const float d = mx - val(logits, i);
                    const double e = std::exp(-(double)d);
                    z_all += e;
                    mass[std::min(NB - 1, (int)(d * scale))] += e;
                }
                int last = NB - 1;
                double cum = 0.0;
                for (int bkt = 0; bkt < NB; ++bkt) {
                    cum += mass[bkt];
                    if (cum >= prm.top_p * z_all) { last = bkt; break; }
                }
                for (int i = 0; i < n; ++i) {
                    const float v = val(logits, i);
                    if (std::min(NB - 1, (int)((mx - v) * scale)) <= last) cand.push_back({i, v, 0.0f});
                }
                std::sort(cand.begin(), cand.end(), by_logit);
                sorted = true;
            } else {
                // No ordering needed: min-p and the draw work on any order. Same e^-40 floor as above.
                const float floor_logit = prm.min_p > 0.0f ? mx + std::log(prm.min_p) : mx - 40.0f;
                for (int i = 0; i < n; ++i) {
                    const float v = val(logits, i);
                    if (v >= floor_logit) cand.push_back({i, v, 0.0f});
                }
            }
        }

        // top-p: smallest prefix (by logit) whose probability reaches top_p (at least one token).
        // Probabilities are over the whole vocabulary when z_all is known, else over the candidates.
        if (prm.top_p < 1.0f && sorted) {
            const float mx = cand.front().logit;
            double z = z_all;
            if (z == 0.0)
                for (const Cand& c : cand) z += std::exp((double)c.logit - mx);
            double cum = 0.0;
            size_t keep = cand.size();
            for (size_t i = 0; i < cand.size(); ++i) {
                cum += std::exp((double)cand[i].logit - mx);
                if (cum >= prm.top_p * z) { keep = i + 1; break; }
            }
            cand.resize(keep);
        }

        // min-p: p >= min_p * p_max, i.e. logit >= max_logit + log(min_p)
        if (prm.min_p > 0.0f) {
            float mx = -INFINITY;
            for (const Cand& c : cand) mx = std::max(mx, c.logit);
            const float cut = mx + std::log(prm.min_p);
            cand.erase(std::remove_if(cand.begin(), cand.end(), [&](const Cand& c) { return c.logit < cut; }), cand.end());
        }

        // Temperature, then draw (in candidate order; sorted or id order, both deterministic)
        float mx = -INFINITY;
        for (const Cand& c : cand) mx = std::max(mx, c.logit);
        double z = 0.0;
        for (Cand& c : cand) {
            c.p = (float)std::exp((double)(c.logit - mx) / prm.temperature);
            z += c.p;
        }
        const double r = uniform() * z;
        double cum = 0.0;
        for (const Cand& c : cand) {
            cum += c.p;
            if (r < cum) return c.id;
        }
        return cand.back().id;
    }

    void accept(int t) {
        if (prm.repeat_last_n <= 0) return;
        history.push_back(t);
        while ((int)history.size() > prm.repeat_last_n) history.pop_front();
    }
};

extern "C" {

sampler_params sampler_default_params(void) {
    sampler_params p;
    p.temperature = 0.8f;
    p.top_k = 40;
    p.top_p = 0.95f;
    p.min_p = 0.05f;
    p.repeat_penalty = 1.0f;
    p.repeat_last_n = 64;
    p.seed = 0;
    return p;
}

sampler_t* sampler_create(const sampler_params* params) {
    return new sampler(params ? *params : sampler_default_params());
}

void sampler_free(sampler_t* s) { delete s; }

int sampler_sample(sampler_t* s, const float* logits, int n_vocab) {
    if (!s || !logits || n_vocab <= 0) return -1;
    return s->sample(logits, n_vocab);
}

void sampler_accept(sampler_t* s, int token) {
    if (s) s->accept(token);
}

void sampler_reset(sampler_t* s) {
    if (!s) return;
    s->history.clear();
    s->rng.seed(s->prm.seed);
}

} // extern "C"
