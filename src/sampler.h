// sampler.h -- next-token sampling over a logits vector (CPU).
//
// Chain, in llama.cpp's default order: repetition penalty -> top-k -> top-p -> min-p -> temperature ->
// draw from the remaining distribution. temperature <= 0 is greedy (argmax after the penalty), so it
// reproduces the reference's greedy output. The logits are only read; the caller can pass the engine's
// pinned logits buffer directly (qwen_model_logits), with no copy.

#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sampler_params {
    float temperature;     // <= 0: greedy
    int top_k;             // keep the k most likely tokens; <= 0: off
    float top_p;           // keep the smallest set with cumulative probability >= top_p; >= 1: off
    float min_p;           // keep tokens with p >= min_p * p_max; <= 0: off
    float repeat_penalty;  // divide positive logits / multiply negative ones of recent tokens; 1: off
    int repeat_last_n;     // how many accepted tokens the penalty looks back over
    uint64_t seed;         // RNG seed; the same seed and inputs give the same tokens
} sampler_params;

typedef struct sampler sampler_t;

// llama.cpp's defaults: temperature 0.8, top_k 40, top_p 0.95, min_p 0.05, repeat_penalty 1 (off),
// repeat_last_n 64, seed 0
sampler_params sampler_default_params(void);

sampler_t* sampler_create(const sampler_params* params);
void sampler_free(sampler_t* s);

// Pick the next token from logits[n_vocab]. Does not record it; call sampler_accept for that.
int sampler_sample(sampler_t* s, const float* logits, int n_vocab);

// Record a token (prompt or generated) for the repetition penalty
void sampler_accept(sampler_t* s, int token);

// Forget the token history and re-seed the RNG (new sequence)
void sampler_reset(sampler_t* s);

#ifdef __cplusplus
}
#endif

#endif // SAMPLER_H
