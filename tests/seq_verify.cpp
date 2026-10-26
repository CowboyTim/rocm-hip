// seq_verify -- sequences (inference_seq_*) and step-wise generation (inference_stream_*).
//
//   seq_verify <model.gguf> [n_ctx]        (run from build/co)
//
// Two prompts are generated greedily, each alone on the default sequence, then on two new sequences stepped in
// turn (a, b, a, b, ...): the interleaved tokens must be exactly the ones generated alone (each sequence keeps its
// own KV cache, recurrent state and logits). Then a third turn continues sequence a with reuse_prefix (its history
// must still be there after b ran), and freeing / reusing a sequence works.

#include "inference.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    g_fail += !ok;
}

static std::vector<int32_t> tok(inference_model_t* m, const char* text) {
    int n = 0;
    int32_t* ids = inference_tokenize(m, text, &n);
    std::vector<int32_t> v(ids, ids + n);
    inference_free_tokens(ids, n);
    return v;
}

static inference_gen_options greedy(int n) {
    inference_gen_options o = inference_gen_default_options();
    o.sampler.temperature = 0.0f;
    o.max_new_tokens = n;
    o.ignore_eog = 1;
    o.reuse_prefix = 1;
    return o;
}

// The whole generation of one stream, stepped to the end
static std::vector<int32_t> run(inference_stream_t* st, std::string* text = nullptr) {
    std::vector<int32_t> out;
    for (;;) {
        int32_t t;
        const char* s;
        int n;
        inference_stop_reason r;
        const int rc = inference_stream_step(st, &t, &s, &n, &r);
        if (t >= 0) out.push_back(t);
        if (text) text->append(s, (size_t)n);
        if (rc <= 0) break;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    inference_model_t* m = inference_load_model(argv[1]);
    const int ctx = argc > 2 ? atoi(argv[2]) : 2048;  // (Gemma-4: 8192 has the sliding-window ring per sequence)
    if (!m || inference_kv_cache_init(m, ctx) != 0) { fprintf(stderr, "load: %s\n", inference_get_last_error()); return 1; }
    const std::vector<int32_t> pa = tok(m, "The capital of France is Paris. The capital of Germany is");
    const std::vector<int32_t> pb = tok(m, "Water boils at a temperature of");
    const int N = 24;
    const inference_gen_options o = greedy(N);

    // alone, on sequence 0 (each from an empty sequence, so that its prompt is prefilled in the same batch as on a
    // new sequence: prefill batches of other sizes may take kernels of another summation order)
    inference_reset(m);
    inference_stream_t* s = inference_stream_begin(m, pa.data(), (int)pa.size(), &o);
    std::string ta;
    const std::vector<int32_t> a0 = run(s, &ta);
    inference_stream_free(s);
    inference_reset(m);
    s = inference_stream_begin(m, pb.data(), (int)pb.size(), &o);
    const std::vector<int32_t> b0 = run(s);
    inference_stream_free(s);
    check((int)a0.size() == N && (int)b0.size() == N, "alone: both generate");

    // interleaved, on two new sequences
    const int qa = inference_seq_new(m), qb = inference_seq_new(m);
    check(qa > 0 && qb > 0 && qa != qb, "two new sequences");
    inference_seq_select(m, qa);
    inference_stream_t* sa = inference_stream_begin(m, pa.data(), (int)pa.size(), &o);
    inference_seq_select(m, qb);
    inference_stream_t* sb = inference_stream_begin(m, pb.data(), (int)pb.size(), &o);
    std::vector<int32_t> a1, b1;
    std::string ta1;
    bool ga = true, gb = true;
    while (ga || gb) {
        int32_t t;
        const char* x;
        int n;
        inference_stop_reason r;
        if (ga) {
            ga = inference_stream_step(sa, &t, &x, &n, &r) > 0;
            if (t >= 0) a1.push_back(t);
            ta1.append(x, (size_t)n);
        }
        if (gb) {
            gb = inference_stream_step(sb, &t, &x, &n, &r) > 0;
            if (t >= 0) b1.push_back(t);
        }
    }
    inference_stream_free(sa);
    inference_stream_free(sb);
    check(a1 == a0, "interleaved: sequence a's tokens = alone");
    check(b1 == b0, "interleaved: sequence b's tokens = alone");
    check(ta1 == ta, "interleaved: the same text");
    printf("  a: %s\n", ta.c_str());

    // a third turn on a: its history must be there (reused), the continuation as on sequence 0
    std::vector<int32_t> pa2 = pa;
    pa2.insert(pa2.end(), a0.begin(), a0.end());
    const std::vector<int32_t> more = tok(m, " And Italy?");
    pa2.insert(pa2.end(), more.begin(), more.end());
    // (on sequence 0, from a's first turn re-run: the same prefill batches as a's continuation below)
    inference_seq_select(m, 0);
    inference_reset(m);
    s = inference_stream_begin(m, pa.data(), (int)pa.size(), &o);
    run(s);
    inference_stream_free(s);
    s = inference_stream_begin(m, pa2.data(), (int)pa2.size(), &o);
    const std::vector<int32_t> c0 = run(s);
    inference_stream_free(s);
    inference_seq_select(m, qa);
    s = inference_stream_begin(m, pa2.data(), (int)pa2.size(), &o);
    int reused = 0, prefilled = 0;
    inference_last_prompt_stats(m, &reused, &prefilled);
    const std::vector<int32_t> c1 = run(s);
    inference_stream_free(s);
    char b[160];
    snprintf(b, sizeof b, "sequence a continues after b ran: reused %d, prefilled %d; the same tokens", reused, prefilled);
    check(c1 == c0 && reused > (int)pa.size() / 2, b);

    // free and reuse
    check(inference_seq_free(m, qb) == 0, "free a sequence");
    const int qc = inference_seq_new(m);
    check(qc == qb, "its number is reused");
    check(inference_seq_free(m, 0) != 0, "sequence 0 cannot be freed");
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
