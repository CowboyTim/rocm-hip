// events_test -- the engine event hook (inference_set_event_cb, WORKLOG logging plan Phase 1) on a real model:
// the events' order and counts, and that the hook changes no output. Usage (from build/co, for the kernels):
//   events_test MODEL.gguf [DRAFT_HEAD.gguf]
// With a draft head, the draft events are checked on MODEL too.

#include "inference.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

struct Log {
    std::vector<inference_event> ev;
};

static int record(void* user, const inference_event* e) {
    static_cast<Log*>(user)->ev.push_back(*e);
    return 0;
}

static int count(const Log& l, int type) {
    int n = 0;
    for (const inference_event& e : l.ev) n += e.type == type;
    return n;
}

static std::vector<int32_t> tokens(inference_model_t* m, const std::string& s) {
    int n = 0;
    int32_t* t = inference_tokenize(m, s.c_str(), &n);
    std::vector<int32_t> v(t, t + n);
    free(t);
    return v;
}

// Loads a fresh model (so every run starts from the same state), with the KV cache for n_ctx positions
static inference_model_t* fresh(const char* path, int n_ctx, const char* draft) {
    inference_model_t* m = inference_load_model(path);
    if (!m) { fprintf(stderr, "load: %s\n", inference_get_last_error()); exit(1); }
    if (draft && inference_set_draft_model(m, draft) != 0) { fprintf(stderr, "draft: %s\n", inference_get_last_error()); exit(1); }
    if (inference_kv_cache_init(m, n_ctx) != 0) { fprintf(stderr, "kv: %s\n", inference_get_last_error()); exit(1); }
    return m;
}

// Greedy generation as the server does it (prefix reuse on); returns the generated tokens
static std::vector<int32_t> generate(inference_model_t* m, const std::vector<int32_t>& p, int max_new, int* rc) {
    inference_gen_options o = inference_gen_default_options();
    o.max_new_tokens = max_new;
    o.sampler.temperature = 0.0f;
    o.reuse_prefix = 1;
    std::vector<int32_t> out(max_new + 8);
    int n_out = 0;
    inference_stop_reason why;
    *rc = inference_generate_stream(m, p.data(), (int)p.size(), &o, out.data(), (int)out.size(), &n_out, &why);
    out.resize(n_out > 0 ? n_out : 0);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: events_test MODEL.gguf [DRAFT_HEAD.gguf]\n"); return 2; }
    const char* model = argv[1];
    const char* draft = argc > 2 ? argv[2] : nullptr;

    std::string base;
    for (int i = 0; i < 80; ++i) base += "The quick brown fox jumps over the lazy dog near the river bank. ";

    // 1. the events of one generation of a long prompt, and the same output without the hook
    printf("prefill events: one generation of a long prompt\n");
    inference_model_t* m = fresh(model, 4096, nullptr);
    const std::vector<int32_t> longp = tokens(m, base + "Question: what color is the fox?");
    Log a;
    inference_set_event_cb(m, record, &a);
    int rc = 0;
    const std::vector<int32_t> with_hook = generate(m, longp, 4, &rc);
    check(rc == 0, "generation succeeds with the hook");
    check(count(a, INFERENCE_EVT_PREFILL_BEGIN) == 1, "one PREFILL_BEGIN per generation");
    check(count(a, INFERENCE_EVT_PREFILL_END) == 1, "one PREFILL_END per generation");
    check(!a.ev.empty() && a.ev.front().type == INFERENCE_EVT_PREFILL_BEGIN, "PREFILL_BEGIN is the first event");
    int reused = -1, prefilled = -1;
    inference_last_prompt_stats(m, &reused, &prefilled);
    for (const inference_event& e : a.ev)
        if (e.type == INFERENCE_EVT_PREFILL_END) check(e.n0 == prefilled, "PREFILL_END n0 is the prefilled count");
    // progress: steps of 512 tokens, counts increasing and below the total, the total the prompt's length
    bool mono = true;
    int last = 0, total = -1, steps = 0;
    for (const inference_event& e : a.ev) {
        if (e.type != INFERENCE_EVT_PREFILL_PROGRESS) continue;
        mono = mono && e.n0 > last && e.n0 < e.n1;
        last = e.n0;
        total = e.n1;
        ++steps;
    }
    check(steps >= 1, "a long prompt reports PREFILL_PROGRESS");
    check(mono && steps > 0 && total == (int)longp.size(), "PREFILL_PROGRESS counts increase below the prompt's length");
    inference_free_model(m);

    m = fresh(model, 4096, nullptr);
    const std::vector<int32_t> no_hook = generate(m, longp, 4, &rc);
    check(rc == 0 && no_hook == with_hook, "the same tokens without the hook");
    printf("no events once the hook is cleared\n");
    Log b;
    const std::vector<int32_t> none = generate(m, tokens(m, base + "A second question."), 4, &rc);
    inference_set_event_cb(m, record, &b);
    inference_set_event_cb(m, nullptr, nullptr);
    generate(m, tokens(m, base + "A third question."), 4, &rc);
    check(b.ev.empty() && !none.empty(), "no events after the hook is cleared");
    inference_free_model(m);

    // 2. the context filling during decoding: CONTEXT_FULL once
    printf("context full\n");
    m = fresh(model, 256, nullptr);
    Log c;
    inference_set_event_cb(m, record, &c);
    std::vector<int32_t> small = tokens(m, base.substr(0, 1200));
    small.resize(250);  // a prompt that fits; decoding runs into the 256 positions
    generate(m, small, 64, &rc);
    check(count(c, INFERENCE_EVT_CONTEXT_FULL) == 1, "CONTEXT_FULL once when generation fills the context");
    inference_free_model(m);

    // 3. the draft events, with a draft head: one per verify pass, the accepted count within the proposed
    if (draft) {
        printf("draft events (head %s)\n", draft);
        m = fresh(model, 4096, draft);
        Log d;
        inference_set_event_cb(m, record, &d);
        const std::vector<int32_t> spec = generate(m, longp, 32, &rc);
        check(rc == 0, "speculative generation succeeds with the hook");
        const int passes = count(d, INFERENCE_EVT_DRAFT);
        check(passes > 0, "the draft events fire");
        bool ok = true;
        for (const inference_event& e : d.ev)
            if (e.type == INFERENCE_EVT_DRAFT) ok = ok && e.n0 >= 1 && e.n1 >= 0 && e.n1 <= e.n0 && e.ms >= 0.0;
        check(ok, "each draft event: 1 <= proposed, 0 <= accepted <= proposed");
        inference_free_model(m);

        m = fresh(model, 4096, draft);
        const std::vector<int32_t> spec_no_hook = generate(m, longp, 32, &rc);
        check(spec == spec_no_hook, "speculative output is the same with the hook");
        inference_free_model(m);
    }

    printf("%s (%d failed)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
