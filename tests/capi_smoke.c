/* capi_smoke.c -- the public C API through libinference.so only (Phase 22 step 3).
 *
 *   capi_smoke <model.gguf> [n_ctx]
 *
 * Plain C, linked against the shared library alone (no static archives, no kernel files: the code objects
 * are embedded), and meant to be run from any directory. Loads the model, reads its chat template, streams
 * one greedy chat turn with a stop string, and checks the stop reason and the text. This is what the Perl
 * XS module will do. */

#include "inference.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;

static void check(int ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

struct Collect {
    char text[4096];
    int len, calls;
};

static int on_token(void* user, int32_t token, const char* text, int len) {
    struct Collect* c = (struct Collect*)user;
    (void)token;
    if (c->len + len < (int)sizeof(c->text)) {
        memcpy(c->text + c->len, text, (size_t)len);
        c->len += len;
        c->text[c->len] = '\0';
    }
    ++c->calls;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_ctx]\n", argv[0]);
        return 2;
    }
    const int n_ctx = argc > 2 ? atoi(argv[2]) : 2048;
    inference_model_t* m = inference_load_model(argv[1]);
    if (!m) {
        fprintf(stderr, "load failed: %s\n", inference_get_last_error());
        return 1;
    }
    if (inference_kv_cache_init(m, n_ctx) != 0) {
        fprintf(stderr, "kv cache init failed: %s\n", inference_get_last_error());
        inference_free_model(m);
        return 1;
    }
    printf("capi_smoke: %s, n_ctx %d\n", argv[1], inference_n_ctx(m));
    check(inference_n_ctx(m) == n_ctx, "inference_n_ctx is the requested context length");

    const char* tmpl = inference_chat_template(m, NULL);
    check(tmpl != NULL && strstr(tmpl, "messages") != NULL, "inference_chat_template returns the GGUF's template");
    const int eos = inference_eos_id(m);
    char eos_text[64] = "";
    if (eos >= 0) inference_detokenize(m, &eos, 1, eos_text, sizeof(eos_text));
    printf("      eos %d \"%s\", eog: %d\n", eos, eos_text, eos >= 0 ? inference_token_is_eog(m, eos) : -1);
    check(eos >= 0 && inference_token_is_eog(m, eos), "EOS id and its text, and it ends generation");

    /* One chat turn, rendered by hand as the template would (rendering is the caller's job) */
    const char* prompt = "<|im_start|>user\nCount from 1 to 10, separated by commas.<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    int n = 0;
    int32_t* ids = inference_tokenize(m, prompt, &n);
    inference_gen_options o = inference_gen_default_options();
    o.sampler.temperature = 0.0f;
    o.max_new_tokens = 64;
    const char* stops[] = {"6"};
    o.stop_strings = stops;
    o.n_stop_strings = 1;
    o.reuse_prefix = 1;
    struct Collect c = {{0}, 0, 0};
    o.on_token = on_token;
    o.user = &c;
    int n_out = 0;
    inference_stop_reason reason = INFERENCE_STOP_NONE;
    const int rc = inference_generate_stream(m, ids, n, &o, NULL, 0, &n_out, &reason);
    inference_free_tokens(ids, n);
    printf("      reply \"%s\" (%d tokens, %d callbacks, stop reason %d)\n", c.text, n_out, c.calls, (int)reason);
    check(rc == 0 && reason == INFERENCE_STOP_STRING && strncmp(c.text, "1, 2, 3, 4, 5", 13) == 0 && !strchr(c.text, '6'),
          "greedy chat turn streams \"1, 2, 3, 4, 5\" and stops before the stop string \"6\"");

    inference_kv_cache_free(m);
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
