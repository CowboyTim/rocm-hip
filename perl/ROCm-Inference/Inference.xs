/* Inference.xs -- thin XS mapping of the inference C library (src/inference.h) for ROCm::Inference.
 *
 * One Perl function per C function, the model handle first; no classes. Conventions:
 *   - errors croak with the library's message (inference_get_last_error);
 *   - text goes in and out as Perl character strings (UTF-8 in the library);
 *   - token lists are array references.
 * The handle is a blessed scalar reference only so that a model is freed when its last reference goes away
 * (DESTROY); free_model frees it explicitly. Everything else (chat templates, sessions, files) is plain Perl
 * in lib/ROCm/Inference.pm. */

#define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"

#include "inference.h"

#define HANDLE_CLASS "ROCm::Inference::Model"

static inference_model_t* handle_ptr(pTHX_ SV* sv, int allow_freed) {
    if (!SvROK(sv) || !sv_derived_from(sv, HANDLE_CLASS)) croak("ROCm::Inference: not a model handle");
    inference_model_t* m = INT2PTR(inference_model_t*, SvIV(SvRV(sv)));
    if (!m && !allow_freed) croak("ROCm::Inference: model handle already freed");
    return m;
}

static void croak_last(pTHX_ const char* what, int rc) {
    const char* e = inference_get_last_error();
    croak("ROCm::Inference::%s: %s (%d)", what, e && e[0] ? e : inference_strerror((inference_error_t)rc), rc);
}

/* Perl array ref of token ids -> malloc'ed int32 array (caller frees with Safefree) */
static int32_t* ids_from_av(pTHX_ SV* ref, int* n) {
    if (!SvROK(ref) || SvTYPE(SvRV(ref)) != SVt_PVAV) croak("ROCm::Inference: expected an array reference of token ids");
    AV* av = (AV*)SvRV(ref);
    const SSize_t len = av_len(av) + 1;
    int32_t* ids;
    Newx(ids, len > 0 ? len : 1, int32_t);
    for (SSize_t i = 0; i < len; ++i) {
        SV** e = av_fetch(av, i, 0);
        ids[i] = e ? (int32_t)SvIV(*e) : 0;
    }
    *n = (int)len;
    return ids;
}

static SV* text_sv(pTHX_ const char* s, STRLEN len) {
    SV* sv = newSVpvn(s, len);
    if (is_utf8_string((const U8*)s, len)) SvUTF8_on(sv);
    return sv;
}

/* on_token callback: calls the Perl sub with (token, text); a true return value stops generation. A die in the
 * sub stops it too and is rethrown after the C call returns. */
struct cb_ctx {
#ifdef PERL_IMPLICIT_CONTEXT
    PerlInterpreter* perl;
#endif
    SV* code;
    SV* error;
};

static int on_token_cb(void* user, int32_t token, const char* text, int len) {
    struct cb_ctx* c = (struct cb_ctx*)user;
#ifdef PERL_IMPLICIT_CONTEXT
    dTHXa(c->perl);
#endif
    dSP;
    int stop = 0;
    ENTER;
    SAVETMPS;
    PUSHMARK(SP);
    XPUSHs(sv_2mortal(newSViv(token)));
    XPUSHs(sv_2mortal(text_sv(aTHX_ text, (STRLEN)len)));
    PUTBACK;
    const int count = call_sv(c->code, G_SCALAR | G_EVAL);
    SPAGAIN;
    if (SvTRUE(ERRSV)) {
        c->error = newSVsv(ERRSV);
        stop = 1;
        if (count > 0) (void)POPs;
    } else if (count > 0) {
        stop = SvTRUE(POPs) ? 1 : 0;
    }
    PUTBACK;
    FREETMPS;
    LEAVE;
    return stop;
}

static SV* opt(pTHX_ HV* h, const char* key) {
    SV** v = h ? hv_fetch(h, key, (I32)strlen(key), 0) : NULL;
    return v && SvOK(*v) ? *v : NULL;
}

MODULE = ROCm::Inference    PACKAGE = ROCm::Inference

PROTOTYPES: DISABLE

SV*
load_model(path)
    const char* path
  CODE:
    inference_model_t* m = inference_load_model(path);
    if (!m) croak_last(aTHX_ "load_model", INFERENCE_ERR_MODEL_LOAD);
    RETVAL = sv_setref_pv(newSV(0), HANDLE_CLASS, (void*)m);
  OUTPUT:
    RETVAL

void
free_model(h)
    SV* h
  CODE:
    inference_model_t* m = handle_ptr(aTHX_ h, 1);
    if (m) {
        inference_free_model(m);  /* also frees the KV cache */
        sv_setiv(SvRV(h), 0);
    }

void
kv_cache_init(h, n_ctx)
    SV* h
    int n_ctx
  CODE:
    const int rc = inference_kv_cache_init(handle_ptr(aTHX_ h, 0), n_ctx);
    if (rc != 0) croak_last(aTHX_ "kv_cache_init", rc);

int
set_lm_head_q8(h, enable)
    SV* h
    int enable
  CODE:
    RETVAL = inference_set_lm_head_q8(handle_ptr(aTHX_ h, 0), enable);
    if (RETVAL < 0) croak_last(aTHX_ "set_lm_head_q8", RETVAL);
  OUTPUT:
    RETVAL

int
set_draft_model(h, path)
    SV* h
    SV* path
  CODE:
    RETVAL = inference_set_draft_model(handle_ptr(aTHX_ h, 0), SvOK(path) ? SvPV_nolen(path) : NULL);
    if (RETVAL < 0) croak_last(aTHX_ "set_draft_model", RETVAL);
  OUTPUT:
    RETVAL

int
set_prefill_batch(h, n)
    SV* h
    int n
  CODE:
    RETVAL = inference_set_prefill_batch(handle_ptr(aTHX_ h, 0), n);
    if (RETVAL < 0) croak_last(aTHX_ "set_prefill_batch", RETVAL);
  OUTPUT:
    RETVAL

int
set_draft_tokens(h, n)
    SV* h
    int n
  CODE:
    RETVAL = inference_set_draft_tokens(handle_ptr(aTHX_ h, 0), n);
    if (RETVAL < 0) croak_last(aTHX_ "set_draft_tokens", RETVAL);
  OUTPUT:
    RETVAL

SV*
tokenize(h, text)
    SV* h
    SV* text
  CODE:
    STRLEN len;
    const char* s = SvPVutf8(text, len);
    int n = 0;
    int32_t* ids = inference_tokenize(handle_ptr(aTHX_ h, 0), s, &n);
    if (!ids && len > 0) croak_last(aTHX_ "tokenize", -1);
    AV* av = newAV();
    av_extend(av, n);
    for (int i = 0; i < n; ++i) av_push(av, newSViv(ids[i]));
    if (ids) inference_free_tokens(ids, n);
    RETVAL = newRV_noinc((SV*)av);
  OUTPUT:
    RETVAL

SV*
detokenize(h, ids_ref)
    SV* h
    SV* ids_ref
  CODE:
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    int n = 0;
    int32_t* ids = ids_from_av(aTHX_ ids_ref, &n);
    const int len = inference_detokenize(m, ids, n, NULL, 0);
    if (len < 0) { Safefree(ids); croak_last(aTHX_ "detokenize", len); }
    char* buf;
    Newx(buf, len + 1, char);
    inference_detokenize(m, ids, n, buf, len + 1);
    RETVAL = text_sv(aTHX_ buf, (STRLEN)len);
    Safefree(buf);
    Safefree(ids);
  OUTPUT:
    RETVAL

SV*
chat_template(h, name = NULL)
    SV* h
    const char* name
  CODE:
    const char* t = inference_chat_template(handle_ptr(aTHX_ h, 0), name);
    RETVAL = t ? text_sv(aTHX_ t, strlen(t)) : &PL_sv_undef;
  OUTPUT:
    RETVAL

int
bos_id(h)
    SV* h
  CODE:
    RETVAL = inference_bos_id(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
eos_id(h)
    SV* h
  CODE:
    RETVAL = inference_eos_id(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
token_is_eog(h, token)
    SV* h
    int token
  CODE:
    RETVAL = inference_token_is_eog(handle_ptr(aTHX_ h, 0), token);
  OUTPUT:
    RETVAL

int
vocab_size(h)
    SV* h
  CODE:
    RETVAL = inference_vocab_size(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
n_ctx(h)
    SV* h
  CODE:
    RETVAL = inference_n_ctx(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
n_past(h)
    SV* h
  CODE:
    RETVAL = inference_n_past(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

void
reset(h)
    SV* h
  CODE:
    inference_reset(handle_ptr(aTHX_ h, 0));

void
prefill(h, ids_ref)
    SV* h
    SV* ids_ref
  CODE:
    int n = 0;
    int32_t* ids = ids_from_av(aTHX_ ids_ref, &n);
    const int rc = inference_prefill(handle_ptr(aTHX_ h, 0), ids, n);
    Safefree(ids);
    if (rc != 0) croak_last(aTHX_ "prefill", rc);

int
decode_token(h, token)
    SV* h
    int token
  CODE:
    const int rc = inference_decode_token(handle_ptr(aTHX_ h, 0), token, &RETVAL);
    if (rc != 0) croak_last(aTHX_ "decode_token", rc);
  OUTPUT:
    RETVAL

SV*
get_logits(h)
    SV* h
  CODE:
    /* The last token's logits as packed native floats: unpack("f*", $logits) */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    const int v = inference_vocab_size(m);
    int* ids;
    Newx(ids, v, int);
    RETVAL = newSV((STRLEN)v * sizeof(float) + 1);
    SvPOK_only(RETVAL);
    const int rc = inference_get_logits(m, ids, (float*)SvPVX(RETVAL));
    Safefree(ids);
    if (rc != 0) { SvREFCNT_dec(RETVAL); croak_last(aTHX_ "get_logits", rc); }
    SvCUR_set(RETVAL, (STRLEN)v * sizeof(float));
  OUTPUT:
    RETVAL

void
generate(h, prompt_ref, opts_ref = NULL)
    SV* h
    SV* prompt_ref
    SV* opts_ref
  PPCODE:
    /* Returns (\@token_ids, $stop_reason). Options (all optional): max_new_tokens, temperature, top_k, top_p,
     * min_p, repeat_penalty, repeat_last_n, seed, stop => [strings], ignore_eog, reuse_prefix, checkpoint_tail,
     * on_token => sub { my ($token, $text) = @_; return $stop } */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    HV* o = NULL;
    if (opts_ref && SvOK(opts_ref)) {
        if (!SvROK(opts_ref) || SvTYPE(SvRV(opts_ref)) != SVt_PVHV) croak("ROCm::Inference::generate: options must be a hash reference");
        o = (HV*)SvRV(opts_ref);
    }
    inference_gen_options g = inference_gen_default_options();
    SV* v;
    if ((v = opt(aTHX_ o, "max_new_tokens"))) g.max_new_tokens = (int)SvIV(v);
    if ((v = opt(aTHX_ o, "temperature"))) g.sampler.temperature = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "top_k"))) g.sampler.top_k = (int)SvIV(v);
    if ((v = opt(aTHX_ o, "top_p"))) g.sampler.top_p = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "min_p"))) g.sampler.min_p = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "repeat_penalty"))) g.sampler.repeat_penalty = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "repeat_last_n"))) g.sampler.repeat_last_n = (int)SvIV(v);
    if ((v = opt(aTHX_ o, "seed"))) g.sampler.seed = (uint64_t)SvUV(v);
    if ((v = opt(aTHX_ o, "ignore_eog"))) g.ignore_eog = SvTRUE(v) ? 1 : 0;
    if ((v = opt(aTHX_ o, "reuse_prefix"))) g.reuse_prefix = SvTRUE(v) ? 1 : 0;
    if ((v = opt(aTHX_ o, "checkpoint_tail"))) g.checkpoint_tail = (int)SvIV(v);
    const char** stops = NULL;
    if ((v = opt(aTHX_ o, "stop"))) {
        if (!SvROK(v) || SvTYPE(SvRV(v)) != SVt_PVAV) croak("ROCm::Inference::generate: stop must be an array reference");
        AV* av = (AV*)SvRV(v);
        const SSize_t ns = av_len(av) + 1;
        Newx(stops, ns > 0 ? ns : 1, const char*);
        for (SSize_t i = 0; i < ns; ++i) {
            SV** e = av_fetch(av, i, 0);
            stops[i] = e && SvOK(*e) ? SvPVutf8_nolen(*e) : "";
        }
        g.stop_strings = stops;
        g.n_stop_strings = (int)ns;
    }
    struct cb_ctx cb;
#ifdef PERL_IMPLICIT_CONTEXT
    cb.perl = aTHX;
#endif
    cb.code = NULL;
    cb.error = NULL;
    if ((v = opt(aTHX_ o, "on_token"))) {
        if (!SvROK(v) || SvTYPE(SvRV(v)) != SVt_PVCV) croak("ROCm::Inference::generate: on_token must be a code reference");
        cb.code = v;
        g.on_token = on_token_cb;
        g.user = &cb;
    }
    int n = 0;
    int32_t* prompt = ids_from_av(aTHX_ prompt_ref, &n);
    const int cap = g.max_new_tokens > 0 ? g.max_new_tokens : 1;
    int32_t* out;
    Newx(out, cap, int32_t);
    int n_out = 0;
    inference_stop_reason reason = INFERENCE_STOP_NONE;
    const int rc = inference_generate_stream(m, prompt, n, &g, out, cap, &n_out, &reason);
    Safefree(prompt);
    Safefree(stops);
    if (cb.error) {
        Safefree(out);
        croak_sv(sv_2mortal(cb.error));
    }
    if (rc != 0) {
        Safefree(out);
        croak_last(aTHX_ "generate", rc);
    }
    AV* av = newAV();
    for (int i = 0; i < n_out && i < cap; ++i) av_push(av, newSViv(out[i]));
    Safefree(out);
    EXTEND(SP, 2);
    PUSHs(sv_2mortal(newRV_noinc((SV*)av)));
    PUSHs(sv_2mortal(newSViv((IV)reason)));

void
last_prompt_stats(h)
    SV* h
  PPCODE:
    /* (tokens reused, tokens prefilled) by the last generate */
    int reused = 0, prefilled = 0;
    inference_last_prompt_stats(handle_ptr(aTHX_ h, 0), &reused, &prefilled);
    EXTEND(SP, 2);
    PUSHs(sv_2mortal(newSViv(reused)));
    PUSHs(sv_2mortal(newSViv(prefilled)));

void
store_open(h, dir, snapshot_every = 4)
    SV* h
    const char* dir
    int snapshot_every
  CODE:
    const int rc = inference_store_open(handle_ptr(aTHX_ h, 0), dir, snapshot_every);
    if (rc != 0) croak_last(aTHX_ "store_open", rc);

void
store_close(h)
    SV* h
  CODE:
    inference_store_close(handle_ptr(aTHX_ h, 0));

SV*
store_stats(h)
    SV* h
  CODE:
    inference_store_stats st;
    if (inference_store_get_stats(handle_ptr(aTHX_ h, 0), &st) != 0) croak("ROCm::Inference::store_stats failed");
    HV* hv = newHV();
    (void)hv_stores(hv, "blocks_published", newSViv((IV)st.blocks_published));
    (void)hv_stores(hv, "blocks_attached", newSViv((IV)st.blocks_attached));
    (void)hv_stores(hv, "snapshots_saved", newSViv((IV)st.snapshots_saved));
    (void)hv_stores(hv, "snapshots_restored", newSViv((IV)st.snapshots_restored));
    (void)hv_stores(hv, "tokens_from_store", newSViv((IV)st.tokens_from_store));
    (void)hv_stores(hv, "block_tokens", newSViv(st.block_tokens));
    (void)hv_stores(hv, "block_bytes", newSViv((IV)st.block_bytes));
    (void)hv_stores(hv, "snapshot_bytes", newSViv((IV)st.snapshot_bytes));
    RETVAL = newRV_noinc((SV*)hv);
  OUTPUT:
    RETVAL

void
store_block_names(h, ids_ref)
    SV* h
    SV* ids_ref
  PPCODE:
    /* The store files (<hash>.kvb) of the token list's full blocks, in order */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    int n = 0;
    int32_t* ids = ids_from_av(aTHX_ ids_ref, &n);
    const int len = inference_store_block_names(m, ids, n, NULL, 0);
    if (len < 0) { Safefree(ids); croak("ROCm::Inference::store_block_names: no store open"); }
    char* buf;
    Newx(buf, len + 1, char);
    inference_store_block_names(m, ids, n, buf, len + 1);
    Safefree(ids);
    for (char *p = buf, *e; *p; p = e + 1) {
        e = strchr(p, '\n');
        if (!e) break;
        XPUSHs(sv_2mortal(newSVpvn(p, (STRLEN)(e - p))));
    }
    Safefree(buf);

SV*
last_error()
  CODE:
    const char* e = inference_get_last_error();
    RETVAL = e && e[0] ? newSVpv(e, 0) : &PL_sv_undef;
  OUTPUT:
    RETVAL

int
_constant_stop(which)
    int which
  CODE:
    /* stop reasons, for the constants in Inference.pm */
    switch (which) {
    case 0: RETVAL = INFERENCE_STOP_NONE; break;
    case 1: RETVAL = INFERENCE_STOP_EOG; break;
    case 2: RETVAL = INFERENCE_STOP_MAX_TOKENS; break;
    case 3: RETVAL = INFERENCE_STOP_STRING; break;
    case 4: RETVAL = INFERENCE_STOP_CALLBACK; break;
    default: RETVAL = INFERENCE_STOP_CONTEXT_FULL; break;
    }
  OUTPUT:
    RETVAL

MODULE = ROCm::Inference    PACKAGE = ROCm::Inference::Model

void
DESTROY(h)
    SV* h
  CODE:
    inference_model_t* m = handle_ptr(aTHX_ h, 1);
    if (m) {
        inference_free_model(m);  /* also frees the KV cache */
        sv_setiv(SvRV(h), 0);
    }
