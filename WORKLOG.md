# WORKLOG: server logging through an engine event hook (PLAN.md item 3)

Plan as of 2026-11-06, grounded in the code as of c234f88. Phases are dependency-ordered:
the C event hook (1) must land before the XS trampoline (2), which must land before the
Perl formatter (3) and the server wiring (4). Decisions in phase 0 gate the rest.

## 0. Gaps found in the current code

- **C library:** no event hook exists. The only diagnostics are `INFER_LOG` in
  `src/inference_log.h` — an `fprintf(stderr)` gated on a cached `INFERENCE_VERBOSE`
  getenv. A host program cannot subscribe to anything.
- **XS layer:** the only Perl-visible callback is `on_token` (`Inference.xs`,
  `on_token_cb` + `cb_ctx` trampoline). No event callback exists.
- **Prefill is one blocking C call:** `inference_generate_stream` (`src/inference.cpp:2116`)
  and `inference_stream_begin` (`:2265`) run `sync_prompt`/`prefill_with_checkpoint`
  synchronously; the chunked-prefill path (`prefill_chunk`) is invisible to the caller.
- **Store/checkpoint events fire nowhere:** `store_snapshot_at`, `store_snapshot_every`
  (`inference.cpp:1499`, `:2041`), `checkpoint_tail` — all internal, no callback.
- **Draft events:** `spec_drafts_for_pass` / `spec_observe` (`:1552-1617`) know
  proposed/accepted per verify pass; nothing publishes them.
- **Server:** `bin/rocm-inference-server` prints unformatted `print STDERR` lines at
  load/unload (lines 130-230) and a `start_line` on request start (`:409-414`);
  Web::Route's `log =>` option (`Route.pm:290`) prints its own access line. No ms
  timestamps, no user/IP/conversation-id fields, no progress events.

## 1. C API (`src/inference.h`, `src/inference.cpp`)

1.1 Add `enum inference_event_type { INFERENCE_EVT_PREFILL_BEGIN, _PROGRESS, _END,
    _CHECKPOINT, _STORE, _DRAFT, _CONTEXT_FULL }`, the event struct, and
    `inference_set_event_cb(inference_model_t*, inference_event_cb cb, void* user)`.
    Struct: `{ int type; int seq; int n0; int n1; double ms; }` — small, no allocation on
    the hot path. `libinference.map` exports `inference_*` wholesale, so no map change.
1.2 Fire `prefill_begin`: in `inference_generate_stream` and `inference_stream_begin`,
    before `sync_prompt`/`prefill_with_checkpoint`: n0 = total prompt tokens,
    n1 = reused from the sequence (`m->last_reused` is set exactly there — read it before
    the prefill, not after).
1.3 Fire `prefill_progress`: in the chunked path (the `chunked`/`from` branch of
    `stream_begin` and its step loop) — one event per chunk, done/total. Non-chunked
    prefill: begin and end only.
1.4 Fire `prefill_end`: when the last prefill token is through, before the first sample;
    n0 = prefilled, n1 = store hits (from the store-attach in `sync_prompt`), ms measured
    with `steady_clock` at the call site.
1.5 Fire `checkpoint` where the `checkpoint_tail` snapshot is taken
    (`qwen_model_checkpoint`, recurrence only — gate on `qwen_model_is_recurrent`).
1.6 Fire `store` at the block-boundary snapshot sites (`store_snapshot_at` /
    `store_snapshot_every`, `inference.cpp:1499`, `:2041`): n0 = blocks reused,
    n1 = blocks published.
1.7 Fire `draft` in `forward_next_spec`/`spec_observe` (`:1583-1617`): proposed k,
    accepted count, pass time in ms, per verify pass.
1.8 Fire `context_full` where `INFERENCE_ERR_CONTEXT_FULL` becomes a stop reason
    (`inference_generate_stream` ~2216, `inference_stream_step`).
1.9 Acceptance test for this phase: new `tests/events_test.cpp` — load the tiny
    qwen4exp GGUF from `tools/gen_qwen4exp_tiny.py`, register a stub callback, and assert
    the event **order**: prefill_begin before the first on_token, progress monotonically
    increasing to the total, checkpoint only on recurrent models, context_full when the
    context fills.

Cost without a hook: one null check of the function pointer per event site — that is the
whole cost. Re-entrancy: events fire synchronously on the caller's thread (the engine is
single-threaded per model, like every other API here), so no mutex; the C callback must
not do I/O — it hands the struct to its host.

## 2. XS binding (`perl/ROCm-Inference/Inference.xs`, `lib/ROCm/Inference.pm`)

2.1 `ROCm::Inference::set_event_handler($model, $cb)` → `inference_set_event_cb`: store
    the code ref in a heap `cb_ctx` (same pattern as `on_token_cb` at `Inference.xs:53` —
    `PERL_IMPLICIT_CONTEXT`, `ENTER`/`SAVETMPS`, `G_EVAL`), call it with
    `($event_name, $info_hashref)`. A `die` in the handler becomes a `warn`, never stops
    generation (differs from `on_token`, where a true return stops — document both).
2.2 Map the C enum to Perl event names (`prefill_begin`, ...) in the XS, not in Perl.
2.3 Export via `@EXPORT_OK` / `:all`; add the new module to `MANIFEST` when Phase 3 lands.
2.4 Test `t/15-events.t` (GPU, `ROCM_INFERENCE_TEST_MODEL`): handler fires with the right
    args on a real generate; a die in the handler warns instead of dying.

## 3. Perl formatter — `ROCm::Inference::Log` (new `lib/ROCm/Inference/Log.pm`)

3.1 `ROCm::Inference::Log->new(%opts)`: `level => 'info'`, `format => 'text'|'json'`,
    `color => 'auto'|'always'|'never'` (auto = `-t STDERR`, honor `NO_COLOR`).
3.2 Line format: `timestamp(ms) PID level user ip conv-id req-id model event message` —
    fixed-width fields so `docker logs` stays greppable; JSON-lines mode for machines.
3.3 Colors per kind: request bold, prefill cyan, generation green, model load/unload
    magenta, warn yellow, error red.
3.4 Throttle: progress events emit at most every 10% or 2 s (whichever first);
    begin/end always emit.
3.5 Pure-Perl unit tests `t/05-log.t` — no GPU: synthetic events in, assert format,
    throttling cadence, `NO_COLOR`, and that `level` filters.

## 4. Server wiring (`perl/ROCm-Inference/bin/rocm-inference-server`)

4.1 Replace the `start_line` `print STDERR` (line 409) with a Log request-begin line;
    carry request id (already generated: `chatcmpl-%x%06x`), client IP
    (`$req->{remote}` — already present), user and conversation id through the request.
4.2 Model load/unload: wrap the load/unload paths (lines 130-230) — t0 + memory
    before/after, emit one load line (model, ms, MB); drop the raw prints.
4.3 Queue wait: in `start_job`/`tick` (486-590) stamp the queue-in time; first-token time
    closes the request line — emit "queued N ms, TTFT M ms" when generation starts.
4.4 Web::Route's request line: the `log =>` option (currently a boolean to STDERR) gains
    a code-ref form; the server passes its Log formatter so the access line goes through
    the same formatter.
4.5 Generation pacing: the `on_token` hook already exists (`Inference.xs:53`); wrap, don't
    replace — the server's wrapper keeps its `on_token` semantics (it decides stop) and
    forwards a copy to Log at the throttle cadence (3.4).

## 5. Options (`bin/rocm-inference-server`, same file as 4)

5.1 `GetOptions` gains `log-level=s`, `log-format=s`, `log-color=s`; validate values and
    die with usage on junk (same style as the existing `--kv-type` check).
5.2 Who fields: user from nginx's `X-Remote-User`, client IP from `X-Real-IP` (falls back
    to the socket peer), `conv_id`/slot header — read at request start, carried in the
    request record the whole lifecycle. Docker/nginx side: `proxy_set_header
    X-Remote-User $remote_user;` in the `/rocm/` location of the
    llama.cpp-rocm-gfx1151-docker repo — file a note there, do not edit it from here.

## 6. Verification before calling it done

6.1 `make regress` stays green: events are observers only — assert byte-identical
    generation with and without a handler (`QMATVEC_NO_WMMA=1` suite untouched).
6.2 t/60-server.t extension: run the server on the tiny model with
    `--log-level debug --log-format json`, issue one chat request, parse the captured
    STDERR as JSON lines, assert the order prefill_begin → prefill_progress… →
    prefill_end → first-token → done, and that progress reached its total.
6.3 Manual soak: one request with a long prompt on the Flash model; prefill progress ticks
    visible in real time, no per-token spam.

## Order and effort

0 (½ d, decisions) → 1 (1 d) → 2 (½ d) → 3 (½ d) → 4 (1 d) → 5 (same day as 4) →
6 (½ d). Phase 1 is the only GPU-touching phase; 0, 2, 3, 5 are CPU/Perl-only and can be
staged while a GPU run is busy. The whole plan is ~3-4 focused days with the same
dependency spine: C event struct → XS trampoline → Perl formatter → server wiring.
