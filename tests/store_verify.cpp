// store_verify -- the KV block store across processes (Phase 22b). One phase per process, sharing one store
// directory (tests/store_verify.sh runs them in order):
//
//   store_verify <model.gguf> <store_dir> 1   fresh store: a long prompt publishes its blocks and snapshots; the
//                                             greedy reply is written to <store_dir>.expected
//   store_verify <model.gguf> <store_dir> 2   a new process: the same prompt starts from the store and gives the
//                                             same reply; a prompt sharing 3 blocks resumes at the 2-block
//                                             snapshot; diverging inside a shared block leaves its file unchanged
//   store_verify <model.gguf> <store_dir> 3   after block files were deleted (eviction): lookup stops at the gap
//
// Snapshots every 2 blocks (and at each prompt's last block boundary). Replies are compared exactly with
// QMATVEC_NO_WMMA=1, else by their first tokens (WMMA prefill sums in a different order when cut differently).

#include "inference.h"
#include "sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static std::vector<int32_t> tokens(inference_model_t* m, const std::string& text) {
    int n = 0;
    int32_t* t = inference_tokenize(m, text.c_str(), &n);
    std::vector<int32_t> v(t, t + n);
    inference_free_tokens(t, n);
    return v;
}

static std::vector<int32_t> run(inference_model_t* m, const std::vector<int32_t>& prompt, int n_gen, int* reused,
                                int* prefilled) {
    inference_gen_options o = inference_gen_default_options();
    o.sampler.temperature = 0.0f;
    o.max_new_tokens = n_gen;
    o.reuse_prefix = 1;
    o.ignore_eog = 1;
    std::vector<int32_t> out((size_t)n_gen);
    int n_out = 0;
    inference_stop_reason why;
    if (inference_generate_stream(m, prompt.data(), (int)prompt.size(), &o, out.data(), n_gen, &n_out, &why) != 0) {
        printf("  generate failed: %s\n", inference_get_last_error());
        return {};
    }
    out.resize((size_t)n_out);
    inference_last_prompt_stats(m, reused, prefilled);
    return out;
}

static std::string file_sha(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    sha256_ctx c;
    sha256_init(&c);
    char buf[1 << 16];
    for (size_t n; (n = fread(buf, 1, sizeof(buf), f)) > 0;) sha256_update(&c, buf, n);
    fclose(f);
    uint8_t h[32];
    sha256_final(&c, h);
    char hex[65];
    for (int i = 0; i < 32; ++i) snprintf(hex + 2 * i, 3, "%02x", h[i]);
    return hex;
}

static std::vector<std::string> files(const std::string& dir, const char* ext) {
    std::vector<std::string> v;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > strlen(ext) && n.compare(n.size() - strlen(ext), strlen(ext), ext) == 0) v.push_back(n);
        }
        closedir(d);
    }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <store_dir> <phase 1|2|3>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    const int phase = atoi(argv[3]);
    const bool exact = getenv("QMATVEC_NO_WMMA") != nullptr;
    inference_model_t* m = inference_load_model(argv[1]);
    // STORE_VERIFY_DRAFT=<mtp head>: with speculative decoding (snapshots must never hold a draft)
    const char* draft = getenv("STORE_VERIFY_DRAFT");
    if (!m || (draft && inference_set_draft_model(m, draft) != 0) || inference_kv_cache_init(m, 4096) != 0 ||
        inference_store_open(m, dir.c_str(), 2) != 0) {
        fprintf(stderr, "setup failed: %s\n", inference_get_last_error());
        return 1;
    }
    inference_store_stats st;
    inference_store_get_stats(m, &st);
    const int B = st.block_tokens;
    printf("store_verify phase %d: %s, blocks of %d tokens (%lld bytes), snapshots %lld bytes\n", phase, dir.c_str(), B,
           st.block_bytes, st.snapshot_bytes);

    // A prompt of about 4.5 blocks
    std::string text;
    for (int i = 0; text.size() < 6000; ++i)
        text += "Paragraph " + std::to_string(i) + ": the archive keeps records of every shipment, its weight, "
                "its origin and the date it arrived at the harbour. ";
    std::vector<int32_t> prompt = tokens(m, text + "\nSummary of the archive:");
    printf("  prompt %zu tokens (%zu full blocks)\n", prompt.size(), prompt.size() / B);
    const std::string expected_path = dir + ".expected";
    char what[256];
    int reused = 0, prefilled = 0;

    if (phase == 0) {  // print the store file of block 1 of the prompt (for the script to delete)
        std::vector<char> names(4096);
        inference_store_block_names(m, prompt.data(), (int)prompt.size(), names.data(), (int)names.size());
        std::string all(names.data());
        const size_t a = all.find('\n') + 1;
        printf("BLOCK1 %s\n", all.substr(a, all.find('\n', a) - a).c_str());
    } else if (phase == 1) {
        const std::vector<int32_t> out = run(m, prompt, 24, &reused, &prefilled);
        inference_store_get_stats(m, &st);
        const long long blocks = (long long)(prompt.size() + out.size() - 1) / B;  // the last token is not decoded
        snprintf(what, sizeof(what), "fresh store: nothing reused, %lld blocks published (expected %lld), %lld snapshots",
                 st.blocks_published, blocks, st.snapshots_saved);
        check(reused == 0 && (int)out.size() == 24 && st.blocks_published == blocks && st.snapshots_saved >= 2, what);
        check((long long)files(dir, ".kvb").size() == blocks && (long long)files(dir, ".rst").size() == st.snapshots_saved,
              "the store directory holds exactly those .kvb and .rst files (no temporary files)");
        FILE* f = fopen(expected_path.c_str(), "w");
        for (int32_t t : out) fprintf(f, "%d\n", t);
        fclose(f);
    } else {
        std::vector<int32_t> expected;
        if (FILE* f = fopen(expected_path.c_str(), "r")) {
            for (int t; fscanf(f, "%d", &t) == 1;) expected.push_back(t);
            fclose(f);
        }
        const int deepest = (int)((prompt.size() - 1) / B) * B;  // snapshot at the prompt's last boundary
        if (phase == 2) {
            const std::vector<int32_t> out = run(m, prompt, 24, &reused, &prefilled);
            inference_store_get_stats(m, &st);
            const bool same = exact ? out == expected : !out.empty() && !expected.empty() && out[0] == expected[0];
            snprintf(what, sizeof(what), "new process, same prompt: %d of %zu tokens from the store (%lld blocks mapped, "
                     "%lld snapshot restored), reply %s", reused, prompt.size(), st.blocks_attached,
                     st.snapshots_restored, same ? (exact ? "identical" : "same first token") : "DIFFERS");
            check(reused == deepest && st.tokens_from_store == deepest && st.snapshots_restored == 1 && same, what);

            // A prompt sharing the first 3 blocks: the deepest snapshot inside them is at 2 blocks (every 2)
            std::vector<int32_t> p2(prompt.begin(), prompt.begin() + 3 * B);
            const std::vector<int32_t> tail = tokens(m, " A completely different ending about mountains.");
            p2.insert(p2.end(), tail.begin(), tail.end());
            inference_reset(m);
            run(m, p2, 8, &reused, &prefilled);
            snprintf(what, sizeof(what), "a prompt sharing 3 blocks resumes at the 2-block snapshot (%d tokens reused)",
                     reused);
            check(reused == 2 * B, what);

            // Diverging inside block 0, which is now a shared store block: its file must not change
            const std::vector<std::string> kvb = files(dir, ".kvb");
            std::vector<std::string> before;
            for (const std::string& f : kvb) before.push_back(file_sha(dir + "/" + f));
            std::vector<int32_t> p3(prompt.begin(), prompt.begin() + B / 2);
            const std::vector<int32_t> other = tokens(m, " Something else entirely, written over the shared block.");
            p3.insert(p3.end(), other.begin(), other.end());
            run(m, p3, 8, &reused, &prefilled);
            bool unchanged = true;
            for (size_t i = 0; i < kvb.size(); ++i) unchanged &= file_sha(dir + "/" + kvb[i]) == before[i];
            check(unchanged, "writing positions inside shared blocks leaves every store file unchanged (copy on write)");
        } else if (phase == 3) {
            // Phase 3: block 1 of the prompt was deleted, so nothing past block 1 can be used: the deepest
            // remaining resume point is 0 (snapshots are at 2, 4, ...)
            const std::vector<int32_t> out = run(m, prompt, 24, &reused, &prefilled);
            snprintf(what, sizeof(what), "after deleting block 1: %d tokens reused, %d prefilled, reply %s", reused,
                     prefilled, !out.empty() && !expected.empty() && out[0] == expected[0] ? "same first token" : "DIFFERS");
            check(reused == 0 && prefilled == (int)prompt.size() && !out.empty() && out[0] == expected[0], what);
            inference_store_get_stats(m, &st);
            check(st.blocks_published >= 1, "the missing block is published again");
        }
    }
    inference_store_close(m);
    inference_kv_cache_free(m);
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
