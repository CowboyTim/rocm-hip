// ref_dump — Phase 20 Step 1 reference oracle.
//
// Links against the ROCmFPX llama.cpp CPU build in ./llama.cpp/build-cpu and produces
// ground truth for the standalone engine to diff against:
//
//   ref_dump types   <model.gguf>                          -> name, type id, type name per tensor
//   ref_dump tokens  <model.gguf> <prompt>                 -> token IDs on stdout
//   ref_dump acts    <model.gguf> <prompt> <outdir> [nthr] -> every F32 graph tensor of one decode
//   ref_dump dequant <model.gguf> <outdir> <tensor>...     -> raw block bytes + dequantized F32
//
// Output layout (acts): <outdir>/tokens.txt, <outdir>/index.tsv and one <seq>_<name>.f32 per tensor.
// index.tsv columns: seq, name, op, ne0, ne1, ne2, ne3, file. Values are raw little-endian float32,
// ne0 fastest (GGUF/ggml order).
//
// Output layout (dequant): <outdir>/<tensor>.raw (exact GGUF bytes), <tensor>.f32 (ggml to_float),
// and a line per tensor in <outdir>/dequant.tsv: name, type_id, type_name, ne0, ne1, raw_bytes.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    int n = -llama_tokenize(vocab, text.c_str(), (int)text.size(), nullptr, 0, add_bos, true);
    std::vector<llama_token> toks(n);
    if (llama_tokenize(vocab, text.c_str(), (int)text.size(), toks.data(), n, add_bos, true) != n) {
        fprintf(stderr, "tokenize failed\n");
        exit(1);
    }
    return toks;
}

static std::string sanitize(const char * name) {
    std::string s;
    for (const char * p = name; *p; ++p) {
        char c = *p;
        s += (isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
    }
    return s;
}

static void mkdirs(const std::string & dir) {
    for (size_t i = 1; i <= dir.size(); ++i) {
        if (i == dir.size() || dir[i] == '/') {
            mkdir(dir.substr(0, i).c_str(), 0755);
        }
    }
}

static FILE * xfopen(const std::string & path, const char * mode) {
    FILE * f = fopen(path.c_str(), mode);
    if (!f) {
        perror(path.c_str());
        exit(1);
    }
    return f;
}

struct dump_state {
    std::string outdir;
    FILE * index = nullptr;
    int seq = 0;
    size_t bytes = 0;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return true; // observe every node
    }
    auto * st = (dump_state *)user_data;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) == 0) {
        return true;
    }
    // KV/recurrent cache views are large and not per-token activations
    if (strncmp(t->name, "cache_", 6) == 0) {
        return true;
    }
    const size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> buf(nb);
    ggml_backend_tensor_get(t, buf.data(), 0, nb);

    char fname[512];
    snprintf(fname, sizeof(fname), "%04d_%s.f32", st->seq, sanitize(t->name).c_str());
    std::string path = st->outdir + "/" + fname;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        perror(path.c_str());
        return false;
    }
    fwrite(buf.data(), 1, nb, f);
    fclose(f);

    fprintf(st->index, "%d\t%s\t%s\t%lld\t%lld\t%lld\t%lld\t%s\n", st->seq, t->name, ggml_op_desc(t),
            (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3], fname);
    st->seq++;
    st->bytes += nb;
    return true;
}

static llama_model * load_model(const char * path) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        exit(1);
    }
    return model;
}

static int cmd_tokens(const char * model_path, const char * prompt) {
    llama_model * model = load_model(model_path);
    auto toks = tokenize(llama_model_get_vocab(model), prompt);
    for (size_t i = 0; i < toks.size(); ++i) {
        printf("%s%d", i ? " " : "", toks[i]);
    }
    printf("\n");
    llama_model_free(model);
    return 0;
}

static int cmd_acts(const char * model_path, const char * prompt, const std::string & outdir, int nthr) {
    mkdirs(outdir);
    llama_model * model = load_model(model_path);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto toks = tokenize(vocab, prompt);

    dump_state st;
    st.outdir = outdir;
    st.index = xfopen(outdir + "/index.tsv", "w");
    fprintf(st.index, "seq\tname\top\tne0\tne1\tne2\tne3\tfile\n");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_threads = nthr;
    cp.n_threads_batch = nthr;
    // REF_DUMP_NO_CB=1: no eval callback (nothing dumped), so llama.cpp runs its graph as llama-completion does
    // (the callback makes the scheduler compute and keep every tensor); only the top-1 next token
    if (!getenv("REF_DUMP_NO_CB")) {
        cp.cb_eval = dump_cb;
        cp.cb_eval_user_data = &st;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }

    FILE * tf = xfopen(outdir + "/tokens.txt", "w");
    for (size_t i = 0; i < toks.size(); ++i) {
        fprintf(tf, "%s%d", i ? " " : "", toks[i]);
    }
    fprintf(tf, "\n");
    fclose(tf);

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int)toks.size()))) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    fclose(st.index);

    // top-1 next token after the last prompt position
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    char piece[256];
    int n = llama_token_to_piece(vocab, best, piece, sizeof(piece) - 1, 0, true);
    piece[n > 0 ? n : 0] = 0;
    FILE * nf = xfopen(outdir + "/next_token.txt", "w");
    fprintf(nf, "%d\t%s\n", best, piece);
    fclose(nf);

    printf("tokens: %zu, tensors dumped: %d (%.1f MB), top-1 next token: %d '%s'\n",
           toks.size(), st.seq, st.bytes / 1e6, best, piece);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

static int cmd_dequant(const char * model_path, const std::string & outdir, int n_names, char ** names) {
    mkdirs(outdir);
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(model_path, gp);
    if (!g) {
        fprintf(stderr, "gguf_init_from_file failed\n");
        return 1;
    }
    FILE * mf = xfopen(model_path, "rb");
    FILE * tsv = xfopen(outdir + "/dequant.tsv", "a");
    const size_t data_off = gguf_get_data_offset(g);

    for (int i = 0; i < n_names; ++i) {
        const char * name = names[i];
        int64_t id = gguf_find_tensor(g, name);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        if (id < 0 || !t) {
            fprintf(stderr, "tensor not found: %s\n", name);
            return 1;
        }
        const ggml_type type = gguf_get_tensor_type(g, id);
        const size_t nb = ggml_nbytes(t);
        std::vector<uint8_t> raw(nb);
        fseeko(mf, (off_t)(data_off + gguf_get_tensor_offset(g, id)), SEEK_SET);
        if (fread(raw.data(), 1, nb, mf) != nb) {
            fprintf(stderr, "short read: %s\n", name);
            return 1;
        }
        const int64_t ne = ggml_nelements(t);
        std::vector<float> out(ne);
        const auto * tr = ggml_get_type_traits(type);
        if (type == GGML_TYPE_F32) {
            memcpy(out.data(), raw.data(), ne * sizeof(float));
        } else if (tr->to_float) {
            tr->to_float(raw.data(), out.data(), ne);
        } else {
            fprintf(stderr, "no to_float for type %d (%s)\n", (int)type, tr->type_name);
            return 1;
        }

        std::string base = outdir + "/" + sanitize(name);
        FILE * f = xfopen(base + ".raw", "wb");
        fwrite(raw.data(), 1, nb, f);
        fclose(f);
        f = xfopen(base + ".f32", "wb");
        fwrite(out.data(), sizeof(float), ne, f);
        fclose(f);
        fprintf(tsv, "%s\t%d\t%s\t%lld\t%lld\t%zu\n", name, (int)type, tr->type_name,
                (long long)t->ne[0], (long long)t->ne[1], nb);
        printf("%-32s type=%d (%s) ne=[%lld, %lld] raw=%zu B  v[0..3]= %.6f %.6f %.6f %.6f\n", name, (int)type,
               tr->type_name, (long long)t->ne[0], (long long)t->ne[1], nb, out[0], out[1], out[2], out[3]);
    }
    fclose(tsv);
    fclose(mf);
    gguf_free(g);
    ggml_free(meta);
    return 0;
}

static int cmd_types(const char * model_path) {
    gguf_init_params gp = { /*no_alloc*/ true, nullptr };
    gguf_context * g = gguf_init_from_file(model_path, gp);
    if (!g) {
        fprintf(stderr, "gguf_init_from_file failed\n");
        return 1;
    }
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const ggml_type type = gguf_get_tensor_type(g, i);
        printf("%s\t%d\t%s\n", gguf_get_tensor_name(g, i), (int)type, ggml_type_name(type));
    }
    gguf_free(g);
    return 0;
}

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage:\n"
            "  %s types   <model.gguf>                 (name, type id, type name per tensor)\n"
            "  %s tokens  <model.gguf> <prompt>\n"
            "  %s acts    <model.gguf> <prompt> <outdir> [threads]\n"
            "  %s dequant <model.gguf> <outdir> <tensor>...\n",
            argv0, argv0, argv0, argv0);
}

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    if (std::string(argv[1]) == "types") {
        return cmd_types(argv[2]);
    }
    if (argc < 4) {
        usage(argv[0]);
        return 1;
    }
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    const std::string cmd = argv[1];
    int rc = 1;
    if (cmd == "tokens") {
        rc = cmd_tokens(argv[2], argv[3]);
    } else if (cmd == "acts" && argc >= 5) {
        rc = cmd_acts(argv[2], argv[3], argv[4], argc >= 6 ? atoi(argv[5]) : 16);
    } else if (cmd == "dequant" && argc >= 5) {
        rc = cmd_dequant(argv[2], argv[3], argc - 4, argv + 4);
    } else {
        usage(argv[0]);
    }
    llama_backend_free();
    return rc;
}
