// dequant_verify — Phase 20 Step 3: our quant_types decoders vs the reference dequant.
//
//   dequant_verify <model.gguf> <refdir>
//
// <refdir> is produced by `build/ref_dump dequant <model.gguf> <refdir> <tensor>...` and holds
// dequant.tsv plus <tensor>.f32 per tensor. Each tensor is read through our own GGUF mmap,
// decoded with qt_to_float() and compared bit-for-bit. Exit code 0 only if all match.

#include "gguf_mmap.h"
#include "quant_types.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::string sanitize(const std::string &name) {
    std::string s;
    for (char c : name) s += (isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_') ? c : '_';
    return s;
}

static bool read_file(const std::string &path, std::vector<float> &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize((size_t)sz / sizeof(float));
    bool ok = fread(out.data(), 1, (size_t)sz, f) == (size_t)sz;
    fclose(f);
    return ok;
}

static int64_t find_tensor(gguf_model_t *g, const char *name) {
    for (int64_t i = 0; i < gguf_n_tensors(g); ++i) {
        if (strcmp(gguf_tensor_name(g, i), name) == 0) return i;
    }
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.gguf> <refdir>\n", argv[0]);
        return 2;
    }
    gguf_model_t *g = gguf_mmap_open(argv[1]);
    if (!g) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    const std::string refdir = argv[2];
    FILE *tsv = fopen((refdir + "/dequant.tsv").c_str(), "r");
    if (!tsv) {
        fprintf(stderr, "cannot open %s/dequant.tsv\n", refdir.c_str());
        return 2;
    }

    int n_fail = 0, n_ok = 0;
    char line[1024];
    std::vector<std::string> seen;
    while (fgets(line, sizeof(line), tsv)) {
        char name[512], tname[64];
        int type;
        if (sscanf(line, "%511s\t%d\t%63s", name, &type, tname) != 3) continue;
        bool dup = false;
        for (auto &s : seen) dup |= (s == name);
        if (dup) continue;
        seen.push_back(name);

        std::vector<float> ref;
        if (!read_file(refdir + "/" + sanitize(name) + ".f32", ref)) {
            printf("  [FAIL] %-32s missing reference .f32\n", name);
            ++n_fail;
            continue;
        }
        const int64_t idx = find_tensor(g, name);
        if (idx < 0) {
            printf("  [FAIL] %-32s not in GGUF\n", name);
            ++n_fail;
            continue;
        }
        const int our_type = gguf_tensor_type(g, idx);
        const qt_traits *tr = qt_get_traits(our_type);
        size_t bytes = 0;
        const void *ptr = gguf_get_tensor_ptr(g, name, &bytes);
        if (!tr || !ptr || our_type != type) {
            printf("  [FAIL] %-32s type %d (ref %d %s) unsupported or mismatched\n", name, our_type, type, tname);
            ++n_fail;
            continue;
        }
        const int64_t n = (int64_t)ref.size();
        const size_t want = (size_t)(n / tr->blck_size) * tr->type_size;
        if (bytes != 0 && bytes < want) {
            printf("  [FAIL] %-32s GGUF tensor is %zu B, expected >= %zu B\n", name, bytes, want);
            ++n_fail;
            continue;
        }
        std::vector<float> ours((size_t)n);
        if (qt_to_float(our_type, ptr, ours.data(), n) != 0) {
            printf("  [FAIL] %-32s qt_to_float rejected n=%lld\n", name, (long long)n);
            ++n_fail;
            continue;
        }
        int64_t n_diff = 0, first = -1;
        double max_abs = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            uint32_t a, b;
            memcpy(&a, &ours[i], 4);
            memcpy(&b, &ref[i], 4);
            if (a != b) {
                if (first < 0) first = i;
                ++n_diff;
                max_abs = std::fmax(max_abs, std::fabs((double)ours[i] - ref[i]));
            }
        }
        if (n_diff == 0) {
            printf("  [PASS] %-32s %-18s %lld values bit-exact\n", name, tr->name, (long long)n);
            ++n_ok;
        } else {
            printf("  [FAIL] %-32s %-18s %lld/%lld differ, max |d|=%g, first @%lld ours=%.9g ref=%.9g\n", name,
                   tr->name, (long long)n_diff, (long long)n, max_abs, (long long)first, ours[first], ref[first]);
            ++n_fail;
        }
    }
    fclose(tsv);
    gguf_mmap_close(g);
    printf("dequant_verify: %d passed, %d failed\n", n_ok, n_fail);
    return n_fail ? 1 : 0;
}
