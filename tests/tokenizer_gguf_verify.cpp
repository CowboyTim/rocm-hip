// tokenizer_gguf_verify -- the GGUF tokenizer vs llama.cpp's, on a fixed prompt corpus (Phase 20).
//
//   tokenizer_gguf_verify <model.gguf> <ref.tsv>
//
// ref.tsv comes from tools/gen_tokenizer_ref.py: one line per prompt, <prompt as hex> TAB <reference ids>.
// Each prompt must encode to exactly the reference ids. Decoding those ids must give back the prompt bytes,
// except for prompts with invalid UTF-8, which the tokenizer (like llama.cpp) replaces with U+FFFD.

#include "gguf_mmap.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string unhex(const std::string& h) {
    std::string out;
    for (size_t i = 0; i + 1 < h.size(); i += 2) out += (char)std::stoi(h.substr(i, 2), nullptr, 16);
    return out;
}

static bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!n || i + n > s.size()) return false;
        for (int k = 1; k < n; ++k)
            if (((unsigned char)s[i + k] & 0xC0) != 0x80) return false;
        i += n;
    }
    return true;
}

static std::string printable(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20 || c == 0x7F) { char b[8]; snprintf(b, sizeof(b), "\\x%02x", c); out += b; }
        else out += (char)c;
    }
    return out.size() > 48 ? out.substr(0, 45) + "..." : out;
}

static std::string join(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <ref.tsv>\n", argv[0]);
        return 2;
    }
    gguf_model_t* g = gguf_mmap_open(argv[1]);
    if (!g) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    tokenizer_t* tok = tokenizer_load_from_gguf(g);
    if (!tok) { fprintf(stderr, "tokenizer_load_from_gguf failed\n"); return 1; }
    printf("tokenizer_gguf_verify: %s, vocab %zu, eos %d\n", argv[1], tokenizer_vocab_size(tok), tokenizer_eos_id(tok));

    // A tokenizer that adds BOS (Gemma-4) encodes "" to that one id: it is not part of the prompt's text.
    // One that writes spaces as U+2581 (SentencePiece style) decodes a literal U+2581 as a space too.
    int probe[4];
    const bool adds_bos = tokenizer_encode(tok, "", probe, 4) == 1 && probe[0] == tokenizer_bos_id(tok);
    const char* lower_block = "\xe2\x96\x81";
    std::vector<int> lb(8);
    lb.resize(std::max(0, tokenizer_encode(tok, lower_block, lb.data(), (int)lb.size())));
    char lbd[16] = {0};
    tokenizer_decode(tok, lb.data() + (adds_bos ? 1 : 0), (int)lb.size() - (adds_bos ? 1 : 0), lbd, sizeof(lbd));
    const bool spm_spaces = std::string(lbd) == " ";

    std::ifstream f(argv[2]);
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    int n = 0, fail = 0;
    std::string line;
    while (std::getline(f, line)) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string prompt = unhex(line.substr(0, tab));
        std::vector<int> want;
        std::istringstream ss(line.substr(tab + 1));
        for (int id; ss >> id;) want.push_back(id);

        std::vector<int> got(prompt.size() + 8);
        const int k = tokenizer_encode(tok, prompt.c_str(), got.data(), (int)got.size());
        got.resize(k < 0 ? 0 : k);
        bool ok = k >= 0 && got == want;

        std::string dec(prompt.size() * 4 + 64, '\0');
        const int skip = adds_bos && !want.empty() && want[0] == tokenizer_bos_id(tok) ? 1 : 0;
        const int dl = tokenizer_decode(tok, want.data() + skip, (int)want.size() - skip, &dec[0], (int)dec.size());
        dec.resize(dl < 0 ? 0 : dl);
        std::string expect = prompt;
        if (spm_spaces)
            for (size_t p; (p = expect.find(lower_block)) != std::string::npos;) expect.replace(p, 3, " ");
        const bool rt = !valid_utf8(prompt) || dec == expect;
        ok = ok && rt;

        ++n;
        if (!ok) ++fail;
        printf("  [%s] %-48s %zu tokens%s\n", ok ? "PASS" : "FAIL", printable(prompt).c_str(), want.size(),
               rt ? "" : "  (decode does not round-trip)");
        if (got != want) printf("      ours: %s\n      ref:  %s\n", join(got).c_str(), join(want).c_str());
    }
    tokenizer_free(tok);
    gguf_mmap_close(g);
    printf("%s (%d of %d prompts failed)\n", fail ? "FAIL" : "PASS", fail, n);
    return fail || n == 0 ? 1 : 0;
}
