// tokenizer.cpp -- GPT-2-style byte-level BPE for GGUF vocabularies (tokenizer.ggml.model = "gpt2").
//
// Matches llama.cpp's llm_tokenizer_bpe (the reference, Phase 20), in four stages:
//   1. Special tokens: CONTROL / USER_DEFINED / UNKNOWN tokens found literally in the text are emitted as
//      themselves, longest first; only the text between them is tokenized further.
//   2. Pre-tokenizer: the text is split into words by the model's regex (tokenizer.ggml.pre). Implemented
//      by hand like llama.cpp's custom splitters, using Unicode category flags from unicode_flags.inc.
//   3. Byte encoding: each UTF-8 byte of a word maps to one printable codepoint (GPT-2 bytes_to_unicode,
//      so ' ' becomes 'Ġ'), the alphabet the vocab and merges are written in.
//   4. BPE: adjacent symbols are merged lowest merge rank first (leftmost on ties) until no pair is in
//      the merge table; each final symbol is looked up in the vocab.
// Supported pre-tokenizers: qwen35, qwen2 (and models that share their regex).
//
// tokenizer.ggml.model = "gemma4" is SentencePiece-style BPE instead (llama.cpp's LLAMA_VOCAB_PRE_TYPE_GEMMA4):
// stage 2 splits only into runs of newlines and runs of other text, after every space became U+2581; stage 3
// is skipped (the vocab and merges are raw UTF-8); a newline run that is a token is taken whole; a symbol not
// in the vocab falls back to <0xXX> byte tokens; BOS is always added. Decoding turns U+2581 back into spaces.

#include "inference_log.h"
#include "tokenizer.h"
#include "gguf_mmap.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "unicode_flags.inc"

namespace {

// ---------------------------------------------------------------- Unicode helpers

enum : uint16_t {
    CPT_UNDEFINED = 0x0001,
    CPT_NUMBER = 0x0002,      // \p{N}
    CPT_LETTER = 0x0004,      // \p{L}
    CPT_ACCENT_MARK = 0x0010, // \p{M}
    CPT_WHITESPACE = 0x0100,  // \s
};

const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;

// Category flags of every codepoint (0x110000 entries), expanded once from the range table
const std::vector<uint16_t>& cpt_flag_table() {
    static const std::vector<uint16_t> table = [] {
        std::vector<uint16_t> t(0x110000, CPT_UNDEFINED);
        const size_t n = sizeof(UNICODE_RANGES) / sizeof(UNICODE_RANGES[0]);
        for (size_t i = 0; i + 1 < n; ++i)
            for (uint32_t c = UNICODE_RANGES[i].start; c < UNICODE_RANGES[i + 1].start; ++c) t[c] = UNICODE_RANGES[i].flags;
        for (uint32_t c : UNICODE_WHITESPACE) t[c] |= CPT_WHITESPACE;
        return t;
    }();
    return table;
}

uint16_t cpt_flags(uint32_t c) {
    if (c == OUT_OF_RANGE) return 0;
    const std::vector<uint16_t>& t = cpt_flag_table();
    return c < t.size() ? t[c] : CPT_UNDEFINED;
}

// UTF-8 decoding as llama.cpp's unicode_cpts_from_utf8: an invalid lead or truncated sequence becomes
// U+FFFD and consumes one byte
std::vector<uint32_t> utf8_to_cpts(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    const unsigned char* p = (const unsigned char*)s.data();
    const size_t n = s.size();
    auto cont = [&](size_t i) { return i < n && (p[i] & 0xC0) == 0x80; };
    for (size_t i = 0; i < n;) {
        const unsigned char c = p[i];
        if (!(c & 0x80)) { out.push_back(c); i += 1; }
        else if (!(c & 0x40)) { out.push_back(0xFFFD); i += 1; }
        else if (!(c & 0x20)) {
            if (cont(i + 1)) { out.push_back(((c & 0x1F) << 6) | (p[i + 1] & 0x3F)); i += 2; }
            else { out.push_back(0xFFFD); i += 1; }
        } else if (!(c & 0x10)) {
            if (cont(i + 1) && cont(i + 2)) {
                out.push_back(((c & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F));
                i += 3;
            } else { out.push_back(0xFFFD); i += 1; }
        } else if (!(c & 0x08)) {
            if (cont(i + 1) && cont(i + 2) && cont(i + 3)) {
                out.push_back(((c & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) | ((p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F));
                i += 4;
            } else { out.push_back(0xFFFD); i += 1; }
        } else { out.push_back(0xFFFD); i += 1; }
    }
    return out;
}

void append_utf8(std::string& s, uint32_t c) {
    if (c < 0x80) s += (char)c;
    else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) {
        s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F));
    } else {
        s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F));
        s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F));
    }
}

// GPT-2 bytes_to_unicode: printable Latin-1 bytes map to themselves, the other 68 to U+0100 upward
struct ByteMap {
    std::string byte_to_utf8[256];
    std::unordered_map<uint32_t, uint8_t> cpt_to_byte;
    ByteMap() {
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            const uint32_t c = printable ? (uint32_t)b : 256u + n++;
            append_utf8(byte_to_utf8[b], c);
            cpt_to_byte[c] = (uint8_t)b;
        }
    }
};

const ByteMap& byte_map() {
    static const ByteMap m;
    return m;
}

uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// ---------------------------------------------------------------- pre-tokenizers

enum PreType { PRE_QWEN2, PRE_QWEN35, PRE_GEMMA4, PRE_LLAMA3 };

// Word lengths (in codepoints) for llama.cpp's qwen2 / qwen35 regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// qwen35 treats combining marks (\p{M}) as letters in the letter run and the punctuation run. llama3 (also
// llama.cpp's pre-tokenizer for pixtral, Nemotron-3) is qwen2's with \p{N}{1,3}: digits in runs of up to max_digits.
// Port of unicode_regex_split_custom_qwen2/_qwen35/_llama3.
std::vector<size_t> split_qwen(const std::vector<uint32_t>& cpts, bool marks, int max_digits = 1) {
    std::vector<size_t> words;
    const size_t end = cpts.size();
    const uint16_t letter = marks ? (CPT_LETTER | CPT_ACCENT_MARK) : CPT_LETTER;
    auto cpt = [&](size_t i) { return i < end ? cpts[i] : OUT_OF_RANGE; };
    auto flags = [&](size_t i) { return cpt_flags(cpt(i)); };

    size_t prev = 0;
    auto add = [&](size_t to) {
        const size_t len = to - prev;
        if (len > 0) words.push_back(len);
        prev = to;
        return len;
    };

    for (size_t pos = 0; pos < end;) {
        const uint32_t c = cpt(pos);
        const uint16_t f = flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && pos + 1 < end) {
            const uint32_t n1 = ascii_lower(cpt(pos + 1));
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { pos += add(pos + 2); continue; }
            if (pos + 2 < end) {
                const uint32_t n2 = ascii_lower(cpt(pos + 2));
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    pos += add(pos + 3);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}(\p{M})]+
        if (!(c == '\r' || c == '\n' || (f & CPT_NUMBER))) {
            if ((f & letter) || (flags(pos + 1) & letter)) {
                pos++;
                while (flags(pos) & letter) pos++;
                add(pos);
                continue;
            }
        }

        // \p{N}{1,max_digits}
        if (f & CPT_NUMBER) {
            size_t n = 0;
            while ((int)n < max_digits && (flags(pos + n) & CPT_NUMBER)) n++;
            pos += n;
            add(pos);
            continue;
        }

        // <space>?[^\s\p{L}(\p{M})\p{N}]+[\r\n]*
        const uint16_t stop = CPT_WHITESPACE | letter | CPT_NUMBER;
        uint16_t f2 = c == ' ' ? flags(pos + 1) : f;
        if (!(f2 & stop) && f) {
            pos += (c == ' ');
            while (!(f2 & stop) && f2) f2 = flags(++pos);
            uint32_t c2 = cpt(pos);
            while (c2 == '\r' || c2 == '\n') c2 = cpt(++pos);
            add(pos);
            continue;
        }

        size_t n_ws = 0, last_rn_end = 0;
        while (flags(pos + n_ws) & CPT_WHITESPACE) {
            const uint32_t c2 = cpt(pos + n_ws);
            if (c2 == '\r' || c2 == '\n') last_rn_end = pos + n_ws + 1;
            n_ws++;
        }
        if (last_rn_end > 0) { pos = last_rn_end; add(pos); continue; }               // \s*[\r\n]+
        if (n_ws > 1 && cpt(pos + n_ws) != OUT_OF_RANGE) { pos += n_ws - 1; add(pos); continue; }  // \s+(?!\S)
        if (n_ws > 0) { pos += n_ws; add(pos); continue; }                               // \s+
        add(++pos);                                                                     // no match
    }
    return words;
}

// ---------------------------------------------------------------- tokenizer

enum TokenType { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER_DEFINED = 4, TT_UNUSED = 5, TT_BYTE = 6 };

} // namespace

struct tokenizer {
    std::vector<std::string> vocab;                    // id -> token text (byte-encoded for normal tokens)
    std::vector<int> type;                             // id -> TokenType
    std::unordered_map<std::string, int> vocab_map;    // token text -> id
    std::unordered_map<std::string, int> merge_rank;   // "left right" -> rank (lower merges first)
    std::vector<int> specials;                         // CONTROL/USER_DEFINED/UNKNOWN ids, longest text first
    PreType pre = PRE_QWEN2;
    int bos_id = -1, eos_id = -1, pad_id = -1, unk_id = -1;
    bool add_bos = false;
    std::vector<int> eog;  // end-of-generation ids, sorted (build_eog)

    int lookup(const std::string& s) const {
        auto it = vocab_map.find(s);
        return it == vocab_map.end() ? -1 : it->second;
    }

    void build_specials() {
        specials.clear();
        for (int id = 0; id < (int)vocab.size(); ++id)
            if (type[id] == TT_CONTROL || type[id] == TT_USER_DEFINED || type[id] == TT_UNKNOWN) specials.push_back(id);
        std::stable_sort(specials.begin(), specials.end(),
                         [&](int a, int b) { return vocab[a].size() > vocab[b].size(); });
    }

    // End-of-generation tokens, as llama.cpp's llama_vocab::impl::load (llama-vocab.cpp) builds them: the GGUF's
    // EOS / EOT / EOM and FIM pad / rep / sep ids, plus every token whose text is one of the end-of-turn
    // markers below (by text, as the reference does)
    void build_eog(const std::vector<int>& ids) {
        static const char* const kText[] = {
            "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>",
            "<end_of_turn>", "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]",
            "<|end_of_text|>", "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>",
            "<\xef\xbd\x9c" "end\xe2\x96\x81" "of\xe2\x96\x81" "sentence\xef\xbd\x9c>"};  // DeepSeek, split: \x eats hex letters
        eog.clear();
        for (int id : ids)
            if (id >= 0 && id < (int)vocab.size()) eog.push_back(id);
        for (const char* t : kText) {
            const int id = lookup(t);
            if (id < 0) continue;
            eog.push_back(id);
            if (type[id] != TT_CONTROL) type[id] = TT_CONTROL;  // as llama.cpp: an end-of-turn marker is a control token
        }
        std::sort(eog.begin(), eog.end());
        eog.erase(std::unique(eog.begin(), eog.end()), eog.end());
        // llama.cpp's workaround for Gemma-4: with <|tool_response> among them, </s> is no end of generation and
        // becomes a normal token (so </s> in text is no longer a special token either)
        const int tr = lookup("<|tool_response>"), s_end = lookup("</s>");
        if (tr >= 0 && s_end >= 0 && is_eog(tr) && is_eog(s_end)) {
            eog.erase(std::find(eog.begin(), eog.end(), s_end));
            type[s_end] = TT_NORMAL;
        }
    }

    bool is_eog(int id) const { return std::binary_search(eog.begin(), eog.end(), id); }

    // BPE over one byte-encoded word; appends the token ids
    void bpe_word(const std::string& word, std::vector<int>& out) const {
        struct Sym { int prev, next; size_t off, n; };
        std::vector<Sym> sym;
        for (size_t off = 0; off < word.size();) {
            const unsigned char c = (unsigned char)word[off];
            const size_t len = std::min(word.size() - off, (size_t)(c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4));
            sym.push_back({(int)sym.size() - 1, -1, off, len});
            off += len;
        }
        for (size_t i = 0; i + 1 < sym.size(); ++i) sym[i].next = (int)i + 1;

        struct Bigram { int rank, left, right; std::string text; };
        auto later = [](const Bigram& a, const Bigram& b) { return a.rank > b.rank || (a.rank == b.rank && a.left > b.left); };
        std::priority_queue<Bigram, std::vector<Bigram>, decltype(later)> q(later);
        auto add_bigram = [&](int l, int r) {
            if (l < 0 || r < 0) return;
            const std::string a = word.substr(sym[l].off, sym[l].n), b = word.substr(sym[r].off, sym[r].n);
            auto it = merge_rank.find(a + " " + b);
            if (it != merge_rank.end()) q.push({it->second, l, r, a + b});
        };
        for (size_t i = 1; i < sym.size(); ++i) add_bigram((int)i - 1, (int)i);

        while (!q.empty()) {
            const Bigram bg = q.top();
            q.pop();
            Sym& L = sym[bg.left];
            Sym& R = sym[bg.right];
            if (L.n == 0 || R.n == 0 || L.n + R.n != bg.text.size() ||
                word.compare(L.off, L.n + R.n, bg.text) != 0 || L.next != bg.right)
                continue;  // stale
            L.n += R.n;
            R.n = 0;
            L.next = R.next;
            if (R.next >= 0) sym[R.next].prev = bg.left;
            add_bigram(L.prev, bg.left);
            add_bigram(bg.left, L.next);
        }

        for (int i = sym.empty() ? -1 : 0; i >= 0; i = sym[i].next) {
            if (sym[i].n == 0) continue;
            const std::string s = word.substr(sym[i].off, sym[i].n);
            const int id = lookup(s);
            if (id >= 0) { out.push_back(id); continue; }
            for (char ch : s) {  // byte fallback, as llama.cpp: single-byte tokens if the vocab has them
                const int b = lookup(pre == PRE_GEMMA4 ? hex_byte_token((unsigned char)ch) : std::string(1, ch));
                if (b >= 0) out.push_back(b);
            }
        }
    }

    static std::string hex_byte_token(unsigned char ch) {
        static const char* hex = "0123456789ABCDEF";
        return std::string("<0x") + hex[ch >> 4] + hex[ch & 15] + ">";
    }

    // Gemma-4: spaces as U+2581, then runs of newlines and runs of other text, BPE'd as raw UTF-8 (invalid
    // UTF-8 as U+FFFD first, as llama.cpp's codepoint split leaves it)
    void encode_text_gemma4(const std::string& text, std::vector<int>& out) const {
        std::string esc;
        esc.reserve(text.size() + text.size() / 4);
        for (uint32_t c : utf8_to_cpts(text)) {
            if (c == ' ') esc += "\xe2\x96\x81";
            else append_utf8(esc, c);
        }
        for (size_t pos = 0; pos < esc.size();) {
            const bool nl = esc[pos] == '\n';
            size_t end = pos;
            while (end < esc.size() && (esc[end] == '\n') == nl) ++end;
            const std::string word = esc.substr(pos, end - pos);
            pos = end;
            const int whole = nl ? lookup(word) : -1;
            if (whole >= 0) out.push_back(whole);
            else bpe_word(word, out);
        }
    }

    // Pre-tokenize + byte-encode + BPE one span of plain text
    void encode_text(const std::string& text, std::vector<int>& out) const {
        if (pre == PRE_GEMMA4) { encode_text_gemma4(text, out); return; }
        const std::vector<uint32_t> cpts = utf8_to_cpts(text);
        const std::vector<size_t> words = split_qwen(cpts, pre == PRE_QWEN35, pre == PRE_LLAMA3 ? 3 : 1);
        const ByteMap& bm = byte_map();
        size_t start = 0;
        for (size_t len : words) {
            std::string raw, enc;
            for (size_t i = start; i < start + len; ++i) append_utf8(raw, cpts[i]);
            start += len;
            for (unsigned char ch : raw) enc += bm.byte_to_utf8[ch];
            // llama3 sets ignore_merges: a word that is a token is that token, without merging
            const int whole = pre == PRE_LLAMA3 ? lookup(enc) : -1;
            if (whole >= 0) out.push_back(whole);
            else bpe_word(enc, out);
        }
    }

    // Split on special tokens (longest first, as llama.cpp's tokenizer_st_partition), BPE the rest
    void encode(const std::string& text, std::vector<int>& out) const {
        struct Frag { int id; size_t off, len; };  // id >= 0: special token; else text[off, off + len)
        std::vector<Frag> frags = {{-1, 0, text.size()}};
        for (int sid : specials) {
            const std::string& st = vocab[sid];
            if (st.empty()) continue;
            std::vector<Frag> next;
            for (const Frag& f : frags) {
                if (f.id >= 0) { next.push_back(f); continue; }
                size_t pos = f.off;
                const size_t fend = f.off + f.len;
                while (true) {
                    const size_t m = text.find(st, pos);
                    if (m == std::string::npos || m + st.size() > fend) break;
                    if (m > pos) next.push_back({-1, pos, m - pos});
                    next.push_back({sid, m, st.size()});
                    pos = m + st.size();
                }
                if (pos < fend) next.push_back({-1, pos, fend - pos});
            }
            frags.swap(next);
        }
        if (add_bos && bos_id >= 0) out.push_back(bos_id);
        for (const Frag& f : frags) {
            if (f.id >= 0) out.push_back(f.id);
            else encode_text(text.substr(f.off, f.len), out);
        }
    }

    // One token's text: byte-decoded for normal tokens, verbatim for control/user-defined ones
    void piece(int id, std::string& out) const {
        if (id < 0 || id >= (int)vocab.size()) return;
        const std::string& t = vocab[id];
        if (type[id] == TT_CONTROL || type[id] == TT_USER_DEFINED || type[id] == TT_UNKNOWN) { out += t; return; }
        if (pre == PRE_GEMMA4) {
            if (type[id] == TT_BYTE && t.size() == 6 && t.compare(0, 3, "<0x") == 0) {
                out += (char)std::stoi(t.substr(3, 2), nullptr, 16);
                return;
            }
            for (size_t i = 0; i < t.size();) {  // U+2581 back to spaces
                if (t.compare(i, 3, "\xe2\x96\x81") == 0) { out += ' '; i += 3; }
                else out += t[i++];
            }
            return;
        }
        const ByteMap& bm = byte_map();
        for (uint32_t c : utf8_to_cpts(t)) {
            auto it = bm.cpt_to_byte.find(c);
            if (it != bm.cpt_to_byte.end()) out += (char)it->second;
            else append_utf8(out, c);
        }
    }
};

extern "C" {

tokenizer_t* tokenizer_create(const char** vocab, size_t vocab_size, const char** merges, size_t merges_size) {
    if (!vocab || vocab_size == 0) return nullptr;
    tokenizer* t = new tokenizer();
    t->vocab.reserve(vocab_size);
    t->vocab_map.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        t->vocab.emplace_back(vocab[i] ? vocab[i] : "");
        t->vocab_map.emplace(t->vocab.back(), (int)i);
    }
    t->type.assign(vocab_size, TT_NORMAL);
    t->merge_rank.reserve(merges_size);
    for (size_t i = 0; i < merges_size; ++i) {
        // "left right"; split at the first space after position 0, as llama.cpp does
        const char* s = merges[i];
        const char* sp = s && s[0] ? strchr(s + 1, ' ') : nullptr;
        if (!sp) continue;
        t->merge_rank.emplace(std::string(s, sp - s) + " " + std::string(sp + 1), (int)i);
    }
    return t;
}

void tokenizer_free(tokenizer_t* tok) {
    delete tok;
}

int tokenizer_encode(tokenizer_t* tok, const char* text, int* out_ids, int max_ids) {
    if (!tok || !text || !out_ids || max_ids < 0) return -1;
    std::vector<int> ids;
    tok->encode(text, ids);
    if ((int)ids.size() > max_ids) return -1;
    std::copy(ids.begin(), ids.end(), out_ids);
    return (int)ids.size();
}

int tokenizer_decode(tokenizer_t* tok, const int* ids, int n_ids, char* out_str, int max_len) {
    if (!tok || !out_str || max_len <= 0 || (n_ids > 0 && !ids)) return -1;
    std::string s;
    for (int i = 0; i < n_ids; ++i) tok->piece(ids[i], s);
    int len = (int)s.size();
    if (len >= max_len) len = max_len - 1;
    memcpy(out_str, s.data(), (size_t)len);
    out_str[len] = '\0';
    return len;
}

int tokenizer_detokenize(tokenizer_t* tok, const int* ids, int n_ids, char* out_str, int max_len) {
    if (!tok || (n_ids > 0 && !ids) || max_len < 0 || (max_len > 0 && !out_str)) return -1;
    std::string s;
    for (int i = 0; i < n_ids; ++i) tok->piece(ids[i], s);
    if (max_len > 0) {
        const size_t len = std::min(s.size(), (size_t)max_len - 1);
        memcpy(out_str, s.data(), len);
        out_str[len] = '\0';
    }
    return (int)s.size();
}

size_t tokenizer_vocab_size(tokenizer_t* tok) { return tok ? tok->vocab.size() : 0; }
int tokenizer_bos_id(tokenizer_t* tok) { return tok ? tok->bos_id : -1; }
int tokenizer_eos_id(tokenizer_t* tok) { return tok ? tok->eos_id : -1; }
int tokenizer_pad_id(tokenizer_t* tok) { return tok ? tok->pad_id : -1; }
int tokenizer_unk_id(tokenizer_t* tok) { return tok ? tok->unk_id : -1; }
int tokenizer_is_eog(tokenizer_t* tok, int id) { return tok && tok->is_eog(id) ? 1 : 0; }

// Integer KV by key (any integer GGUF type), or def
static int64_t kv_int(const gguf_model_t* m, const char* key, int64_t def) {
    const int64_t idx = gguf_find_kv(m, key);
    if (idx < 0) return def;
    uint32_t u32; int32_t i32; uint64_t u64; int64_t i64;
    if (gguf_kv_get_u32(m, idx, &u32) == 0) return u32;
    if (gguf_kv_get_i32(m, idx, &i32) == 0) return i32;
    if (gguf_kv_get_u64(m, idx, &u64) == 0) return (int64_t)u64;
    if (gguf_kv_get_i64(m, idx, &i64) == 0) return i64;
    return def;
}

static bool kv_bool(const gguf_model_t* m, const char* key, bool def) {
    const int64_t idx = gguf_find_kv(m, key);
    uint8_t v;
    return (idx >= 0 && gguf_kv_get_u8(m, idx, &v) == 0) ? v != 0 : def;
}

tokenizer_t* tokenizer_load_from_gguf(void* gguf_model) {
    const gguf_model_t* m = (const gguf_model_t*)gguf_model;
    if (!m) return nullptr;

    const char* model = gguf_tokenizer_type(m);
    const bool gemma4 = model && strcmp(model, "gemma4") == 0;
    if (!model || (strcmp(model, "gpt2") != 0 && !gemma4)) {
        INFER_LOG("[tokenizer] unsupported tokenizer.ggml.model '%s' (gpt2 BPE, gemma4)\n", model ? model : "");
        return nullptr;
    }
    const char* pre = gguf_pre_type(m);
    PreType pt;
    if (gemma4) pt = PRE_GEMMA4;
    else if (pre && strcmp(pre, "qwen35") == 0) pt = PRE_QWEN35;
    else if (pre && strcmp(pre, "qwen2") == 0) pt = PRE_QWEN2;
    else if (pre && (!strcmp(pre, "llama3") || !strcmp(pre, "llama-v3") || !strcmp(pre, "llama-bpe") ||
                     !strcmp(pre, "pixtral")))
        pt = PRE_LLAMA3;
    else {
        INFER_LOG("[tokenizer] unsupported tokenizer.ggml.pre '%s' (supported: qwen35, qwen2, llama3, pixtral)\n",
                  pre ? pre : "");
        return nullptr;
    }

    const char** vocab = nullptr;
    size_t n_vocab = 0;
    if (gguf_kv_get_string_array(m, "tokenizer.ggml.tokens", &vocab, &n_vocab) != 0 || n_vocab == 0) return nullptr;
    const char** merges = nullptr;
    size_t n_merges = 0;
    gguf_kv_get_string_array(m, "tokenizer.ggml.merges", &merges, &n_merges);

    tokenizer* t = tokenizer_create(vocab, n_vocab, merges, n_merges);
    if (!t) return nullptr;
    t->pre = pt;

    int32_t at = -1;
    const void* types = nullptr;
    size_t n_types = 0;
    if (gguf_kv_get_array(m, "tokenizer.ggml.token_type", &at, &types, &n_types) == 0 && n_types == n_vocab &&
        (at == 5 /* INT32 */ || at == 4 /* UINT32 */)) {
        for (size_t i = 0; i < n_vocab; ++i) {
            int32_t v;
            memcpy(&v, (const char*)types + i * 4, 4);
            t->type[i] = v;
        }
    }
    t->bos_id = (int)kv_int(m, "tokenizer.ggml.bos_token_id", -1);
    t->eos_id = (int)kv_int(m, "tokenizer.ggml.eos_token_id", -1);
    t->pad_id = (int)kv_int(m, "tokenizer.ggml.padding_token_id", -1);
    t->unk_id = (int)kv_int(m, "tokenizer.ggml.unknown_token_id", -1);
    // llama.cpp forces BOS for gemma4; llama3 / pixtral default to it unless the GGUF says otherwise
    t->add_bos = kv_bool(m, "tokenizer.ggml.add_bos_token", pt == PRE_LLAMA3) || pt == PRE_GEMMA4;
    std::vector<int> eog_ids = {t->eos_id};
    for (const char* key : {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id", "tokenizer.ggml.fim_pad_token_id",
                            "tokenizer.ggml.fim_rep_token_id", "tokenizer.ggml.fim_sep_token_id"})
        eog_ids.push_back((int)kv_int(m, key, -1));
    t->build_eog(eog_ids);
    t->build_specials();  // after build_eog, which can turn a special token into a normal one
    return t;
}

} // extern "C"
