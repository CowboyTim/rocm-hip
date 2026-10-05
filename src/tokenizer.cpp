// tokenizer.cpp -- Phase 5: minimal BPE tokenizer for GGUF vocab+merges.
// Implements encode/decode using the merge rules as in GPT-2 / LLaMA tokenizers.

#include "tokenizer.h"
#include "gguf_mmap.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <climits>

struct Tokenizer {
    std::vector<std::string> vocab;                 // id -> token string
    std::unordered_map<std::string, int> vocab_map; // token string -> id
    std::vector<std::pair<std::string,std::string>> merges; // ordered merges
    std::unordered_map<std::string, int> merge_rank; // "a b" -> rank (lower = earlier)
    int bos_id = -1;
    int eos_id = -1;
    int pad_id = -1;
    int unk_id = -1;
    // special token maps
    std::unordered_map<std::string, int> special_str_to_id;
    std::unordered_map<int, std::string> id_to_special_str;
};

extern "C" {

tokenizer_t* tokenizer_create(const char** vocab, size_t vocab_size,
                              const char** merges, size_t merges_size)
{
    Tokenizer* t = new Tokenizer();
    t->vocab.reserve(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) {
        t->vocab.emplace_back(vocab[i]);
        t->vocab_map[t->vocab.back()] = (int)i;
    }
    t->merges.reserve(merges_size);
    for (size_t i = 0; i < merges_size; ++i) {
        const char* s = merges[i];
        // format: "token1 token2"
        const char* space = strchr(s, ' ');
        if (!space) continue;
        std::string left(s, space - s);
        std::string right(space + 1);
        t->merges.emplace_back(left, right);
        t->merge_rank[left + " " + right] = (int)t->merges.size() - 1;
    }
    return (tokenizer_t*)t;
}

void tokenizer_free(tokenizer_t* tok) {
    delete (Tokenizer*)tok;
}

static std::vector<std::string> split_utf8(const std::string& str) {
    // Very simple: treat each byte as a character for now (ASCII only).
    // For real UTF-8 need proper decoding; omitted for brevity.
    std::vector<std::string> out;
    for (unsigned char c : str) {
        char buf[2] = { (char)c, 0 };
        out.emplace_back(buf);
    }
    return out;
}

int tokenizer_encode(tokenizer_t* tok, const char* text, int* out_ids, int max_ids) {
    Tokenizer* t = (Tokenizer*)tok;
    if (!t || !text || !out_ids || max_ids <= 0) return -1;

    // Check for special token strings that map to a single token ID
    auto it_special = t->special_str_to_id.find(text);
    if (it_special != t->special_str_to_id.end()) {
        if (max_ids >= 1) {
            out_ids[0] = it_special->second;
            return 1;
        }
        return -1;
    }

    // Initial tokenization: split into bytes (or use vocab tokens if present)
    std::string input(text);
    std::vector<std::string> tokens = split_utf8(input);

    // Map initial tokens to ids (fallback to byte fallback if unknown)
    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const auto& tok_str : tokens) {
        auto it = t->vocab_map.find(tok_str);
        if (it != t->vocab_map.end()) ids.push_back(it->second);
        else {
            // byte fallback: use 256..511 for bytes 0..255
            unsigned char b = (unsigned char)tok_str[0];
            ids.push_back(256 + b);
        }
    }

    // Apply BPE merges in order
    bool changed = true;
    while (changed) {
        changed = false;
        int best_rank = INT_MAX;
        int best_idx = -1;
        for (size_t i = 0; i + 1 < ids.size(); ++i) {
            std::string pair = t->vocab[ids[i]] + " " + t->vocab[ids[i+1]];
            auto it = t->merge_rank.find(pair);
            if (it != t->merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = (int)i;
            }
        }
        if (best_idx >= 0) {
            // merge
            std::string merged = t->vocab[ids[best_idx]] + t->vocab[ids[best_idx+1]];
            int merged_id = -1;
            auto itv = t->vocab_map.find(merged);
            if (itv != t->vocab_map.end()) merged_id = itv->second;
            else {
                // new token not in vocab -> assign new id (extend vocab)
                merged_id = (int)t->vocab.size();
                t->vocab.push_back(merged);
                t->vocab_map[merged] = merged_id;
            }
            ids[best_idx] = merged_id;
            ids.erase(ids.begin() + best_idx + 1);
            changed = true;
        }
    }

    int n = (int)ids.size();
    if (n > max_ids) n = max_ids;
    for (int i = 0; i < n; ++i) out_ids[i] = ids[i];
    return n;
}

int tokenizer_decode(tokenizer_t* tok, const int* ids, int n_ids, char* out_str, int max_len) {
    Tokenizer* t = (Tokenizer*)tok;
    if (!t || !ids || n_ids <= 0 || !out_str || max_len <= 0) return -1;
    std::string result;
    result.reserve(n_ids * 4);
    for (int i = 0; i < n_ids; ++i) {
        int id = ids[i];
        // Check for special token ID
        auto it_special = t->id_to_special_str.find(id);
        if (it_special != t->id_to_special_str.end()) {
            result += it_special->second;
        } else if (id >= 0 && id < (int)t->vocab.size()) {
            result += t->vocab[id];
        } else if (id >= 256 && id < 512) {
            // byte fallback
            char c = (char)(id - 256);
            result.push_back(c);
        }
    }
    int len = (int)result.size();
    if (len >= max_len) len = max_len - 1;
    memcpy(out_str, result.c_str(), len);
    out_str[len] = '\0';
    return len;
}

size_t tokenizer_vocab_size(tokenizer_t* tok) {
    Tokenizer* t = (Tokenizer*)tok;
    return t ? t->vocab.size() : 0;
}

int tokenizer_bos_id(tokenizer_t* tok) {
    Tokenizer* t = (Tokenizer*)tok;
    return t ? t->bos_id : -1;
}

int tokenizer_eos_id(tokenizer_t* tok) {
    Tokenizer* t = (Tokenizer*)tok;
    return t ? t->eos_id : -1;
}

int tokenizer_pad_id(tokenizer_t* tok) {
    Tokenizer* t = (Tokenizer*)tok;
    return t ? t->pad_id : -1;
}

int tokenizer_unk_id(tokenizer_t* tok) {
    Tokenizer* t = (Tokenizer*)tok;
    return t ? t->unk_id : -1;
}

// Helper to read a uint32 scalar from GGUF KV by key using public API
static int gguf_kv_get_u32_by_key(const gguf_model_t* model, const char* key, uint32_t* out) {
    if (!model || !key || !out) return -1;
    int64_t n_kv = gguf_n_kv(model);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* k = gguf_kv_key(model, i);
        if (k && strcmp(k, key) == 0) {
            int type = gguf_kv_type(model, i);
            if (type == 4) { // GGUF_TYPE_UINT32
                if (gguf_kv_get_u32(model, i, out) == 0) return 0;
            } else if (type == 5) { // GGUF_TYPE_INT32
                if (gguf_kv_get_i32(model, i, (int32_t*)out) == 0) return 0;
            } else if (type == 10) { // GGUF_TYPE_UINT64
                if (gguf_kv_get_u64(model, i, (uint64_t*)out) == 0) return 0;
            } else if (type == 11) { // GGUF_TYPE_INT64
                if (gguf_kv_get_i64(model, i, (int64_t*)out) == 0) return 0;
            }
            return -1;
        }
    }
    return -1;
}

// Load tokenizer from GGUF model metadata (vocab, merges, added_tokens)
// Returns NULL on failure.
tokenizer_t* tokenizer_load_from_gguf(void* gguf_model) {
    if (!gguf_model) return nullptr;
    // Get vocab array
    const char** vocab_ptrs = nullptr;
    size_t vocab_count = 0;
    if (gguf_kv_get_string_array((const gguf_model_t*)gguf_model, "tokenizer.ggml.tokens", &vocab_ptrs, &vocab_count) != 0 || vocab_count == 0) {
        return nullptr;
    }
    // Get merges array
    const char** merges_ptrs = nullptr;
    size_t merges_count = 0;
    gguf_kv_get_string_array((const gguf_model_t*)gguf_model, "tokenizer.ggml.merges", &merges_ptrs, &merges_count);
    // Create tokenizer using existing create function
    tokenizer_t* tok = tokenizer_create(vocab_ptrs, vocab_count, merges_ptrs, merges_count);
    if (!tok) return nullptr;
    Tokenizer* t = (Tokenizer*)tok;
    // Try to read special token IDs from GGUF metadata (common keys)
    uint32_t val;
    if (gguf_kv_get_u32_by_key((const gguf_model_t*)gguf_model, "tokenizer.ggml.bos_token_id", &val) == 0) t->bos_id = (int)val;
    if (gguf_kv_get_u32_by_key((const gguf_model_t*)gguf_model, "tokenizer.ggml.eos_token_id", &val) == 0) t->eos_id = (int)val;
    if (gguf_kv_get_u32_by_key((const gguf_model_t*)gguf_model, "tokenizer.ggml.pad_token_id", &val) == 0) t->pad_id = (int)val;
    if (gguf_kv_get_u32_by_key((const gguf_model_t*)gguf_model, "tokenizer.ggml.unk_token_id", &val) == 0) t->unk_id = (int)val;
    // Try to read added_tokens array (strings) and map known special tokens
    const char** added_ptrs = nullptr;
    size_t added_count = 0;
    if (gguf_kv_get_added_tokens((const gguf_model_t*)gguf_model, "tokenizer.ggml.added_tokens", &added_ptrs, &added_count) == 0) {
        for (size_t i = 0; i < added_count; ++i) {
            const char* s = added_ptrs[i];
            if (!s) continue;
            auto it = t->vocab_map.find(s);
            if (it != t->vocab_map.end()) {
                int id = it->second;
                // Map known special token strings
                if (strcmp(added_ptrs[i], "<|endoftext|>") == 0) {
                    t->eos_id = id;
                    t->special_str_to_id[s] = id;
                    t->id_to_special_str[id] = s;
                } else if (strcmp(added_ptrs[i], "<|pad|>") == 0) {
                    t->pad_id = id;
                    t->special_str_to_id[s] = id;
                    t->id_to_special_str[id] = s;
                } else if (strcmp(added_ptrs[i], "<|unk|>") == 0) {
                    t->unk_id = id;
                    t->special_str_to_id[s] = id;
                    t->id_to_special_str[id] = s;
                } else if (strcmp(added_ptrs[i], "<|bos|>") == 0) {
                    t->bos_id = id;
                    t->special_str_to_id[s] = id;
                    t->id_to_special_str[id] = s;
                }
            }
        }
    }
    // Populate special token maps from IDs we have
    if (t->bos_id >= 0) {
        // find string for bos_id
        if ((size_t)t->bos_id < t->vocab.size()) {
            const std::string& s = t->vocab[t->bos_id];
            t->special_str_to_id[s] = t->bos_id;
            t->id_to_special_str[t->bos_id] = s;
        }
    }
    if (t->eos_id >= 0) {
        if ((size_t)t->eos_id < t->vocab.size()) {
            const std::string& s = t->vocab[t->eos_id];
            t->special_str_to_id[s] = t->eos_id;
            t->id_to_special_str[t->eos_id] = s;
        }
    }
    if (t->pad_id >= 0) {
        if ((size_t)t->pad_id < t->vocab.size()) {
            const std::string& s = t->vocab[t->pad_id];
            t->special_str_to_id[s] = t->pad_id;
            t->id_to_special_str[t->pad_id] = s;
        }
    }
    if (t->unk_id >= 0) {
        if ((size_t)t->unk_id < t->vocab.size()) {
            const std::string& s = t->vocab[t->unk_id];
            t->special_str_to_id[s] = t->unk_id;
            t->id_to_special_str[t->unk_id] = s;
        }
    }
    return tok;
}

} // extern "C"