#!/usr/bin/env python3
"""Record reference embeddings for tests/embed_verify (EmbeddingGemma).

Runs llama.cpp's llama-embedding (the reference; normalized as llama-server's /v1/embeddings) on each text below
and writes one line per text: <text as hex bytes> TAB <space-separated floats>.

    tools/gen_embed_ref.py llama.cpp/build-cpu/bin/llama-embedding /models/embeddinggemma-300M-Q8_0.gguf \\
        > refdata/embeddinggemma_embed.tsv
"""

import json
import subprocess
import sys

LONG = " ".join(f"Line {i}: the parser reads a token, checks its kind and pushes it on the stack." for i in range(90))

TEXTS = [
    "def add(a, b):\n    return a + b",
    "def sum_two(x, y):\n    return x + y",
    "How do I add two numbers in Python?",
    "The weather in Ghent is rainy today.",
    "static int sync_prompt(inference_model_t* m, const int32_t* p, int n, int tail) {",
    "café naïve résumé Über straße 你好，世界",
    "x",
    LONG,
]
SEP = "<#sep#>"


def main():
    exe, model = sys.argv[1], sys.argv[2]
    out = subprocess.run(
        [exe, "-m", model, "-p", SEP.join(TEXTS), "--embd-separator", SEP, "--embd-output-format", "json",
         "--embd-normalize", "2", "-c", "2048", "-b", "2048", "-ub", "2048", "-t", "16"],
        check=True, capture_output=True, text=True).stdout
    data = json.loads(out[out.index("{"):])["data"]
    assert len(data) == len(TEXTS), (len(data), len(TEXTS))
    for text, d in zip(TEXTS, data):
        print(text.encode().hex() + "\t" + " ".join(f"{v:.7g}" for v in d["embedding"]))


if __name__ == "__main__":
    main()
