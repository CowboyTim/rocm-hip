#!/usr/bin/env python3
"""Record reference token IDs for tests/tokenizer_gguf_verify (Phase 20, tokenizer fix).

Runs `build/ref_dump tokens <gguf> <prompt>` (llama.cpp, the reference) on each prompt below and writes
one line per prompt: <prompt as hex bytes> TAB <space-separated token ids>. Hex keeps CR/LF, tabs, and
invalid UTF-8 intact.

    tools/gen_tokenizer_ref.py build/ref_dump /dev/shm/models/Qwen3.5-4B-ROCMFP4.gguf > tests/data/qwen35_tokenizer.tsv
"""

import subprocess
import sys

PROMPTS = [
    b"The capital of France is",
    b"Hello, world!",
    b"  leading spaces and trailing spaces   ",
    b"I'm sure they'll say it's fine; we've done what we'd planned. DON'T SHOUT, I'D SAY",
    b"Numbers: 12345 3.14159 -42 1,000,000 2026-10-16",
    b"line one\nline two\r\nline three\n\n\nend",
    b"tabs\tand\t\tmixed \t whitespace\x0b\x0c",
    b"caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9 \xc3\x9cber stra\xc3\x9fe",
    b"combining: e\xcc\x81 a\xcc\x8a n\xcc\x83o",
    b"\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81 \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf",
    b"\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80 \xce\x93\xce\xb5\xce\xb9\xce\xac \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7",
    b"emoji \xf0\x9f\x98\x80\xf0\x9f\x9a\x80 and flags \xf0\x9f\x87\xb3\xf0\x9f\x87\xb1",
    b"def f(x):\n    return x ** 2  # square\n\nprint(f(3))",
    b"<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    b"a<|endoftext|>b <|im_end|>",
    b"!!!??? ... --- *** ((( )))",
    b"URL: https://example.com/path?q=1&r=two#frag",
    b"invalid utf8: \xff\xfe \xc3 end",
    b"x",
    b" ",
    b"\n",
]


def main():
    ref_dump, gguf = sys.argv[1], sys.argv[2]
    for p in PROMPTS:
        ids = subprocess.run([ref_dump, "tokens", gguf, p], check=True, capture_output=True).stdout.decode().strip()
        sys.stdout.write("%s\t%s\n" % (p.hex(), ids))


if __name__ == "__main__":
    main()
