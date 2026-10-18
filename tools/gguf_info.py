#!/usr/bin/env python3
# gguf_info.py -- a GGUF's metadata and tensor table, grouped by tensor name across layers.
#
#   gguf_info.py <file.gguf> [full]
#
# Reads only the header, so a partial download (the first megabytes of a file) works too. Long arrays and
# strings are abbreviated unless "full" is given. Pure Python.

import collections, re, struct, sys

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
         11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
         19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
         27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 100: "ROCMFP4",
         101: "ROCMFP4_FAST", 102: "Q6_0_ROCMFPX", 103: "Q8_0_ROCMFPX", 104: "Q3_0_ROCMFPX", 107: "Q2_0_ROCMFPX",
         108: "Q4_0_ROCMI4"}


def main(path, full):
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(rd("Q")).decode("utf-8", "replace")
    assert f.read(4) == b"GGUF", "not a GGUF file"
    ver, nt, nkv = rd("I"), rd("Q"), rd("Q")
    scal = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def val(t):
        if t == 8:
            return rs()
        if t == 9:
            at, n = rd("I"), rd("Q")
            return [val(at) for _ in range(n)]
        return rd(scal[t])
    print(f"version {ver}, {nt} tensors, {nkv} kv")
    for _ in range(nkv):
        k = rs()
        v = val(rd("I"))
        if isinstance(v, list) and len(v) > 16 and not full:
            v = f"[{len(v)} items] {v[:8]}..."
        elif isinstance(v, str) and len(v) > 200 and not full:
            v = f"<{len(v)} chars> {v[:120]!r}..."
        print(f"  {k} = {v}")
    groups = collections.OrderedDict()
    for _ in range(nt):
        name = rs()
        dims = [rd("Q") for _ in range(rd("I"))]
        t = rd("I")
        rd("Q")
        g = groups.setdefault(re.sub(r"blk\.\d+\.", "blk.N.", name),
                              {"dims": dims, "types": collections.Counter(), "layers": []})
        g["types"][TYPES.get(t, t)] += 1
        m = re.match(r"blk\.(\d+)\.", name)
        if m:
            g["layers"].append(int(m.group(1)))
    for k, g in groups.items():
        ls = g["layers"]
        lr = f" layers {min(ls)}..{max(ls)} ({len(ls)})" if ls else ""
        print(f"  {k:40s} {str(g['dims']):28s} {dict(g['types'])}{lr}")


main(sys.argv[1], len(sys.argv) > 2)
