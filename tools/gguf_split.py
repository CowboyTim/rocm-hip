#!/usr/bin/env python3
# gguf_split.py -- split a GGUF into n files the way llama.cpp's gguf-split does, for testing split loading.
#
#   gguf_split.py <in.gguf> <out_prefix> <n>   -> <out_prefix>-00001-of-0000n.gguf ...
#
# The first file keeps all metadata; every file gets split.no (u16), split.count (u16) and split.tensors.count
# (i32); the tensors are dealt out in order, about the same number per file. Pure Python.

import struct, sys


def main():
    src, prefix, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    f = open(src, "rb")
    data = f.read()
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0]

    def skip_str():
        nonlocal pos
        pos += 8 + struct.unpack_from("<Q", data, pos)[0]

    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip_val(t):
        nonlocal pos
        if t == 8:
            skip_str()
        elif t == 9:
            at, cnt = take("I"), take("Q")
            for _ in range(cnt):
                skip_val(at)
        else:
            pos += size[t]

    assert data[:4] == b"GGUF"
    pos = 4
    take("I")
    nt, nkv = take("Q"), take("Q")
    kvs, align = [], 32
    for _ in range(nkv):
        start = pos
        klen = struct.unpack_from("<Q", data, pos)[0]
        key = data[pos + 8:pos + 8 + klen].decode()
        skip_str()
        t = take("I")
        vstart = pos
        skip_val(t)
        if key == "general.alignment":
            align = struct.unpack_from("<I", data, vstart)[0]
        if not key.startswith("split."):
            kvs.append(data[start:pos])
    tensors = []
    for _ in range(nt):
        nlen = struct.unpack_from("<Q", data, pos)[0]
        name = data[pos + 8:pos + 8 + nlen]
        skip_str()
        nd = take("I")
        dims = [take("Q") for _ in range(nd)]
        gt, off = take("I"), take("Q")
        tensors.append([name, dims, gt, off])
    base = (pos + align - 1) // align * align
    offs = sorted(t[3] for t in tensors) + [len(data) - base]
    for t in tensors:  # each tensor's bytes run to the next tensor's offset (padding included)
        t.append(data[base + t[3]:base + offs[offs.index(t[3]) + 1]])

    def s(x):
        return struct.pack("<Q", len(x)) + x

    per = (nt + n - 1) // n
    for i in range(n):
        part = tensors[i * per:(i + 1) * per]
        meta = kvs if i == 0 else []
        split = [s(b"split.no") + struct.pack("<IH", 2, i), s(b"split.count") + struct.pack("<IH", 2, n),
                 s(b"split.tensors.count") + struct.pack("<Ii", 5, nt)]
        out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(part), len(meta) + 3))
        for kv in list(meta) + split:
            out += kv
        off = 0
        for name, dims, gt, _, blob in part:
            out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            out += struct.pack("<IQ", gt, off)
            off += (len(blob) + align - 1) // align * align
        out += b"\0" * (-len(out) % align)
        path = f"{prefix}-{i + 1:05d}-of-{n:05d}.gguf"
        with open(path, "wb") as o:
            o.write(out)
            for _, _, _, _, blob in part:
                o.write(blob)
                o.write(b"\0" * (-len(blob) % align))
        print(path, len(part), "tensors")


main()
