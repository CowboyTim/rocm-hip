#!/usr/bin/env python3
# gguf_nvfp4_scales.py -- give every NVFP4 weight of a GGUF a global output scale, as ModelOpt NVFP4 checkpoints
# carry (llama.cpp's <weight>.scale: F32 [1], or [n_expert] for stacked experts), without changing the model.
#
#   gguf_nvfp4_scales.py <in.gguf> <out.gguf>
#
# Each tensor (each expert of a 3-D one) gets .scale = 2^-k while its UE4M3 block scales are multiplied by 2^k: in
# UE4M3 exactly (subnormals become normals for k >= 3; k is chosen so the largest exponent stays <= 14, short of the
# 0x7f zero sentinel), and in the float products of the dot products as well, so the scaled file computes the same
# outputs bit for bit as the input file, in llama.cpp and in the engine. For testing the engine's scale support
# against llama.cpp and against the unscaled model. Pure Python.

import struct, sys

NVFP4, F32 = 40, 0
BLOCK, VALUES = 36, 64


def ue4m3_times(k):
    """Byte map: a UE4M3 scale byte (sign ignored, as ggml) -> the byte of its value times 2^k (k >= 3)."""
    t = bytearray(256)
    for b in range(256):
        x = b & 0x7F
        if x == 0 or x == 0x7F:
            t[b] = x
            continue
        ex, m = x >> 3, x & 7
        if ex == 0:  # m * 2^-9 -> a normal (8 + m') * 2^(ex' - 10)
            v, e = m, -9 + k
            while v < 8:
                v, e = v * 2, e - 1
            ex2, m2 = e + 10, v - 8
        else:
            ex2, m2 = ex + k, m
        t[b] = (ex2 << 3) | m2 if ex2 <= 14 else 0xFF  # (not in the tensor: k <= 14 - its largest exponent)
    return bytes(t)


EXP_OF = bytes(((b & 0x7F) >> 3) if (b & 0x7F) != 0x7F else 0 for b in range(256))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    f = open(src, "rb")
    head = f.read(1 << 24)
    pos = 0

    def take(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, head, pos)
        pos += struct.calcsize("<" + fmt)
        return v[0]

    def take_str():
        nonlocal pos
        n = take("Q")
        s = head[pos:pos + n]
        pos += n
        return s

    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    align = 32

    def skip_val(t):
        nonlocal pos
        if t == 8:
            take_str()
        elif t == 9:
            at, cnt = take("I"), take("Q")
            for _ in range(cnt):
                skip_val(at)
        else:
            pos += size[t]

    assert head[:4] == b"GGUF"
    pos = 4
    version, n_t, n_kv = take("I"), take("Q"), take("Q")
    kv_start = pos
    for _ in range(n_kv):
        key = take_str()
        t = take("I")
        if key == b"general.alignment" and t == 4:
            align = struct.unpack_from("<I", head, pos)[0]
        skip_val(t)
    kv_bytes = head[kv_start:pos]
    tensors = []
    for _ in range(n_t):
        name = take_str()
        nd = take("I")
        ne = [take("Q") for _ in range(nd)]
        tensors.append([name, ne, take("I"), take("Q")])
    data0 = (pos + align - 1) // align * align

    def nbytes(ne, t):
        n = 1
        for d in ne:
            n *= d
        return n // VALUES * BLOCK if t == NVFP4 else None

    f.seek(0, 2)
    end = f.tell() - data0  # the data section's size

    # per NVFP4 weight: k for each expert (or the whole tensor), from its largest exponent
    plan = {}
    names = {t[0] for t in tensors}
    for name, ne, t, off in tensors:
        # (as llama.cpp reads them: the projections and the LM head, not the token embeddings nor the MTP's eh_proj)
        if (t != NVFP4 or not name.endswith(b".weight") or name[:-7] + b".scale" in names or
                name == b"token_embd.weight" or b".nextn." in name):
            continue
        n_exp = ne[2] if len(ne) == 3 else 1
        per = nbytes(ne, t) // n_exp
        ks = []
        for e in range(n_exp):
            f.seek(data0 + off + e * per)
            buf = f.read(per)
            mx = max(max(buf[j::BLOCK].translate(EXP_OF)) for j in range(4))
            ks.append(min(6, 14 - mx))
        if min(ks) < 3:
            print(f"{name.decode()}: largest exponent too high, left without a scale", file=sys.stderr)
            continue
        plan[name] = (n_exp, per, ks)

    # the new tensors: <name>.scale F32 [n_exp], appended after the data
    new = []
    off = (end + align - 1) // align * align
    for name, (n_exp, per, ks) in plan.items():
        new.append((name[:-7] + b".scale", [n_exp] if n_exp > 1 else [1], off,
                    struct.pack(f"<{n_exp}f", *[2.0 ** -k for k in ks])))
        off = (off + 4 * n_exp + align - 1) // align * align

    out = open(dst, "wb")
    hdr = bytearray(b"GGUF" + struct.pack("<IQQ", version, n_t + len(new), n_kv) + kv_bytes)
    for name, ne, t, o in tensors:
        hdr += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
        hdr += struct.pack("<IQ", t, o)
    for name, ne, o, _ in new:
        hdr += struct.pack("<Q", len(name)) + name + struct.pack("<IQ", 1, ne[0]) + struct.pack("<IQ", F32, o)
    hdr += b"\0" * ((-len(hdr)) % align)
    out.write(hdr)

    # the data: NVFP4 weights with their scale bytes mapped, the rest as it is
    by_off = sorted(tensors, key=lambda x: x[3])
    f.seek(data0)
    cur = 0
    for name, ne, t, o in by_off:
        if o > cur:
            out.write(f.read(o - cur))
            cur = o
        if name not in plan:
            continue
        n_exp, per, ks = plan[name]
        for e in range(n_exp):
            buf = bytearray(f.read(per))
            tab = ue4m3_times(ks[e])
            for j in range(4):
                buf[j::BLOCK] = bytes(buf[j::BLOCK]).translate(tab)
            out.write(buf)
        cur += n_exp * per
    while cur < end:
        chunk = f.read(min(1 << 26, end - cur))
        out.write(chunk)
        cur += len(chunk)
    for name, ne, o, blob in new:
        out.write(b"\0" * (o - cur))
        out.write(blob)
        cur = o + len(blob)
    out.close()
    print(f"{len(new)} scales added", file=sys.stderr)


if __name__ == "__main__":
    main()
