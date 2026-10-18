#!/usr/bin/env python3
# gen_qwen4exp_tiny.py -- a small random-weight qwen4exp (Qwen3.8-Flash-Next) GGUF for the oracle tests (Phase 21).
#
#   gen_qwen4exp_tiny.py <real_header.gguf> <out.gguf> [seed]
#
# The real model (98.5 GiB) does not fit here, so the architecture is verified on this file instead: the same
# tensor names, layer pattern, quant types per tensor and tokenizer as the release GGUF, with small widths. The
# tokenizer metadata (and the PLE hash multipliers / EOS ids) are copied from the first megabytes of the real
# first shard (curl -r 0-33554431 ...), which hold its whole header. Weights are random blocks scaled to about
# 1/sqrt(K), so activations stay O(1) through the layers. Pure Python (no numpy): its own GGUF reader / writer.

import math, os, random, struct, sys

# ---------------------------------------------------------------- GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCAL = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}


def read_header_kv(path):
    """The metadata of a GGUF (only the header is needed), as [(key, type, value, subtype)]."""
    f = open(path, "rb")
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    rs = lambda: f.read(rd("Q")).decode("utf-8")
    assert f.read(4) == b"GGUF"
    rd("I"); rd("Q"); nkv = rd("Q")
    out = []
    for _ in range(nkv):
        k = rs(); t = rd("I")
        if t == ARR:
            st, n = rd("I"), rd("Q")
            v = [rs() if st == STR else rd(SCAL[st]) for _ in range(n)]
            out.append((k, t, v, st))
        else:
            out.append((k, t, rs() if t == STR else rd(SCAL[t]), None))
    return out


class Writer:
    def __init__(self):
        self.kv = []       # (key, type, value, subtype)
        self.tensors = []  # (name, dims, ggml type, bytes)

    def add(self, key, t, v, st=None):
        self.kv.append((key, t, v, st))

    def tensor(self, name, dims, gtype, data):
        self.tensors.append((name, dims, gtype, bytes(data)))

    def write(self, path, align=32):
        def s(x):
            b = x.encode("utf-8")
            return struct.pack("<Q", len(b)) + b

        def val(t, v):
            return s(v) if t == STR else struct.pack("<" + SCAL[t], v)
        out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
        for k, t, v, st in self.kv:
            out += s(k) + struct.pack("<I", t)
            if t == ARR:
                out += struct.pack("<IQ", st, len(v)) + b"".join(val(st, x) for x in v)
            else:
                out += val(t, v)
        off = 0
        for name, dims, gtype, data in self.tensors:
            out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            out += struct.pack("<IQ", gtype, off)
            off += (len(data) + align - 1) // align * align
        out += b"\0" * (-len(out) % align)
        with open(path, "wb") as f:
            f.write(out)
            for _, _, _, data in self.tensors:
                f.write(data)
                f.write(b"\0" * (-len(data) % align))


# ---------------------------------------------------------------- random quantized blocks
T_F32, T_F16, T_Q5_1, T_Q5_K, T_Q6_K, T_FP4, T_FP4F = 0, 1, 7, 13, 14, 100, 101
rng = random.Random(1)


def f16(x):
    return struct.pack("<e", x)


def ue4m3_byte(d):
    """The ROCmFP4 scale byte whose half-scale value is closest to d (ue4m3_half in quant_types.cpp)."""
    best, bb = None, 0
    for e in range(0x7f):
        ex, m = e >> 3, e & 7
        v = m * 2.0 ** -10 if ex == 0 else (8 + m) * 2.0 ** (ex - 11)
        if best is None or abs(v - d) < abs(best - d):
            best, bb = v, e
    return bb


def quant(gtype, n_values, std):
    """n_values random weights of the given type with about this standard deviation."""
    if gtype == T_F32:
        return b"".join(struct.pack("<f", rng.gauss(0, std)) for _ in range(n_values))
    if gtype == T_F16:
        return b"".join(f16(rng.gauss(0, std)) for _ in range(n_values))
    nb = n_values // (256 if gtype in (T_Q5_K, T_Q6_K) else 32)
    out = bytearray()
    if gtype in (T_FP4, T_FP4F):  # codebook rms ~5.36 at half scale
        scales = [ue4m3_byte(std / 5.36 * rng.uniform(0.7, 1.4)) for _ in range(8)]
        body = os.urandom(16 * nb)
        for i in range(nb):
            out += body[16 * i:16 * i + 16]
            out.append(scales[i % 8])
            if gtype == T_FP4:
                out.append(scales[(i + 3) % 8])
    elif gtype == T_Q5_1:  # d * q + m, q uniform 0..31 (std ~9.2)
        d = std / 9.2
        body = os.urandom(20 * nb)
        for i in range(nb):
            out += f16(d) + f16(-15.5 * d) + body[20 * i:20 * i + 20]
    elif gtype == T_Q5_K:  # every sub-block scale and min = 16: d*16*q - dmin*16
        d = std / (16 * 9.2)
        sc = bytes([0x50] * 8 + [0x00] * 4)
        body = os.urandom(160 * nb)
        for i in range(nb):
            out += f16(d) + f16(15.5 * d) + sc + body[160 * i:160 * i + 160]
    elif gtype == T_Q6_K:  # d*16*(q - 32), q uniform 0..63 (std ~18.5)
        d = std / (16 * 18.5)
        body = os.urandom(192 * nb)
        for i in range(nb):
            out += body[192 * i:192 * i + 192] + bytes([16] * 16) + f16(d)
    else:
        raise ValueError(gtype)
    return out


def main():
    real, path = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng.seed(seed)
    kv = {k: (t, v, st) for k, t, v, st in read_header_kv(real)}

    # widths: the kernels need head_dim 256 and d_state 128; K % 256 everywhere the int8 path runs
    E, HC, LR = 512, 4, 320                  # embedding, hyper-connection streams, their low rank (as released)
    NL, FAI = 8, 4                            # layers, full attention every 4th (3 and 7)
    HQ, HKV, HD, NROT = 4, 2, 256, 64
    NG, NV, DS, DCONV = 2, 4, 128, 4          # DeltaNet: q/k heads, v heads, head size, conv width
    DI = NV * DS
    NE, NU, FE, FS = 16, 4, 256, 256          # experts, used, expert width, shared expert width
    IH, IK = 4, 128                           # indexer (unused: compress ratios are 0)
    PLE_L, NGRAM, PER, PCONV = 1, 3, 8, 4     # PLE layer, n-gram size, heads per n-gram, conv width
    PH = (NGRAM - 1) * PER                    # 16 PLE heads
    PD = E // PH                              # their row width (32): PH * PD feeds ple_key / ple_value
    V = len(kv["tokenizer.ggml.tokens"][1])
    HCD = HC * E

    w = Writer()
    a = "qwen4exp."
    w.add("general.architecture", STR, "qwen4exp")
    w.add("general.name", STR, "qwen4exp-tiny-random")
    for k, t, v in [("block_count", U32, NL), ("context_length", U32, 262144), ("embedding_length", U32, E),
                    ("attention.head_count", U32, HQ), ("attention.head_count_kv", U32, HKV),
                    ("rope.freq_base", F32, 10000000.0), ("attention.layer_norm_rms_epsilon", F32, 1e-6),
                    ("expert_count", U32, NE), ("expert_used_count", U32, NU), ("attention.key_length", U32, HD),
                    ("attention.value_length", U32, HD), ("expert_feed_forward_length", U32, FE),
                    ("expert_shared_feed_forward_length", U32, FS), ("ssm.conv_kernel", U32, DCONV),
                    ("ssm.state_size", U32, DS), ("ssm.group_count", U32, NG), ("ssm.time_step_rank", U32, NV),
                    ("ssm.inner_size", U32, DI), ("full_attention_interval", U32, FAI),
                    ("rope.dimension_count", U32, NROT), ("hyper_connection.count", U32, HC),
                    ("hyper_connection.low_rank", U32, LR), ("attention.indexer.head_count", U32, IH),
                    ("attention.indexer.key_length", U32, IK), ("attention.indexer.top_k", U32, 2048),
                    ("ple.ngram_size", U32, NGRAM), ("ple.heads_per_ngram", U32, PER),
                    ("ple.conv_kernel", U32, PCONV), ("ple.eos_token_id", U32, kv[a + "ple.eos_token_id"][1]),
                    ("ple.image_token_id", U32, kv[a + "ple.image_token_id"][1]),
                    ("embedding_length_per_layer_input", U32, PD)]:
        w.add(a + k, t, v)
    w.add(a + "rope.dimension_sections", ARR, [11, 11, 10, 0], I32)
    w.add(a + "attention.compress_ratios", ARR, [0] * NL, U32)
    w.add(a + "ple.layers", ARR, [PLE_L], U32)
    w.add(a + "ple.layer_multipliers", ARR, kv[a + "ple.layer_multipliers"][1], kv[a + "ple.layer_multipliers"][2])
    # small prime-ish head vocabularies, so the table stays small but hashing still spreads
    hv = [997 + 6 * h for h in range(PH)]
    offs = [sum(hv[:h]) for h in range(PH)]
    w.add(a + "ple.head_offsets", ARR, offs, U64)
    w.add(a + "ple.head_vocab_sizes", ARR, hv, U64)
    for k, (t, v, st) in kv.items():
        if k.startswith("tokenizer."):
            w.add(k, t, v, st)

    s = lambda K: 1.0 / math.sqrt(K)
    ones = lambda n, jit=0.1: b"".join(struct.pack("<f", 1.0 + rng.uniform(-jit, jit)) for _ in range(n))
    vec = lambda n, lo, hi: b"".join(struct.pack("<f", rng.uniform(lo, hi)) for _ in range(n))

    w.tensor("token_embd.weight", [E, V], T_Q5_K, quant(T_Q5_K, E * V, 1.0))
    w.tensor("output.weight", [E, V], T_Q6_K, quant(T_Q6_K, E * V, 2.0 * s(E)))
    w.tensor("output_hc_norm.weight", [HCD], T_F32, ones(HCD))
    w.tensor("output_hc_down.weight", [HCD, LR], T_FP4F, quant(T_FP4F, HCD * LR, s(HCD)))
    w.tensor("output_hc_up.weight", [LR, HCD], T_FP4F, quant(T_FP4F, LR * HCD, 2 * s(LR)))
    ple_rows = (sum(hv) + 255) // 256 * 256
    w.tensor("per_layer_token_embd.weight", [PD, ple_rows], T_Q5_1, quant(T_Q5_1, PD * ple_rows, 1.0))

    for l in range(NL):
        b = f"blk.{l}."
        for m in ("attn", "ffn"):
            w.tensor(b + f"hc_{m}_norm.weight", [HCD], T_F32, ones(HCD))
            w.tensor(b + f"hc_{m}_down.weight", [HCD, LR], T_FP4F, quant(T_FP4F, HCD * LR, s(HCD)))
            w.tensor(b + f"hc_{m}_up.weight", [LR, HCD], T_FP4F, quant(T_FP4F, LR * HCD, 2 * s(LR)))
            w.tensor(b + f"hc_{m}_inject.weight", [HCD, HC], T_FP4F, quant(T_FP4F, HCD * HC, 2 * s(HCD)))
        if (l + 1) % FAI != 0:  # Gated DeltaNet
            conv_dim = 2 * NG * DS + DI
            w.tensor(b + "attn_qkv.weight", [E, conv_dim], T_FP4, quant(T_FP4, E * conv_dim, s(E)))
            w.tensor(b + "attn_gate.weight", [E, DI], T_FP4F, quant(T_FP4F, E * DI, s(E)))
            w.tensor(b + "ssm_conv1d.weight", [DCONV, conv_dim], T_F32, vec(DCONV * conv_dim, -0.5, 0.5))
            w.tensor(b + "ssm_dt.bias", [NV], T_F32, vec(NV, -1.0, 1.0))
            w.tensor(b + "ssm_a", [NV], T_F32, vec(NV, -2.0, -0.2))
            w.tensor(b + "ssm_beta.weight", [E, NV], T_FP4F, quant(T_FP4F, E * NV, s(E)))
            w.tensor(b + "ssm_alpha.weight", [E, NV], T_FP4F, quant(T_FP4F, E * NV, s(E)))
            w.tensor(b + "ssm_norm.weight", [DS], T_F32, ones(DS))
            w.tensor(b + "ssm_out.weight", [DI, E], T_FP4F, quant(T_FP4F, DI * E, s(DI)))
        else:  # gated full attention (+ the indexer, unused at compress ratio 0)
            w.tensor(b + "attn_q.weight", [E, 2 * HQ * HD], T_FP4F, quant(T_FP4F, E * 2 * HQ * HD, s(E)))
            w.tensor(b + "attn_k.weight", [E, HKV * HD], T_FP4, quant(T_FP4, E * HKV * HD, s(E)))
            w.tensor(b + "attn_v.weight", [E, HKV * HD], T_FP4, quant(T_FP4, E * HKV * HD, s(E)))
            w.tensor(b + "attn_output.weight", [HQ * HD, E], T_FP4F, quant(T_FP4F, HQ * HD * E, s(HQ * HD)))
            w.tensor(b + "attn_q_norm.weight", [HD], T_F32, ones(HD))
            w.tensor(b + "attn_k_norm.weight", [HD], T_F32, ones(HD))
            w.tensor(b + "indexer.k_norm.weight", [IK], T_F32, ones(IK))
            w.tensor(b + "indexer.k_proj.weight", [E, IK], T_FP4F, quant(T_FP4F, E * IK, s(E)))
            w.tensor(b + "indexer.q_norm.weight", [IK], T_F32, ones(IK))
            w.tensor(b + "indexer.q_proj.weight", [E, IH * IK], T_FP4F, quant(T_FP4F, E * IH * IK, s(E)))
        if l == PLE_L:
            w.tensor(b + "ple_conv1d.weight", [PCONV, HCD], T_F16, quant(T_F16, PCONV * HCD, 0.4))
            w.tensor(b + "ple_key.weight", [E, HCD], T_FP4F, quant(T_FP4F, E * HCD, s(E)))
            w.tensor(b + "ple_value.weight", [E, E], T_FP4F, quant(T_FP4F, E * E, s(E)))
            for n in ("conv", "key", "query"):
                w.tensor(b + f"ple_norm_{n}.weight", [HCD], T_F32, ones(HCD))
        w.tensor(b + "ffn_gate_inp.weight", [E, NE], T_F32, quant(T_F32, E * NE, s(E)))
        w.tensor(b + "ffn_gate_inp_shexp.weight", [E], T_F32, quant(T_F32, E, s(E)))
        w.tensor(b + "ffn_gate_exps.weight", [E, FE, NE], T_FP4F, quant(T_FP4F, E * FE * NE, s(E)))
        w.tensor(b + "ffn_up_exps.weight", [E, FE, NE], T_FP4F, quant(T_FP4F, E * FE * NE, s(E)))
        w.tensor(b + "ffn_down_exps.weight", [FE, E, NE], T_FP4F, quant(T_FP4F, FE * E * NE, s(FE)))
        w.tensor(b + "ffn_gate_shexp.weight", [E, FS], T_FP4F, quant(T_FP4F, E * FS, s(E)))
        w.tensor(b + "ffn_up_shexp.weight", [E, FS], T_FP4F, quant(T_FP4F, E * FS, s(E)))
        w.tensor(b + "ffn_down_shexp.weight", [FS, E], T_FP4F, quant(T_FP4F, FS * E, s(FS)))
    w.write(path)
    print(f"{path}: {len(w.tensors)} tensors, vocab {V}, PLE table {ple_rows} rows")


main()
