#!/usr/bin/env python3
"""Parse GGUF files — metadata + tensor listing."""

import struct
import sys


def read_string(f):
    """Read a GGUF string: [len->uint64][bytes]."""
    slen = struct.unpack('<Q', f.read(8))[0]
    return f.read(slen).decode('utf-8')


def read_value(f):
    """Read one GGUF value from *f*. Returns (type_name, parsed_value)."""
    vtype = struct.unpack('<I', f.read(4))[0]

    # Scalar types (no length prefix)
    if vtype == 0:   # uint8
        return ("u8", struct.unpack('<B', f.read(1))[0])
    elif vtype == 1: # int8
        return ("i8", struct.unpack('<b', f.read(1))[0])
    elif vtype == 2: # uint16
        return ("u16", struct.unpack('<H', f.read(2))[0])
    elif vtype == 3: # int16
        return ("i16", struct.unpack('<h', f.read(2))[0])
    elif vtype == 4: # uint32
        val = struct.unpack('<I', f.read(4))[0]
        return ("u32", val)
    elif vtype == 5: # int32
        val = struct.unpack('<i', f.read(4))[0]
        return ("i32", val)
    elif vtype == 6: # float32
        val = struct.unpack('<f', f.read(4))[0]
        return ("f32", val)
    elif vtype == 7: # bool (stored as int8)
        val = struct.unpack('<b', f.read(1))[0] != 0
        return ("bool", val)
    elif vtype == 8: # string — [len->uint64][bytes]
        return ("str", read_string(f))

    # Array type marker — GGUF v3 layout: ARRAY(vtype=9) + element_type(int32) + count(uint64) + data
    elif vtype == 9:  # is_array, sub-type follows in the data stream
        _etype = struct.unpack('<I', f.read(4))[0]  # element type
        cnt = struct.unpack('<Q', f.read(8))[0]      # count

        elem_sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1}
        if _etype in elem_sizes:
            f.seek(cnt * elem_sizes[_etype], 1)
        elif _etype == 8:  # string array — each str has uint64 len prefix
            for _ in range(cnt):
                s = struct.unpack('<Q', f.read(8))[0]
                f.seek(s, 1)
        else:
            raise ValueError(f"Unknown array element type {_etype}")

        return (f"arr[{_etype}]x{cnt}", None)

    else:
        raise ValueError(f"Unknown value type {vtype}")


def parse_gguf(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f"Not a GGUF file: {magic}"

        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_keys = struct.unpack('<Q', f.read(8))[0]

        print(f"Magic: {magic}, Version: {version}")
        print(f"Tensors: {n_tensors}, Keys: {n_keys}\n")

        # KV pairs
        for key_idx in range(n_keys):
            key_name = read_string(f)
            type_name, val = read_value(f)

            if val is not None and isinstance(val, str):
                display_val = val[:50] + ('...' if len(val) > 50 else '')
            else:
                display_val = str(val) if val is not None else ''

            print(f"[Key {key_idx}] '{key_name}' -> {type_name} ({display_val})", file=sys.stderr)

        # Tensors
        # Tensor data type (ggml_type enum from ggml.h)
        # Values copied from the ROCmFPX fork's ggml/include/ggml.h (100+ are fork-only);
        # fall back to raw value for unknowns
        type_names_tensor = {
            0:"F32", 1:"F16", 2:"Q4_0", 3:"Q4_1", 6:"Q5_0", 7:"Q5_1", 8:"Q8_0", 9:"Q8_1",
            10:"Q2_K", 11:"Q3_K", 12:"Q4_K", 13:"Q5_K", 14:"Q6_K", 15:"Q8_K",
            16:"IQ2_XXS", 17:"IQ2_XS", 18:"IQ3_XXS", 19:"IQ1_S", 20:"IQ4_NL", 21:"IQ3_S",
            22:"IQ2_S", 23:"IQ4_XS", 24:"I8", 25:"I16", 26:"I32", 27:"I64", 28:"F64",
            29:"IQ1_M", 30:"BF16", 34:"TQ1_0", 35:"TQ2_0", 39:"MXFP4", 40:"NVFP4", 41:"Q1_0",
            100:"Q4_0_ROCMFP4", 101:"Q4_0_ROCMFP4_FAST", 102:"Q6_0_ROCMFPX", 103:"Q8_0_ROCMFPX",
            104:"Q3_0_ROCMFPX", 107:"Q2_0_ROCMFPX", 108:"Q4_0_ROCMI4",
        }

        print("=" * 120)
        print(f"{'No':<5} {'Type':<18} {'Name':<65} {'Dims'}")
        print("=" * 120)

        for i in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = []
            for j in range(n_dims):
                dims.append(struct.unpack('<q', f.read(8))[0])
            ttype = struct.unpack('<I', f.read(4))[0]

            type_name = type_names_tensor.get(ttype, f"TYPE{ttype}")
            dims_str = 'x'.join(str(d) for d in dims) if dims else '(scalar)'

            # Read and skip tensor data offset (uint64)
            offset = struct.unpack('<Q', f.read(8))[0]

            print(f"{i:<5} {type_name:<18} {name:<65} {dims_str}")


if __name__ == '__main__':
    parse_gguf(sys.argv[1] if len(sys.argv) > 1 else 'Qwen3.5-4B-ROCMFP4.gguf')
