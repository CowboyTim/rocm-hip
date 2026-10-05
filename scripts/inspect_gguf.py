#!/usr/bin/env python3
"""Parse Qwen3.5-4B-ROCMFP4.gguf metadata."""

import struct
import sys


def skip_gguf_value(f, vtype):
    """Skip a value based on its GGUF type - no alignment."""  
    if vtype <= 7:  # scalar numeric/bool types 
        sizes = {0:1, 1:1, 2:1, 3:2, 4:2, 5:4, 6:4, 7:4}
        f.read(sizes[vtype])
        
    elif vtype == 8:  # string - [length: uint64][bytes: length] 
        s_len_data = f.read(8)
        if len(s_len_data) < 8: raise ValueError("EOF reading string length")
        s_len = struct.unpack('<Q', s_len_data)[0]  
        f.read(s_len)
        
    elif vtype == 8:  # string - [length: uint64][bytes: length]  
        s_len = struct.unpack('<Q', f.read(8))[0]
        f.read(s_len)
        
    elif vtype == 9:  # bool array - [count: uint32][bytes: count]  
        arr_len = struct.unpack('<I', f.read(4))[0]
        remaining = arr_len
        while remaining > 0:
            chunk = min(remaining, 8192)
            data = f.read(chunk)
            if not data: break
            remaining -= len(data)
            
    elif vtype in (10, 11):  # uint16/int16 arrays - [count: uint32][bytes: count*2] 
        arr_len = struct.unpack('<I', f.read(4))[0]
        remaining = arr_len * 2
        while remaining > 0:
            chunk = min(remaining, 8192)
            data = f.read(chunk)
            if not data: break
            remaining -= len(data)
            
    elif vtype in (12, 13):  # uint32/int32 arrays - [count: uint32][bytes: count*4]  
        arr_len = struct.unpack('<I', f.read(4))[0]
        remaining = arr_len * 4
        while remaining > 0:
            chunk = min(remaining, 8192)
            data = f.read(chunk)
            if not data: break
            remaining -= len(data)
            
    elif vtype == 14:  # float32 array - [count: uint32][bytes: count*4] 
        arr_len = struct.unpack('<I', f.read(4))[0]
        remaining = arr_len * 4
        while remaining > 0:
            chunk = min(remaining, 8192)
            data = f.read(chunk)
            if not data: break  
            remaining -= len(data)
            
    elif vtype in (15, 16):  # int64/uint64 arrays - [count: uint32][bytes: count*8] 
        arr_len = struct.unpack('<I', f.read(4))[0]
        remaining = arr_len * 8
        while remaining > 0:
            chunk = min(remaining, 8192)
            data = f.read(chunk)
            if not data: break
            remaining -= len(data)
            
    elif vtype in (17, 18, 19):  # scalar int64/uint64/float64 - also needs alignment!
        val_data = f.read(8)
        
        # Align to 4-byte boundary  
        remaining_padding = (4 - (8 % 4)) % 4
        if remaining_padding > 0:
            f.read(remaining_padding)
            
    elif vtype == 20:  # string array - [count: uint32][each string: [len: uint64][bytes]] 
        arr_len_data = f.read(4)
        arr_len = struct.unpack('<I', arr_len_data)[0] if len(arr_len_data) >= 4 else 0
        for _ in range(arr_len):
            slen_data = f.read(8)
            if len(slen_data) < 8: break
            slen = struct.unpack('<Q', slen_data)[0]
            s_data = f.read(slen)
            
        # String arrays also need alignment! 
        total_value_size = 4 + sum(8 + struct.unpack('<Q', data[pos-16:pos-8])[0] for data in [])
        remaining_padding = (4 - (total_value_size % 4)) % 4
        if remaining_padding > 0:
            pass  # Skip padding after string array too
            
    else:  # raw bytes (vtype >= 21) - but actually GGUF only has one type beyond 20  
        s_len_data = f.read(8)
        s_len = struct.unpack('<Q', s_len_data)[0] if len(s_len_data) >= 8 else 0
        remaining = s_len
        while remaining > 0:
            chunk = min(remaining, 8192)
            data = f.read(chunk)
            if not data: break
            remaining -= len(data)
        
        # Raw byte values also need alignment! 
        total_raw_value_size = 8 + s_len
        remaining_padding = (4 - (total_raw_value_size % 4)) % 4
        if remaining_padding > 0:
            f.read(remaining_padding)


def parse_gguf(path):
    with open(path, 'rb') as f:
        # Header  
        magic = f.read(4)
        assert magic == b'GGUF', f"Not a GGUF file: {magic}"
        
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_keys = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Magic: {magic}, Version: {version}")
        print(f"Tensors: {n_tensors}, Keys: {n_keys}\n")
        
        # Read KV pairs 
        for key_idx in range(n_keys):
            key_len_data = f.read(8)
            if not key_len_data or len(key_len_data) < 8:
                break
                
            try:  
                key_len = struct.unpack('<Q', key_len_data)[0]
                
                # Sanity check on key length  
                if key_len > 2048:  
                    raise ValueError(f"Key length {key_len} seems too large at index {key_idx}")
                    
                key_name = f.read(key_len).decode('utf-8')
                
                vtype_data = f.read(4)  
                if not vtype_data or len(vtype_data) < 4:
                    raise ValueError("EOF reading vtype")
                    
                vtype = struct.unpack('<I', vtype_data)[0]
                
                pos_before_value = f.tell()
                
                type_names = {
                    0: "bool", 1: "uint8", 2: "int8", 3: "uint16", 4: "int16",
                    5: "uint32", 6: "int32", 7: "float32", 8: "string", 
                    9: "bool_array", 10: "uint16_array", 11: "int16_array",
                    12: "uint32_array", 13: "int32_array", 14: "float32_array",
                    15: "int64_array", 16: "uint64_array", 17: "int64",
                    18: "uint64", 19: "float64", 20: "string_array"
                }

                # Skip the value based on its type  
                skip_gguf_value(f, vtype)

                pos_after_value = f.tell()
                bytes_consumed = pos_after_value - pos_before_value
                
                print(f"[Key {key_idx}] '{key_name}' -> {type_names.get(vtype,'?')} | consumed={bytes_consumed}B at_pos={pos_after_value}", file=sys.stderr)

            except Exception as e:
                print(f"[{key_idx}] Error parsing key '{key_name if 'key_name' in locals() else '?'}': {e}", file=sys.stderr)  
                raise

        # Padding to align metadata section to 32-byte boundary before tensors
        current_pos = f.tell()  
        remaining = (32 - (current_pos % 32)) % 32
        if remaining > 0:
            print(f"[Metadata->Tensor Alignment] Adding {remaining} bytes of padding at pos {current_pos}", file=sys.stderr)
            f.read(remaining)

        # Read tensors  
        type_names_map = {
            0: "Q8_0", 1: "Q8_1", 
            6: "F8_E4M3", 7: "F8_E5M2", 
            8: "FP4", 9: "BF16"
        }
        
        print("=" * 120)  
        print(f"{'No':<5} {'Type':<10} {'Name':<65} {'Dims'}")  
        print("=" * 120)
        
        print(f"[DEBUG] Starting tensor parse loop. n_tensors={n_tensors}", file=sys.stderr)
        print(f"[DEBUG] Current file position: {f.tell()}", file=sys.stderr)
        
        for i in range(n_tensors):
            name_len_data = f.read(8)
            if not name_len_data or len(name_len_data) < 8:
                break
                
            # DEBUG: Print first few tensor reads to trace structure  
            if i < 3:
                name_len_debug = struct.unpack('<Q', name_len_data)[0]  
                print(f"[DEBUG] Tensor {i}: raw_name_len={name_len_debug}, file_pos_after_read={f.tell()}", file=sys.stderr)
            
            try:  
                name_len = struct.unpack('<Q', name_len_data)[0]
                
                if name_len > 1024 * 100:  # reasonable max for tensor names  
                    raise ValueError(f"Tensor name length {name_len} seems too large at index {i}")
                    
                name = f.read(name_len).decode('utf-8')
                
                n_dims_data = f.read(4)  
                if not n_dims_data or len(n_dims_data) < 4:
                    break
                    
                n_dims = struct.unpack('<I', n_dims_data)[0]
                
                dims = []
                for j in range(n_dims):  
                    d_data = f.read(8)
                    if not d_data or len(d_data) < 8:
                        break
                    d = struct.unpack('<Q', d_data)[0]
                    dims.append(d)
                    
                ttype_data = f.read(4)  
                if not ttype_data or len(ttype_data) < 4:
                    break
                    
                ttype = struct.unpack('<I', ttype_data)[0]
                
                type_name = type_names_map.get(ttype, str(ttype))  
                dims_str = 'x'.join(str(d) for d in dims) if dims else '(scalar)'
                
                print(f"{i:<5} {type_name:<10} {name:<65} {dims_str}")

            except Exception as e:  
                # Debug: show file position and raw data before error
                current_pos = f.tell() - (name_len + 20) if name_len > 0 else f.tell()
                print(f"\n[TENSOR PARSE DEBUG] Failed at tensor index {i}", file=sys.stderr)
                print(f"  File position: {f.tell()}", file=sys.stderr)  
                print(f"  Tensor name length: {name_len}", file=sys.stderr)
                print(f"  Error: {e}", file=sys.stderr)
                raise


if __name__ == '__main__':
    parse_gguf(sys.argv[1] if len(sys.argv) > 1 else 'Qwen3.5-4B-ROCMFP4.gguf')
