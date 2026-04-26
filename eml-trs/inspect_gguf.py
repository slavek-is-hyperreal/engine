import struct

def parse_gguf(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        print("Magic:", magic)
        version = struct.unpack('<I', f.read(4))[0]
        print("Version:", version)
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        print("Tensor count:", tensor_count)
        kv_count = struct.unpack('<Q', f.read(8))[0]
        print("KV count:", kv_count)
        
        for i in range(kv_count):
            # print(f"\n--- KV {i} ---")
            key_len = struct.unpack('<Q', f.read(8))[0]
            # print("Key length:", key_len)
            if key_len > 10000:
                print(f"Aborting at KV {i}, huge key length: {key_len}")
                break
            key = f.read(key_len)
            # print("Key:", key.decode('utf-8', errors='replace'))
            val_type = struct.unpack('<I', f.read(4))[0]
            # print("Value type:", val_type)
            
            # Skip the value
            def skip_val(t):
                if t in [0, 1, 7]: f.seek(1, 1)
                elif t in [2, 3]: f.seek(2, 1)
                elif t in [4, 5, 6]: f.seek(4, 1)
                elif t in [10, 11, 12]: f.seek(8, 1)
                elif t == 8:
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.seek(slen, 1)
                elif t == 9: # array
                    at = struct.unpack('<I', f.read(4))[0]
                    alen = struct.unpack('<Q', f.read(8))[0]
                    for _ in range(alen): skip_val(at)
                else: print("UNKNOWN TYPE:", t)
            
            skip_val(val_type)


parse_gguf("models/tinyllama-f16.gguf")
