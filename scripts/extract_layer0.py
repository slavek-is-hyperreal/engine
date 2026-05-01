import gguf
import json
import numpy as np
import os

def extract_layer0(gguf_path, out_path):
    print(f"Reading {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)
    
    layer0_data = {
        "hidden_dim": 2048,
        "weights": {}
    }
    
    tensor_map = {
        "attn_norm": "blk.0.attn_norm.weight",
        "q": "blk.0.attn_q.weight",
        "k": "blk.0.attn_k.weight",
        "v": "blk.0.attn_v.weight",
        "o": "blk.0.attn_output.weight",
        "ffn_norm": "blk.0.ffn_norm.weight",
        "gate": "blk.0.ffn_gate.weight",
        "up": "blk.0.ffn_up.weight",
        "down": "blk.0.ffn_down.weight"
    }
    
    for key, tensor_name in tensor_map.items():
        tensor = next((t for t in reader.tensors if t.name == tensor_name), None)
        if tensor is None:
            print(f"Warning: Tensor {tensor_name} not found!")
            continue
            
        # Force 2D for consistency in Rust parser
        data = tensor.data.astype(np.float32)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        layer0_data["weights"][key] = data.tolist()
        print(f"Extracted {tensor_name} (shape: {data.shape})")
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(layer0_data, f)
    print(f"Saved layer 0 to {out_path}")

if __name__ == "__main__":
    extract_layer0("models/tinyllama-f16.gguf", "models/layer0_full.json")
