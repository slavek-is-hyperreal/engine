#!/usr/bin/env python3
"""
extract_weights.py — wyciąga wagi z TinyLlama GGUF i zapisuje do JSON.

Użycie:
    # Aktywuj venv (jeśli jeszcze nie ma, skrypt sam go stworzy)
    python3 scripts/extract_weights.py [ścieżka_do_gguf]

Wynik: models/layer0_weights.json
"""

import sys
import os
import json
import struct
import numpy as np

GGUF_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/tinyllama-f16.gguf"
OUT_PATH   = "models/layer0_weights.json"
SEED       = 42
K          = 2048   # wymiar wejściowy
D_K        = 64     # wymiar na head (TinyLlama: d_model/n_heads = 2048/32)

print(f"Loading {GGUF_PATH} via gguf library...")

try:
    import gguf
except ImportError:
    print("ERROR: pip install gguf")
    sys.exit(1)

# Wczytaj plik GGUF przez oficjalną bibliotekę llama.cpp
reader = gguf.GGUFReader(GGUF_PATH, "r")

def get_tensor(name: str) -> np.ndarray:
    """Pobiera tensor po nazwie i konwertuje do f32."""
    for t in reader.tensors:
        if t.name == name:
            data = t.data
            # f16 → f32
            if data.dtype == np.float16:
                data = data.astype(np.float32)
            elif data.dtype != np.float32:
                data = data.astype(np.float32)
            return data
    raise KeyError(f"Tensor '{name}' not found in GGUF")

print("Available tensors (layer 0):")
layer0_tensors = [t.name for t in reader.tensors if "blk.0" in t.name]
for n in sorted(layer0_tensors):
    t = next(t for t in reader.tensors if t.name == n)
    print(f"  {n}: shape={list(t.shape)}, dtype={t.data.dtype}")

# Q projection: shape [d_out, d_in] = [2048, 2048]
print(f"\nLoading blk.0.attn_q.weight...")
w_q_full = get_tensor("blk.0.attn_q.weight")
print(f"  shape: {w_q_full.shape}, dtype: {w_q_full.dtype}")

# TinyLlama: W_Q shape może być [2048, 2048] (row-major: output × input)
# Bierzemy pierwsze D_K wierszy = head 0
if w_q_full.ndim == 1:
    # flat — reshape
    w_q_full = w_q_full.reshape(K, K)

w_q_head0 = w_q_full[:D_K, :]   # [64, 2048]
print(f"  head 0: shape={w_q_head0.shape}")

# RMSNorm weights
print("Loading blk.0.attn_norm.weight...")
rms_norm = get_tensor("blk.0.attn_norm.weight")
print(f"  shape: {rms_norm.shape}")

# Input sample: zakres (1.5, 4.5) — bezpieczne dla mul_cf (wymaga x > 1)
# seed=42 dla reprodukowalności
rng = np.random.default_rng(SEED)
input_sample = rng.uniform(1.5, 4.5, size=K).astype(np.float32)
print(f"Input sample: min={input_sample.min():.4f}, max={input_sample.max():.4f}, mean={input_sample.mean():.4f}")

# ALU reference: pełna projekcja dla head 0
alu_reference = (w_q_head0 @ input_sample).tolist()
print(f"ALU reference (head 0): [{alu_reference[0]:.6f}, {alu_reference[1]:.6f}, ...]")

# Zapisz JSON
out = {
    "k": K,
    "d_k": D_K,
    "seed": SEED,
    "w_q_head0": w_q_head0.tolist(),        # [64][2048]
    "rms_norm": rms_norm.tolist(),          # [2048]
    "input_sample": input_sample.tolist(),   # [2048]
    "alu_reference": alu_reference,          # [64] — ground truth
    "meta": {
        "model": "TinyLlama-1.1B",
        "layer": 0,
        "head": 0,
        "n_heads": 32,
        "d_model": 2048,
        "note": "input_sample in (1.5, 4.5) for EML mul_cf stability (requires x > 1)"
    }
}

os.makedirs("models", exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out, f)

size_mb = os.path.getsize(OUT_PATH) / 1e6
print(f"\nSaved to {OUT_PATH} ({size_mb:.1f} MB)")
print("Ready for: cargo run --bin eml_benchmark --release")
