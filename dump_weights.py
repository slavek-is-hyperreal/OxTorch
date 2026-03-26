import torch
from safetensors import safe_open
import numpy as np

model_path = "models/bitnet-2B-ternary/model.safetensors"

with safe_open(model_path, framework="pt", device="cpu") as f:
    key = "model.layers.1.self_attn.q_proj.weight"
    tensor = f.get_tensor(key)
    print(f"Key: {key}")
    print(f"Shape: {tensor.shape}")
    print(f"First 10 uint8 values: {tensor[0, :10].tolist()}")
    
    # Try to unpack first byte
    first_byte = tensor[0, 0].item()
    print(f"First byte: {first_byte} (bin: {bin(first_byte)})")
    for i in range(4):
        val = (first_byte >> (i * 2)) & 0x03
        print(f"  Unpacked val {i}: {val}")
