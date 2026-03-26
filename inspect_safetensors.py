import json
import os
from safetensors import safe_open

model_path = "models/bitnet-2B-ternary/model.safetensors"

with safe_open(model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"Key: {key:<50} Shape: {str(tensor.shape):<20} Dtype: {tensor.dtype}")
