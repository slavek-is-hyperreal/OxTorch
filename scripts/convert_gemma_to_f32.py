import os
import glob
import torch
import numpy as np
from safetensors import safe_open
from tqdm import tqdm

def get_cache_dir():
    # Respect .env file
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("VNN_CACHE_DIR="):
                    return line.split("=")[1].strip()
    return "/vectorlegis_ssd_pool/vnn_cache"

def convert_to_f32():
    src_dir = "gemma-weights"
    cache_base = get_cache_dir()
    dst_dir = os.path.join(cache_base, "gemma-3n-E4B-it-f32")
    os.makedirs(dst_dir, exist_ok=True)
    
    print(f"Starting conversion: {src_dir} -> {dst_dir}")
    print(f"Using cache base from .env: {cache_base}")
    
    files = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    
    mapping = {}
    
    for f_path in files:
        print(f"Processing {f_path}...")
        with safe_open(f_path, framework="pt", device="cpu") as f:
            for key in tqdm(f.keys(), desc="Tensors"):
                tensor = f.get_tensor(key)
                # Convert to float32
                tensor_f32 = tensor.to(torch.float32)
                
                # We save as raw binary .bin files for easiest loading into VNN-Rusted
                # Sanitize key for filesystem
                safe_key = key.replace(".", "_")
                bin_filename = f"{safe_key}.bin"
                bin_path = os.path.join(dst_dir, bin_filename)
                
                # NumPy export
                data = tensor_f32.numpy()
                data.tofile(bin_path)
                
                # Store metadata for the loader
                mapping[key] = {
                    "path": bin_path,
                    "shape": list(data.shape),
                    "elements": int(data.size)
                }

    # Save mapping for the loader
    import json
    with open(os.path.join(dst_dir, "vnn_weights_map.json"), "w") as j:
        json.dump(mapping, j, indent=2)
        
    print(f"\nSuccess! 32GB FP32 weights prepared at: {dst_dir}")
    print("The model is now too big for your 23GB RAM. Ready for OOM-Safe Demo!")

if __name__ == "__main__":
    convert_to_f32()
