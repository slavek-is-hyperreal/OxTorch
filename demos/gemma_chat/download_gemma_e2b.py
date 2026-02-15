import os
import json
import subprocess
import tarfile
import torch 
import numpy as np
from tqdm import tqdm

def download_gemma_3n_e2b():
    # 1. Read credentials
    with open("vulkan_nn_lib/kaggle.json", "r") as f:
        creds = json.load(f)
    
    username = creds["username"]
    key = creds["key"]
    
    model_dir = "weights_gemma_3n_e2b"
    os.makedirs(model_dir, exist_ok=True)
    if os.path.exists(tar_path):
        print(f"Tarball {tar_path} already exists, skipping download.")
    else:
        print(f"--- Downloading Gemma 3n E2B from Kaggle ---")
        url = "https://www.kaggle.com/api/v1/models/google/gemma-3n/transformers/gemma-3n-e2b-it/2/download"
        
        cmd = [
            "curl", "-L", "-u", f"{username}:{key}",
            "-o", tar_path,
            url
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Download complete: {tar_path}")
    
    # 2. Extract
    has_weights = any(f.endswith(".safetensors") or f.endswith(".bin") for f in os.listdir(model_dir)) if os.path.exists(model_dir) else False

    if has_weights:
        print(f"Weights already found in {model_dir}, skipping extraction.")
    else:
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
        print(f"Extraction complete to {model_dir}")
    
    # 3. Convert to VulkanNN format
    vulkan_weights_dir = "vulkan_nn_lib/weights/gemma_3n_e2b"
    os.makedirs(vulkan_weights_dir, exist_ok=True)
    
    # Filter specifically for model shards to avoid unpickling other files
    potential_files = [f for f in os.listdir(model_dir) if (f.startswith("model") and f.endswith(".safetensors")) or (f.startswith("pytorch_model") and f.endswith(".bin"))]
    
    if not potential_files:
        potential_files = [f for f in os.listdir(model_dir) if f.endswith(".bin") or f.endswith(".safetensors")]
    
    print(f"Found weight shards to convert: {potential_files}")
    
    for weight_file in potential_files:
        full_path = os.path.join(model_dir, weight_file)
        if weight_file.endswith(".safetensors"):
             from safetensors.torch import load_file
             state_dict = load_file(full_path)
        else:
             state_dict = torch.load(full_path, map_location="cpu", weights_only=False)
             
        for key, tensor in tqdm(state_dict.items(), desc=f"Converting {weight_file}"):
            clean_name = key.replace(".", "_")
            out_path = os.path.join(vulkan_weights_dir, f"{clean_name}.bin")
            data = tensor.float().numpy()
            data.tofile(out_path)
            
    print(f"--- Conversion Complete! ---")
    print(f"Vulkan ready weights are in: {vulkan_weights_dir}")

if __name__ == "__main__":
    download_gemma_3n_e2b()
