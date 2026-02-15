import os
import json
import subprocess
import tarfile
import torch # We use torch briefly only for loading the downloaded state dict
import numpy as np
from tqdm import tqdm

def download_gemma_3n():
    # 1. Read credentials
    with open("vulkan_nn_lib/kaggle.json", "r") as f:
        creds = json.load(f)
    
    username = creds["username"]
    key = creds["key"]
    
    model_dir = "weights_gemma_3n"
    os.makedirs(model_dir, exist_ok=True)
    tar_path = "gemma_3n_raw.tar.gz"
    
    print(f"--- Downloading Gemma 3n from Kaggle ---")
    url = "https://www.kaggle.com/api/v1/models/google/gemma-3n/transformers/gemma-3n-e4b-it/2/download"
    
    # Use curl as per user's prompt
    cmd = [
        "curl", "-L", "-u", f"{username}:{key}",
        "-o", tar_path,
        url
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Download complete: {tar_path}")
    
    # 2. Extract
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    print(f"Extraction complete to {model_dir}")
    
    # 3. Convert to VulkanNN format (Raw Binaries)
    print(f"--- Converting Weights to VulkanNN format ---")
    vulkan_weights_dir = "vulkan_nn_lib/weights/gemma_3n"
    os.makedirs(vulkan_weights_dir, exist_ok=True)
    
    # Check if we have bin or safetensors
    potential_files = [f for f in os.listdir(model_dir) if f.endswith(".bin") or f.endswith(".safetensors") or f.endswith(".pt")]
    print(f"Found weight files: {potential_files}")
    
    for weight_file in potential_files:
        full_path = os.path.join(model_dir, weight_file)
        print(f"Loading {weight_file}...")
        
        # We try loading with torch (most common kaggle/hf format)
        if weight_file.endswith(".safetensors"):
             from safetensors.torch import load_file
             state_dict = load_file(full_path)
        else:
             state_dict = torch.load(full_path, map_location="cpu")
             
        # Extract individual tensors
        for key, tensor in tqdm(state_dict.items(), desc=f"Converting {weight_file}"):
            # Flatten name for filesystem
            clean_name = key.replace(".", "_")
            out_path = os.path.join(vulkan_weights_dir, f"{clean_name}.bin")
            
            # Save as raw float32 binary
            data = tensor.float().numpy()
            data.tofile(out_path)
            
    print(f"--- Conversion Complete! ---")
    print(f"Vulkan ready weights are in: {vulkan_weights_dir}")

if __name__ == "__main__":
    download_gemma_3n()
