import time
import numpy as np
import torch
import os
import sys
import json
import importlib

# Force line-buffered output
sys.stdout.reconfigure(line_buffering=True)

_VNN_CANDIDATES = [
    "vulkannn_rusted_exp",
    "vulkannn_rusted_dev",
    "vulkannn_rusted_test",
    "vulkannn_rusted_main",
    "vulkannn_rusted",
]

def load_vnn():
    for _mod_name in _VNN_CANDIDATES:
        try:
            vnn = importlib.import_module(_mod_name)
            return vnn, _mod_name
        except ImportError:
            continue
    raise ImportError("No vulkannn_rusted module found.")

def get_system_metrics():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            cpu_temp = float(f.read().strip()) / 1000.0
    except:
        cpu_temp = -1.0
    try:
        load1, _, _ = os.getloadavg()
    except:
        load1 = -1.0
    return cpu_temp, load1

def get_torch_backend_label(dtype_str):
    parts = ["CPU"]
    if torch.backends.mkl.is_available():
        parts.append("MKL")
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'openmp') and torch.backends.openmp.is_available():
        parts.append("OpenMP")
    dtype_map = {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "int8": "int8"}
    parts.append(dtype_map.get(dtype_str, dtype_str))
    return "·".join(parts)

def check_parity(vnn_tensor, torch_tensor, name, op=""):
    import torch
    import numpy as np
    
    rtol = 1e-3
    atol = 1e-2
    name_l = name.lower()
    if "bf16" in name_l:
        atol = 1.0
        if "sum" in name_l: atol = 50.0
    elif "f16" in name_l and "sum" in name_l:
        atol = 0.5
    if "int8" in name_l:
        if "sum" in name_l: atol = 5000.0
        elif "matmul" in name_l or op == "MatMul": atol = 300.0
    
    # Extract numpy arrays from whatever we got (VNN tensor, Torch tensor, or already Numpy)
    if isinstance(vnn_tensor, np.ndarray):
        v_np = vnn_tensor.flatten()
    else:
        v_np = vnn_tensor.to_numpy().flatten()

    if isinstance(torch_tensor, np.ndarray):
        t_np = torch_tensor.flatten()
    else:
        # Pytorch doesn't support bfloat16 -> numpy directly, so we cast to f32
        t_np = torch_tensor.detach().cpu().to(torch.float32).numpy().flatten()
    
    try:
        np.testing.assert_allclose(v_np, t_np, atol=atol, rtol=rtol)
        return True, 0.0
    except AssertionError:
        diff = np.abs(v_np - t_np)
        return False, np.max(diff)

from datetime import datetime

def save_benchmark_result(name, result_data):
    results_dir = "/my_data/gaussian_room/tests/results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(' ', '_').lower()
    file_path = os.path.join(results_dir, f"{safe_name}_{timestamp}.json")
    
    # Save the individual result
    with open(file_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    # Also maintain a 'latest' file for easy access
    latest_path = os.path.join(results_dir, f"{safe_name}_latest.json")
    with open(latest_path, 'w') as f:
        json.dump(result_data, f, indent=4)
        
    print(f"[benchmark] Result saved to {file_path} (Latest: {latest_path})")
