import torch
import time
import numpy as np
import vulkannn_rusted as vnn
import os

# Configuration
# RAM Resident: 1GB per tensor (250M float32)
SIZE_RAM = 250_000_000 
# OOM Resident: 8GB per tensor (2.0B float32) -> 3 tensors = 24GB total.
# This should trigger the new Tiling logic once implemented.
SIZE_OOM = 2_000_000_000 

POOL_DIR = "/vectorlegis_ssd_pool/vnn_cache"
os.makedirs(POOL_DIR, exist_ok=True)

def gen_file(path, elements):
    size_bytes = elements * 4
    if os.path.exists(path) and os.path.getsize(path) == size_bytes:
        return
    print(f"Generating {path} ({size_bytes/1024/1024:.1f} MB)...")
    with open(path, "wb") as f:
        chunk_elements = 25_000_000 
        data = np.full(chunk_elements, 1.0, dtype=np.float32).tobytes()
        written = 0
        while written < elements:
            to_write = min(chunk_elements, elements - written)
            if to_write == chunk_elements:
                f.write(data)
            else:
                f.write(np.full(to_write, 1.0, dtype=np.float32).tobytes())
            written += to_write

def run_bench(name, op_pt, op_vnn, size, mode="cpu"):
    print(f"\n--- Benchmarking {name} | Size: {size/1e6:.1f}M elements | Mode: {mode} ---")
    
    # PyTorch Baseline (RAM Only)
    t_pt = 0
    res_pt_np = None
    if size <= SIZE_RAM:
        try:
            a_pt = torch.ones(size)
            b_pt = torch.ones(size) if name == "Add" else None
            
            start = time.time()
            if name == "Add":
                res_pt = a_pt + b_pt
            else:
                res_pt = op_pt(a_pt)
            t_pt = time.time() - start
            print(f"  [PyTorch CPU] Execution: {t_pt:.4f}s")
            res_pt_np = res_pt.numpy()
        except Exception as e:
            print(f"  [PyTorch CPU] Error: {e}")

    # VNN Rusted
    file_a = os.path.join(POOL_DIR, f"bench_{name}_A.bin")
    file_b = os.path.join(POOL_DIR, f"bench_{name}_B.bin")
    gen_file(file_a, size)
    if name == "Add":
        gen_file(file_b, size)

    t_a = vnn.Tensor.from_ssd(file_a, [size])
    t_a.device = mode
    
    t_b = None
    if name == "Add":
        t_b = vnn.Tensor.from_ssd(file_b, [size])
        t_b.device = mode

    try:
        start = time.time()
        if size > SIZE_RAM:
            # Use SSD Result Streaming for OOM tests
            file_res = os.path.join(POOL_DIR, f"bench_{name}_RES.bin")
            t_res = vnn.Tensor.new_ssd(file_res, [size])
            t_res.device = mode
            if name == "Add":
                t_a.add_into(t_b, t_res)
            else:
                if name == "ReLU":
                    t_a.relu_into(t_res)
                # ...
        else:
            if name == "Add":
                t_res = t_a + t_b
            else:
                t_res = op_vnn(t_a)
        t_vnn = time.time() - start
        print(f"  [VNN Rusted {mode}] Execution: {t_vnn:.4f}s")

        # Parity check
        if res_pt_np is not None:
            res_vnn_np = t_res.to_numpy()
            parity = np.allclose(res_pt_np, res_vnn_np, atol=1e-3)
            if parity:
                print("  ✅ PARITY OK")
            else:
                print(f"  ❌ PARITY ERROR (Max Diff: {np.abs(res_pt_np - res_vnn_np).max()})")
    except Exception as e:
        print(f"  ❌ VNN ERROR: {e}")
        t_vnn = 0

    if t_pt > 0 and t_vnn > 0:
        print(f"  Ratio: {t_vnn/t_pt:.2f}x")

if __name__ == "__main__":
    # Test RAM Resident
    for mode in ["cpu", "vulkan", "hybrid"]:
        run_bench("Add", None, None, SIZE_RAM, mode=mode)
        run_bench("ReLU", torch.relu, lambda t: t.relu(), SIZE_RAM, mode=mode)
    
    # OOM Resident Test
    print("\n\n=== STARTING OOM STRESS TESTS ===")
    run_bench("ReLU", torch.relu, lambda t: t.relu(), SIZE_OOM, mode="cpu")
    run_bench("Add", None, None, SIZE_OOM, mode="vulkan")
