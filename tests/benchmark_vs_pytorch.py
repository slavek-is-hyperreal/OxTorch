import torch
import time
import numpy as np
import vulkannn_rusted as vnn
import os

# Configuration
M, K, N = 10000, 10000, 10000 # 400MB per matrix, 1.2GB total (RAM-resident)
M_LARGE, K_LARGE, N_LARGE = 40000, 40000, 40000 # ~6.4GB per matrix, 19.2GB total (OOM-resident)

POOL_DIR = "/vectorlegis_ssd_pool/vnn_cache"
os.makedirs(POOL_DIR, exist_ok=True)

def gen_file(path, size):
    if os.path.exists(path) and os.path.getsize(path) == size * 4:
        return
    print(f"Generating {path} ({size*4/1024/1024:.1f} MB)...")
    with open(path, "wb") as f:
        # Write in chunks of 100MB to avoid RAM spikes on the host
        chunk_size = 25_000_000 
        data = np.full(chunk_size, 1.0, dtype=np.float32).tobytes()
        written = 0
        while written < size:
            to_write = min(chunk_size, size - written)
            if to_write == chunk_size:
                f.write(data)
            else:
                f.write(np.full(to_write, 1.0, dtype=np.float32).tobytes())
            written += to_write

def test_pytorch(m, k, n):
    print(f"\n[PyTorch CPU] Running {m}x{k}x{n}...")
    a = torch.ones(m, k)
    b = torch.ones(k, n)
    t1 = time.time()
    c = torch.mm(a, b)
    t2 = time.time()
    print(f"   Execution: {t2-t1:.4f}s")
    return c.numpy(), t2-t1

def test_rusted_cpu(m, k, n, file_a, file_b):
    print(f"\n[VNN Rusted CPU] Running {m}x{k}x{n} from SSD...")
    gen_file(file_a, m*k)
    gen_file(file_b, k*n)
    
    t_a = vnn.Tensor.from_ssd(file_a, [m, k])
    t_b = vnn.Tensor.from_ssd(file_b, [k, n])
    t_a.device = "cpu"
    t_b.device = "cpu"
    
    t1 = time.time()
    t_c = t_a @ t_b
    t2 = time.time()
    print(f"   Execution: {t2-t1:.4f}s")
    return t_c.to_numpy(), t2-t1

if __name__ == "__main__":
    print("=== PERFORMANCE & PARITY AUDIT: VNN RUSTED VS PYTORCH ===")
    
    # 1. RAM-Resident Test + Parity
    pt_res, t_pt = test_pytorch(M, K, N)
    rv_res, t_rv = test_rusted_cpu(M, K, N, os.path.join(POOL_DIR, "stress_A.bin"), os.path.join(POOL_DIR, "stress_B.bin"))
    
    # Check parity
    print("\nVerifying Parity...")
    parity = np.allclose(pt_res, rv_res, rtol=1e-3, atol=1e-3)
    if parity:
        print("✅ PARITY OK!")
    else:
        print("❌ PARITY ERROR! Max Diff:", np.abs(pt_res - rv_res).max())
        print("Samples PT:", pt_res[0,0:5])
        print("Samples RV:", rv_res[0,0:5])

    print(f"\nRatio (Rusted/PyTorch): {t_rv/t_pt:.2f}x")
    
    # 2. OOM Test (Optional but let's run it)
    print("\n=== STARTING MASSIVE OOM TEST (16GB+ DATA) ===")
    FILE_A_BIG = os.path.join(POOL_DIR, "massive_A.bin")
    FILE_B_BIG = os.path.join(POOL_DIR, "massive_B.bin")
    
    test_rusted_cpu(M_LARGE, K_LARGE, N_LARGE, FILE_A_BIG, FILE_B_BIG)
