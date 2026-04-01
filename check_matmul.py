import numpy as np
import oxtorch as torch_ox
import torch as th
import time

def check_matmul():
    print("--- OxTorch vs PyTorch vs NuGet Math Validation ---")
    device = "vulkan"
    
    # Large scale test: 4096 x 128 (Close to SfM workloads)
    N = 4096
    K = 128 
    
    print(f"Testing Matrix Size: {N}x{K} matmul {K}x{N}")
    
    # Generate data (Normalised for Cosine Similarity)
    np_a = np.random.rand(N, K).astype(np.float32)
    norms = np.linalg.norm(np_a, axis=1, keepdims=True)
    np_a /= (norms + 1e-7)
    
    np_b = np_a.copy() # Symmetric test
    
    # 1. NumPy CPU
    print("CPU Baseline (NumPy)...")
    start = time.time()
    expected = np.matmul(np_a, np_b.T)
    print(f"  NumPy Time: {time.time()-start:.4f}s | Max val: {expected.max():.4f}")
    
    # 2. PyTorch CPU
    print("CPU Reference (PyTorch)...")
    th_a = th.from_numpy(np_a)
    th_b = th.from_numpy(np_b)
    start = time.time()
    result_th = th.matmul(th_a, th_b.T).numpy()
    print(f"  PyTorch Time: {time.time()-start:.4f}s | Max val: {result_th.max():.4f}")
    
    # 3. OxTorch Vulkan (The Card)
    print("GPU Test (OxTorch Vulkan - Pre-transposed/Contiguous)...")
    try:
        t1 = torch_ox.from_numpy(np_a).to(device)
        # Manually transpose in NumPy to ensure it's contiguous when sent to GPU
        np_b_t = np_b.T.copy() 
        t2 = torch_ox.from_numpy(np_b_t).to(device) 
        
        start = time.time()
        # Note: We now call matmul on two contiguous buffers, no logical transpose in OxTorch
        result_ox = torch_ox.matmul(t1, t2).to_numpy()
        print(f"  OxTorch Time: {time.time()-start:.4f}s | Max val: {result_ox.max():.4f}")
        
        # Cross-validation
        diff_th = np.abs(result_th - result_ox)
        max_diff = np.max(diff_th)
        invalid_mask = result_ox > 1.0001
        invalid_count = np.sum(invalid_mask)
        
        print("\n--- RESULTS ---")
        print(f"Max Difference vs PyTorch: {max_diff:.6f}")
        print(f"Invalid Values (> 1.0): {invalid_count}")
        
        if max_diff < 1e-4 and invalid_count == 0:
            print("STATUS: SUCCESS - OxTorch matches PyTorch baseline.")
        else:
            print("STATUS: FAILURE - Numerical divergence detected!")
            if invalid_count > 0:
                print(f"  CRITICAL: Shader produces non-normalized results (max={result_ox.max():.4f})")
                
    except Exception as e:
        print(f"OxTorch Error: {e}")

if __name__ == "__main__":
    check_matmul()
