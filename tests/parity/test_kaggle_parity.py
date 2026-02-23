import os
import numpy as np
import vulkan_nn_lib.torch_shim as torch
import time
from vulkan_nn_lib.config import get_kaggle_user

def run_parity_test():
    import sys
    target_op = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("--- Kaggle Mode Parity Test ---")
    
    # 1. Setup environment for testing
    os.environ["VNN_KAGGLE_MODE"] = "1"
    os.environ["VNN_KAGGLE_THRESHOLD"] = "1024" # Trigger Kaggle for anything > 1KB
    
    def check_parity(a, b, op_name, op_func):
        print(f"\nTesting {op_name}...")
        
        # Local calculation (NumPy)
        a_np = a.to_numpy()
        b_np = b.to_numpy() if b is not None else None
        
        if b is not None:
            expected = op_func(a_np, b_np)
        else:
            expected = op_func(a_np)
            
        # Kaggle calculation
        try:
            start_time = time.time()
            if b is not None:
                res = op_func(a, b)
            else:
                res = op_func(a)
            duration = time.time() - start_time
            print(f"  Kaggle {op_name} took {duration:.2f}s")
            
            res_np = res.to_numpy()
            
            # Compare
            diff = np.abs(res_np - expected).max()
            print(f"  Max Diff: {diff:.2e}")
            
            if diff < 1e-4:
                print(f"  SUCCESS: {op_name} parity achieved.")
                return f"SUCCESS: {op_name} parity achieved."
            else:
                print(f"  FAILURE: {op_name} mismatch!")
                return f"FAILURE: {op_name} mismatch!"
        except Exception as e:
            print(f"  ERROR: {op_name} failed: {e}")
            return f"ERROR: {op_name} failed: {e}"

    # (Name, OpFunc, A_vnn, B_vnn or Extra)
    tests = [
        ("Add", lambda x, y: x + y, torch.randn(256, 256), torch.randn(256, 256)),
        ("Sub", lambda x, y: x - y, torch.randn(256, 256), torch.randn(256, 256)),
        ("Mul", lambda x, y: x * y, torch.randn(256, 256), torch.randn(256, 256)),
        ("Div", lambda x, y: x / y, torch.randn(256, 256), torch.randn(256, 256)),
        ("Sum", lambda x, y=None: x.sum(), torch.randn(512, 512), None),
        ("MatMul", lambda x, y: x @ y, torch.randn(32, 128), torch.randn(128, 64)),
    ]

    results = []
    for name, func, a, b in tests:
        if target_op and target_op.lower() not in name.lower():
            continue
        print(f"Testing {name}...")
        results.append(check_parity(a, b, name, func))

    print("\n--- Final Results ---")
    failures = [r for r in results if "FAILED" in r or "ERROR" in r]
    if not failures:
        print("ALL TESTS PASSED! 🎉")
    else:
        for f in failures:
            print(f"  {f}")
        print(f"SOME TESTS FAILED: {len(failures)} failures.")
    
    return not failures

if __name__ == "__main__":
    run_parity_test()
