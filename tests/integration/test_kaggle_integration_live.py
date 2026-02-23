import os
import numpy as np
import vulkan_nn_lib.torch_shim as torch
import time

def test_kaggle_small():
    """
    Triggers a small Kaggle offload to verify the flow.
    Note: Requires VNN_KAGGLE_MODE=1 and a lowered threshold in streaming_ops.py for small tests.
    """
    print("Testing Kaggle Remote Offload (Small Scale Verification)")
    
    # Enable Kaggle Mode for this process
    os.environ["VNN_KAGGLE_MODE"] = "1"
    os.environ["VNN_KAGGLE_THRESHOLD"] = "1024" # Trigger on tiny tensors for testing framework
    
    # 1. Create tensors
    # We use a size that would normally trigger SSD if not for Kaggle
    a = torch.randn(1024, 1024) # 1M elements, ~4MB
    b = torch.randn(1024, 1024)
    
    print(f"Tensor A device: {a.device}")
    
    # 2. Perform operation
    # NOTE: Normally Kaggle triggers at 1GB+. 
    # To test with 1M elements, the threshold in streaming_ops.py must be lowered below 4MB.
    print("Executing: c = a + b")
    try:
        start_time = time.time()
        c = a + b
        end_time = time.time()
        
        print(f"Operation complete in {end_time - start_time:.2f}s")
        print(f"Result device: {c.device}")
        
        # 3. Verify parity
        a_np = a.to_numpy()
        b_np = b.to_numpy()
        c_np = c.to_numpy()
        
        expected = a_np + b_np
        diff = np.abs(c_np - expected).max()
        print(f"Max difference vs local NumPy: {diff}")
        
        if diff < 1e-5:
            print("SUCCESS: Kaggle result matches local calculation.")
        else:
            print("FAILURE: Numerical mismatch.")
            
    except Exception as e:
        print(f"Error during Kaggle execution: {e}")

if __name__ == "__main__":
    test_kaggle_small()
