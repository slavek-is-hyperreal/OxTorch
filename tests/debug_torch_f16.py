import torch
import time
import numpy as np

def test_torch_f16_cpu(size=2048):
    print(f"Testing PyTorch MatMul F16 CPU with size {size}...")
    a = torch.randn(size, size).to(torch.float16)
    b = torch.randn(size, size).to(torch.float16)
    
    t0 = time.perf_counter()
    # This might be very slow or hang if not supported well
    res = torch.matmul(a, b)
    duration = time.perf_counter() - t0
    print(f"DONE in {duration:.4f}s")

if __name__ == "__main__":
    test_torch_f16_cpu(512)
    test_torch_f16_cpu(1024)
    test_torch_f16_cpu(2048)
