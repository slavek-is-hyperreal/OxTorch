import time
import numpy as np
import vulkannn_rusted as vnn
from vulkannn_rusted import Tensor

def test_f16_matmul(size=128):
    print(f"Testing F16 MatMul CPU with size {size}...")
    a_np = np.random.randn(size, size).astype(np.float32)
    b_np = np.random.randn(size, size).astype(np.float32)
    
    a = Tensor(data=a_np, dtype=vnn.DataType.F16, device="cpu")
    b = Tensor(data=b_np, dtype=vnn.DataType.F16, device="cpu")
    
    t0 = time.perf_counter()
    res = a @ b
    duration = time.perf_counter() - t0
    print(f"DONE in {duration:.4f}s")
    return res

if __name__ == "__main__":
    test_f16_matmul(128)
    test_f16_matmul(256)
    test_f16_matmul(512)
    test_f16_matmul(1024)
    test_f16_matmul(2048)
