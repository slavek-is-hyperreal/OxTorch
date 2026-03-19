import vulkannn_rusted as vnn
import numpy as np
import torch
import time

m, k, n = 128, 128, 128
a_np = np.random.randn(m, k).astype(np.float32)
b_np = np.random.randn(k, n).astype(np.float32)
c_expected = a_np @ b_np

a_vnn = vnn.Tensor(data=a_np, dtype=vnn.DataType.F16, device="cpu")
b_vnn = vnn.Tensor(data=b_np, dtype=vnn.DataType.F16, device="cpu")

print(f"Testing {m}x{k}x{n} MatMul F16 on CPU (VNN)...")
start = time.time()
c_vnn = a_vnn @ b_vnn
vnn_time = time.time() - start
print(f"VNN Time: {vnn_time:.4f}s ({ (2*m*k*n)/(vnn_time*1e9):.2f} GFLOPS)")

print(f"Testing {m}x{k}x{n} MatMul F16 on CPU (PyTorch)...")
a_pt = torch.from_numpy(a_np).half()
b_pt = torch.from_numpy(b_np).half()
start = time.time()
c_pt = a_pt @ b_pt
pt_time = time.time() - start
print(f"PyTorch Time: {pt_time:.4f}s ({ (2*m*k*n)/(pt_time*1e9):.2f} GFLOPS)")

out = c_vnn.to_numpy()
max_diff = np.abs(out - c_expected).max()
print(f"Max diff vs F32: {max_diff}")
if max_diff < 0.5:
    print("✅ Parity OK")
else:
    print("❌ Parity FAIL")

diff_pt = np.abs(out - c_pt.float().numpy()).max()
print(f"Max diff vs PyTorch F16: {diff_pt}")
