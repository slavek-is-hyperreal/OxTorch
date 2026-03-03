import torch
import numpy as np
import Python_Legacy.vulkan_nn_lib.core as vnn
from Python_Legacy.vulkan_nn_lib import kernels as K
from tests.utils import check_close

def test_raw_matmul():
    print("Testing RAW Matmul...")
    M, N, K_dim = 4, 64, 128
    
    A_np = np.random.randn(M, K_dim).astype(np.float32)
    B_np = np.random.randn(K_dim, N).astype(np.float32)
    
    tt_A = torch.from_numpy(A_np)
    tt_B = torch.from_numpy(B_np)
    tt_C = tt_A @ tt_B
    
    # 2. VNN
    vnn_A = vnn.Tensor(A_np, device='vulkan')
    vnn_B = vnn.Tensor(B_np, device='vulkan')
    vnn_C = vnn_A @ vnn_B
    
    print("Difference between VNN Matmul and Torch Matmul:")
    max_diff = np.max(np.abs(vnn_C.to_numpy() - tt_C.numpy()))
    print(f"Max diff: {max_diff}")
    
    # Check linear with custom weights
    layer = torch.nn.Linear(K_dim, N)
    x = torch.from_numpy(A_np)
    y = layer(x)

    vlinear = vnn.Linear(K_dim, N)
    vlinear.weight.load_from_numpy(layer.weight.detach().numpy().T)
    vlinear.bias.load_from_numpy(layer.bias.detach().numpy())
    
    diff_w = np.max(np.abs(vlinear.weight.to_numpy() - layer.weight.detach().numpy().T))
    diff_b = np.max(np.abs(vlinear.bias.to_numpy() - layer.bias.detach().numpy()))
    print(f"Max diff in loaded Weights: {diff_w}")
    print(f"Max diff in loaded Biases: {diff_b}")
    
    vy = vlinear(vnn_A)
    
    max_diff_lin = np.max(np.abs(vy.to_numpy() - y.detach().numpy()))
    print(f"Max diff Linear layer: {max_diff_lin}")
    
    # 3. Step by step Addition
    tt_bias = layer.bias
    tt_out = tt_C + tt_bias
    
    vnn_bias = vnn.Tensor(layer.bias.detach().numpy(), device='cpu')
    vnn_out = vnn_C + vnn_bias
    
    diff_add = np.max(np.abs(vnn_out.to_numpy() - tt_out.detach().numpy()))
    print(f"Max diff manual add: {diff_add}")
    
    # 4. Manual Linear Layer calculation using direct VNN Tensors
    vnn_w = vnn.Tensor(layer.weight.detach().numpy().T, device='cpu')
    vnn_b = vnn.Tensor(layer.bias.detach().numpy(), device='cpu')
    
    vy_manual = vnn_A @ vnn_w + vnn_b
    diff_manual_lin = np.max(np.abs(vy_manual.to_numpy() - y.detach().numpy()))
    print(f"Max diff manual linear: {diff_manual_lin}")

if __name__ == "__main__":
    test_raw_matmul()
