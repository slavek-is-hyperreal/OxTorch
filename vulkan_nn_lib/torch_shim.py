import sys
import types
from . import core as vnn
import numpy as np

# 1. Create real module objects for the shim
torch = types.ModuleType("torch")
nn = types.ModuleType("torch.nn")
functional = types.ModuleType("torch.nn.functional")

# 2. Populate functional (torch.nn.functional)
functional.relu = vnn.F.relu
functional.leaky_relu = vnn.F.leaky_relu
functional.max_pool2d = vnn.F.max_pool2d
functional.silu = lambda x: vnn.SiLU()(x)
functional.softmax = lambda x, dim=-1: vnn.Softmax(dim)(x)

# 3. Populate nn (torch.nn)
nn.Module = vnn.Module
nn.Linear = vnn.Linear
nn.TiledLinear = vnn.TiledLinear
nn.Conv2d = vnn.Conv2d
nn.ReLU = vnn.ReLU
nn.LeakyReLU = vnn.LeakyReLU
nn.SiLU = vnn.SiLU
nn.Softmax = vnn.Softmax
nn.RMSNorm = vnn.RMSNorm
nn.Embedding = vnn.Embedding
nn.MaxPool2d = vnn.MaxPool2d
nn.Upsample = vnn.Upsample
nn.Sequential = vnn.Sequential
nn.Parameter = lambda x: x
nn.functional = functional

# 4. Populate torch
torch.nn = nn
torch.Tensor = vnn.Tensor
torch.from_numpy = lambda x: vnn.Tensor(x, dtype=vnn.ti.i32 if x.dtype in [np.int32, np.int64] else vnn.ti.f32)
torch.load = lambda path: np.load(path, allow_pickle=True).item() if path.endswith(".npy") else {}
torch.device = lambda name: name
torch.no_grad = vnn.Module.to # Mock context manager

class NoGrad:
    def __enter__(self): pass
    def __exit__(self, *a): pass
torch.no_grad = lambda: NoGrad()

# 5. Optional: Hijack sys.modules
def hijack_torch():
    """Call this to force other libraries to use VulkanNN instead of real PyTorch."""
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = nn
    sys.modules['torch.nn.functional'] = functional
    print("VulkanTorch: Global Hijack Active!")

# Export for 'import vulkan_torch as torch'
F = functional
Tensor = vnn.Tensor
from_numpy = torch.from_numpy
no_grad = torch.no_grad
load = torch.load
device = torch.device

print("VulkanTorch: Shim loaded. Use hijack_torch() to replace real PyTorch globally.")
