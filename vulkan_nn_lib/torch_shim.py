import sys
import types
from . import core as vnn
import numpy as np

# 1. Create real module objects for the shim
torch = types.ModuleType("torch")
nn = types.ModuleType("torch.nn")
functional = types.ModuleType("torch.nn.functional")
optim = types.ModuleType("torch.optim")

# 2. Populate functional (torch.nn.functional)
functional.relu = lambda x, inplace=False: x.relu()
functional.leaky_relu = vnn.F.leaky_relu
functional.max_pool2d = vnn.F.max_pool2d
functional.gelu = lambda x, approximate=None: x.silu() if approximate is None else vnn.GELUTanh()(x)
functional.silu = lambda x: x.silu()
functional.softmax = lambda x, dim=-1: vnn.Softmax(dim)(x)
functional.linear = lambda x, w, b=None: vnn.Linear(w.shape[1], w.shape[0])(x).from_numpy(w.to_numpy())
functional.embedding = lambda x, w: vnn.Embedding(w.shape[0], w.shape[1])(x).from_numpy(w.to_numpy())
functional.rms_norm = lambda x, w, eps=1e-6: vnn.RMSNorm(w.shape[0], eps=eps)(x)

# 3. Populate optim
optim.SGD = vnn.SGD
optim.Adam = vnn.Adam

# 4. Populate nn (torch.nn)
nn.Module = vnn.Module
nn.Linear = vnn.Linear
nn.TiledLinear = vnn.TiledLinear
nn.Conv2d = vnn.Conv2d
nn.ReLU = vnn.ReLU
nn.LeakyReLU = vnn.LeakyReLU
nn.GELUTanh = vnn.GELUTanh
nn.Softmax = vnn.Softmax
nn.RMSNorm = vnn.RMSNorm
nn.Embedding = vnn.Embedding
nn.MaxPool2d = vnn.MaxPool2d
nn.Upsample = vnn.Upsample
nn.Sequential = vnn.Sequential
nn.ModuleList = vnn.ModuleList

class Parameter(vnn.Tensor):
    def __new__(cls, data, requires_grad=True):
        if isinstance(data, vnn.Tensor):
            data.requires_grad = requires_grad
            data.__class__ = cls
            return data
        return vnn.Tensor(data, requires_grad=requires_grad)
nn.Parameter = Parameter
nn.functional = functional

# 5. Populate torch
torch.nn = nn
torch.optim = optim
torch.Tensor = vnn.Tensor
torch.from_numpy = lambda x: vnn.Tensor(x, requires_grad=False)
torch.as_tensor = lambda x, **kw: vnn.Tensor(x)
torch.load = lambda path, **kwargs: np.load(path, allow_pickle=True).item() if path.endswith(".npy") else {}
torch.device = lambda name: name
torch.float32 = vnn.ti.f32
torch.float16 = vnn.ti.f32
torch.float = vnn.ti.f32
torch.int64 = vnn.ti.i32
torch.int32 = vnn.ti.i32
torch.bool = vnn.ti.i32

torch.rsqrt = lambda x: x.pow(-0.5)
torch.sqrt = lambda x: x.sqrt()
torch.tanh = lambda x: x.tanh()
torch.matmul = lambda a, b: a @ b
torch.cat = vnn.Tensor.cat
torch.stack = lambda ts, dim=0: vnn.Tensor.cat([t.unsqueeze(dim) for t in ts], dim=dim)

class NoGrad:
    def __enter__(self): pass
    def __exit__(self, *a): pass
torch.no_grad = lambda: NoGrad()

def hijack_torch():
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = nn
    sys.modules['torch.nn.functional'] = functional
    sys.modules['torch.optim'] = optim
    print("VulkanTorch: Global Hijack Active (including Optimizers)!")

F = functional
Tensor = vnn.Tensor
from_numpy = torch.from_numpy
no_grad = torch.no_grad
load = torch.load
device = torch.device

print("VulkanTorch: Shim loaded. Use hijack_torch() for training & inference.")
