from .tensor import Tensor
from .modules.layers import ReLU, LeakyReLU, Softmax, SiLU
from .modules.base import Module
from . import kernels as K

class GELUTanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        K.k_gelu_tanh(x.arr, x.total_size)
        return x

class F:
    @staticmethod
    def relu(x): return ReLU()(x)
    @staticmethod
    def leaky_relu(x, alpha=0.01): return LeakyReLU(alpha)(x)
    @staticmethod
    def softmax(x, dim=-1): return Softmax(dim)(x)
    @staticmethod
    def silu(x): return SiLU()(x)
    # ... other functional wrappers
