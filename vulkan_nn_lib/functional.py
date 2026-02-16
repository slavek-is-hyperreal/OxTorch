from .tensor import Tensor
from .modules.layers import ReLU, LeakyReLU, Softmax, SiLU
from .modules.base import Module
from . import kernels as K

class GELUTanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        res = x.clone()
        K.k_gelu_tanh(res.arr, res.total_size)
        res._prev = {x}
        res.requires_grad = x.requires_grad
        def _backward():
            if x.requires_grad:
                if x.grad is None: x.zero_grad()
                K.k_gelu_tanh_backward(x.arr, res.grad.arr, x.grad.arr, x.total_size)
        res._backward_fn = _backward
        return res

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
