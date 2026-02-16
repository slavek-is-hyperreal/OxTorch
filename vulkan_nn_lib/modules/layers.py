import numpy as np
from ..tensor import Tensor
from .base import Module
from .. import kernels as K

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * 0.02, requires_grad=True)
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        M = 1
        for s in orig_shape[:-1]: M *= s
        x_flat = x.reshape(M, self.in_features)
        out = x_flat @ self.weight
        if self.has_bias: out = out + self.bias
        return out.reshape(*(list(orig_shape[:-1]) + [self.out_features]))

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        res = x.clone()
        K.k_silu_1d(res.arr, res.total_size)
        res._prev = {x}
        res.requires_grad = x.requires_grad
        def _backward():
            if x.requires_grad:
                if x.grad is None: x.zero_grad()
                K.k_silu_backward(x.arr, res.grad.arr, x.grad.arr, x.total_size)
        res._backward_fn = _backward
        return res

class RMSNorm(Module):
    def __init__(self, dim, eps=1e-6, add_unit_offset=True):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = Tensor(np.zeros(dim, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        N = shape[-1]
        M = x.total_size // N
        res = Tensor(None, shape=shape)
        K.k_rmsnorm_1d(x.arr, self.weight.arr, res.arr, M, N, self.eps, 1 if self.add_unit_offset else 0)
        res._prev = {x, self.weight}
        res.requires_grad = x.requires_grad or self.weight.requires_grad
        def _backward():
            if x.requires_grad:
                if x.grad is None: x.zero_grad()
            if self.weight.requires_grad:
                if self.weight.grad is None: self.weight.zero_grad()
            if x.requires_grad or self.weight.requires_grad:
                grad_x = x.grad.arr if x.requires_grad else Tensor(None, shape=x.shape, device=x.device).arr
                grad_w = self.weight.grad.arr if self.weight.requires_grad else Tensor(None, shape=self.weight.shape, device=self.weight.device).arr
                K.k_rmsnorm_backward(x.arr, self.weight.arr, res.grad.arr, grad_x, grad_w, M, N, self.eps, 1 if self.add_unit_offset else 0)
        res._backward_fn = _backward
        return res

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        N = shape[-1]
        M = x.total_size // N
        res = Tensor(None, shape=shape)
        K.k_softmax_1d(x.arr, res.arr, M, N)
        res._prev = {x}
        res.requires_grad = x.requires_grad
        def _backward():
            if x.requires_grad:
                if x.grad is None: x.zero_grad()
                K.k_softmax_backward(res.arr, res.grad.arr, x.grad.arr, M, N)
        res._backward_fn = _backward
        return res

class LeakyReLU(Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    def forward(self, x: Tensor) -> Tensor:
        res = x.clone()
        K.k_leaky_relu_1d(res.arr, res.total_size, self.alpha)
        res._prev = {x}
        res.requires_grad = x.requires_grad
        def _backward():
            if x.requires_grad:
                if x.grad is None: x.zero_grad()
                K.k_leaky_relu_backward(x.arr, res.grad.arr, x.grad.arr, x.total_size, self.alpha)
        res._backward_fn = _backward
        return res

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.kh, self.kw = kh, kw
        self.weight = Tensor(np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * 0.1, requires_grad=True)
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        b, ic, h, w = x.shape
        oh, ow = h - self.kh + 1, w - self.kw + 1
        out = Tensor(None, shape=(b, self.out_channels, oh, ow))
        K.k_conv2d_1d(x.arr, self.weight.arr, self.bias.arr if self.has_bias else None, out.arr, b, ic, self.out_channels, h, w, self.kh, self.kw)
        return out

class MaxPool2d(Module):
    def __init__(self, kernel_size=2):
        super().__init__()
        self.k = kernel_size
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        out = Tensor(None, shape=(b, c, h // self.k, w // self.k))
        K.k_pool2d_1d(x.arr, out.arr, b, c, h, w, self.k)
        return out

class Upsample(Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale = scale_factor
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        out = Tensor(None, shape=(b, c, h * self.scale, w * self.scale))
        K.k_upsample2d_1d(x.arr, out.arr, b, c, h, w, self.scale)
        return out

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        B, L = x.shape
        D = self.weight.shape[1]
        out = Tensor(None, shape=(B, L, D))
        K.k_embedding_1d(x.arr, self.weight.arr, out.arr, B, L, D)
        return out
