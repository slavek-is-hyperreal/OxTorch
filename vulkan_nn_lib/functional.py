from .tensor import Tensor
from . import kernels as K
import numpy as np

def relu(x: Tensor) -> Tensor:
    return x.relu()

def silu(x: Tensor) -> Tensor:
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

def leaky_relu(x: Tensor, alpha=0.01) -> Tensor:
    res = x.clone()
    K.k_leaky_relu_1d(res.arr, res.total_size, alpha)
    res._prev = {x}
    res.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            if x.grad is None: x.zero_grad()
            K.k_leaky_relu_backward(x.arr, res.grad.arr, x.grad.arr, x.total_size, alpha)
    res._backward_fn = _backward
    return res

def softmax(x: Tensor, dim=-1) -> Tensor:
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

def gelu_tanh(x: Tensor) -> Tensor:
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

GELUTanh = gelu_tanh

class F:
    relu = staticmethod(relu)
    leaky_relu = staticmethod(leaky_relu)
    softmax = staticmethod(softmax)
    silu = staticmethod(silu)
    gelu_tanh = staticmethod(gelu_tanh)
