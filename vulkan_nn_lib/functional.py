from .tensor import Tensor
from .streaming_ops import SOE
from .tensor import Tensor
import numpy as np
import taichi as ti

def relu(x: Tensor) -> Tensor:
    res = SOE.elementwise_op(x, None, 'relu')
    res._prev = {x}
    res.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            if x.grad is None: x.zero_grad()
            mask = SOE.elementwise_op(x, 0.0, 'gt')
            # x.grad += res.grad * mask
            term = SOE.elementwise_op(res.grad, mask, 'mul')
            x._acc_grad(term)
    res._backward_fn = _backward
    return res

def silu(x: Tensor) -> Tensor:
    res = SOE.elementwise_op(x, None, 'silu')
    res._prev = {x}
    res.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            if x.grad is None: x.zero_grad()
            # SiLU'(x) = sig(x) * (1 + x * (1 - sig(x)))
            # We use a specialized kernel alias or explicit SOE chain
            term = SOE.elementwise_op(x, None, 'silu_backward_direct', extra=res.grad)
            x._acc_grad(term)
    res._backward_fn = _backward
    return res

def sigmoid(x: Tensor) -> Tensor:
    res = SOE.elementwise_op(x, None, 'sigmoid')
    res._prev = {x}
    res.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            if x.grad is None: x.zero_grad()
            # Sigmoid'(x) = res * (1 - res)
            term = SOE.elementwise_op(res, None, 'sigmoid_backward', extra=res.grad)
            x._acc_grad(term)
    res._backward_fn = _backward
    return res

def leaky_relu(x: Tensor, alpha=0.01) -> Tensor:
    res = SOE.elementwise_op(x, None, 'leaky_relu', extra=alpha)
    res._prev = {x}
    res.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            if x.grad is None: x.zero_grad()
            mask = SOE.elementwise_op(x, 0.0, 'gt')
            # grad = res.grad * (mask + (1-mask)*alpha)
            # Simplified: leaky_relu_backward(x, grad_out, alpha)
            term = SOE.elementwise_op(x, res.grad, 'leaky_relu_backward', extra=alpha)
            x._acc_grad(term)
    res._backward_fn = _backward
    return res

def softmax(x: Tensor, dim=-1) -> Tensor:
    # Softmax is still a bit tricky for tiling, keep it as is if it works
    # But it was failing on CPU sometimes.
    from . import kernels as K
    shape = x.shape
    N = shape[-1]
    M = x.total_size // N
    res = Tensor(None, shape=shape, device=x.device, dtype=x.dtype)
    K.k_softmax_1d(x.arr, res.arr, M, N)
    if x.device == 'vulkan': ti.sync()
    res._prev = {x}
    res.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            if x.grad is None: x.zero_grad()
            K.k_softmax_backward(res.arr, res.grad.arr, x.grad.arr, M, N)
            if x.device == 'vulkan': ti.sync()
    res._backward_fn = _backward
    return res

def gelu_tanh(x: Tensor) -> Tensor:
    res = SOE.elementwise_op(x, None, 'gelu_tanh')
    res._prev = {x}
    res.requires_grad = x.requires_grad
    def _backward():
        if x.requires_grad:
            if x.grad is None: x.zero_grad()
            term = SOE.elementwise_op(x, None, 'gelu_tanh_backward', extra=res.grad)
            x._acc_grad(term)
    res._backward_fn = _backward
    return res

GELUTanh = gelu_tanh

class F:
    relu = staticmethod(relu)
    leaky_relu = staticmethod(leaky_relu)
    softmax = staticmethod(softmax)
    silu = staticmethod(silu)
    sigmoid = staticmethod(sigmoid)
    gelu_tanh = staticmethod(gelu_tanh)
