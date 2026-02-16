import taichi as ti
import numpy as np
from . import kernels as K

class Tensor:
    """A wrapper around ti.ndarray for Vulkan NN operations with Autograd."""
    def __init__(self, data=None, shape=None, requires_grad=False, device='vulkan'):
        self._shape = shape
        self._requires_grad = False
        self._prev = set()
        self._backward_fn = None
        self.grad = None
        self.device = device
        
        if data is not None:
            if isinstance(data, np.ndarray):
                self._shape = data.shape if shape is None else shape
                self.np_arr = data.astype(np.float32).flatten()
            elif isinstance(data, (list, tuple)):
                self.np_arr = np.array(data, dtype=np.float32).flatten()
                self._shape = (len(data),) if shape is None else shape
            elif isinstance(data, (Tensor, ti.Ndarray, np.ndarray)):
                if isinstance(data, Tensor):
                    self.arr = data.arr
                    self._shape = data.shape
                    self.device = data.device
                    if self.device == 'ram' or self.device == 'cpu': self.np_arr = data.np_arr
                elif isinstance(data, ti.Ndarray):
                    self.arr = data
                    self._shape = data.shape if shape is None else shape
                    self.device = 'vulkan'
                else: # numpy array
                    self._shape = data.shape if shape is None else shape
                    self.np_arr = data.astype(np.float32).flatten()
            else: # Scalar
                self.np_arr = np.array([data], dtype=np.float32).flatten()
                self._shape = (1,) if shape is None else shape
            
            if self.device == 'vulkan':
                self.arr = ti.ndarray(dtype=ti.f32, shape=(self.total_size,))
                if hasattr(self, 'np_arr'):
                    self.arr.from_numpy(self.np_arr)
                    ti.sync()
                    # Forced round-trip to guarantee synchronization
                    _ = self.arr.to_numpy()
            else: # device == 'ram' or 'cpu'
                if hasattr(self, 'np_arr'): self.arr = self.np_arr
                else: self.arr = np.zeros(self.total_size, dtype=np.float32)
        else:
            if self.device == 'vulkan':
                self.arr = ti.ndarray(dtype=ti.f32, shape=(self.total_size,))
            else:
                self.arr = np.zeros(self.total_size, dtype=np.float32)

        self.requires_grad = requires_grad

    @property
    def requires_grad(self): return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, val):
        self._requires_grad = val
        if val and self.grad is None: self.zero_grad()

    def zero_grad(self):
        if self.grad is None:
            self.grad = Tensor(None, shape=self.shape, device=self.device)
        if self.device == 'vulkan':
            K.k_zero(self.grad.arr, self.total_size)
        else:
            self.grad.arr.fill(0.0)

    def backward(self, grad=None):
        if grad is None:
            if self.total_size != 1:
                raise RuntimeError("backward() can only be called on scalar outputs or with explicit grad.")
            grad = Tensor([1.0], shape=self.shape)
        elif not isinstance(grad, Tensor):
            grad = Tensor(grad, shape=self.shape)
        
        if self.grad is None: self.grad = grad
        else: K.k_add(self.grad.arr, grad.arr, self.total_size, grad.total_size)

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: build_topo(child)
                topo.append(v)
        build_topo(self)

        for v in reversed(topo):
            if v._backward_fn:
                v._backward_fn()
                if v.device == 'vulkan': ti.sync()

    def to_numpy(self):
        if self.device == 'vulkan':
            ti.sync()
            return self.arr.to_numpy().reshape(self.shape)
        return self.arr.reshape(self.shape)

    @property
    def total_size(self):
        sz = 1
        for s in self.shape: sz *= s
        return sz

    @property
    def shape(self): return self._shape
    @shape.setter
    def shape(self, val): self._shape = val

    def clone(self):
        new_t = Tensor(None, shape=self.shape)
        K.k_copy(self.arr, new_t.arr, self.total_size)
        return new_t

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
        new_shape = list(shape)
        if -1 in new_shape:
            idx = new_shape.index(-1)
            known = 1
            for i, s in enumerate(new_shape):
                if i != idx: known *= s
            new_shape[idx] = self.total_size // known
        res = Tensor(self.arr, shape=tuple(new_shape), requires_grad=self.requires_grad)
        res._prev = {self}
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
                K.k_add(self.grad.arr, res.grad.arr, self.total_size, res.total_size)
        res._backward_fn = _backward
        return res

    def relu(self):
        res = Tensor(None, shape=self.shape)
        K.k_copy(self.arr, res.arr, self.total_size)
        K.k_relu_1d(res.arr, self.total_size)
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
                K.k_relu_backward(self.arr, res.grad.arr, self.grad.arr, self.total_size)
        res._backward_fn = _backward
        return res

    def matmul(self, other):
        M, K_dim = self.shape
        K2, N = other.shape
        res = Tensor(None, shape=(M, N))
        K.k_matmul(self.arr, other.arr, res.arr, M, N, K_dim)
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
            if other.requires_grad:
                if other.grad is None: other.zero_grad()
            grad_a = self.grad.arr if self.requires_grad else Tensor(None, shape=self.shape).arr
            grad_b = other.grad.arr if other.requires_grad else Tensor(None, shape=other.shape).arr
            if self.requires_grad or other.requires_grad:
                K.k_matmul_backward(res.grad.arr, self.arr, other.arr, grad_a, grad_b, M, N, K_dim)
        res._backward_fn = _backward
        return res

    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, shape=self.shape)
        res = Tensor(None, shape=self.shape)
        K.k_copy(self.arr, res.arr, self.total_size)
        K.k_add(res.arr, other.arr, res.total_size, other.total_size)
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
            if other.requires_grad:
                if other.grad is None: other.zero_grad()
            if self.requires_grad and other.requires_grad:
                K.k_add_backward(res.grad.arr, self.grad.arr, other.grad.arr, res.total_size, other.total_size)
            elif self.requires_grad:
                K.k_add(self.grad.arr, res.grad.arr, self.total_size, res.total_size)
            elif other.requires_grad:
                dummy = Tensor(None, shape=self.shape)
                K.k_add_backward(res.grad.arr, dummy.arr, other.grad.arr, res.total_size, other.total_size)
            # print(f"DEBUG: va.grad sum after kernel: {self.grad.to_numpy().sum()}")
        res._backward_fn = _backward
        return res

    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, shape=self.shape)
        res = Tensor(None, shape=self.shape)
        K.k_copy(self.arr, res.arr, self.total_size)
        K.k_sub(res.arr, other.arr, res.total_size, other.total_size)
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
            if other.requires_grad:
                if other.grad is None: other.zero_grad()
            if self.requires_grad and other.requires_grad:
                K.k_sub_backward(res.grad.arr, self.grad.arr, other.grad.arr, res.total_size, other.total_size)
            elif self.requires_grad:
                K.k_add(self.grad.arr, res.grad.arr, self.total_size, res.total_size)
            elif other.requires_grad:
                dummy = Tensor(None, shape=self.shape)
                K.k_sub_backward(res.grad.arr, dummy.arr, other.grad.arr, res.total_size, other.total_size)
        res._backward_fn = _backward
        return res

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            val = float(other)
            res = self.clone()
            K.k_scale(res.arr, val, res.total_size)
            res._prev = {self}
            res.requires_grad = self.requires_grad
            def _backward():
                if self.requires_grad:
                    K.k_scale_backward(res.grad.arr, val, self.grad.arr, self.total_size)
            res._backward_fn = _backward
            return res
        res = self.clone()
        K.k_mul(res.arr, other.arr, res.total_size, other.total_size)
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
            if other.requires_grad:
                if other.grad is None: other.zero_grad()
            if self.requires_grad and other.requires_grad:
                K.k_mul_backward(res.grad.arr, self.arr, other.arr, self.grad.arr, other.grad.arr, res.total_size, other.total_size)
            elif self.requires_grad:
                dummy_grad = Tensor(None, shape=other.shape)
                K.k_mul_backward(res.grad.arr, self.arr, other.arr, self.grad.arr, dummy_grad.arr, res.total_size, other.total_size)
            elif other.requires_grad:
                dummy_grad = Tensor(None, shape=self.shape)
                K.k_mul_backward(res.grad.arr, self.arr, other.arr, dummy_grad.arr, other.grad.arr, res.total_size, other.total_size)
        res._backward_fn = _backward
        return res

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            val = float(other)
            res = self.clone()
            K.k_scale(res.arr, 1.0 / val, res.total_size)
            res._prev = {self}
            res.requires_grad = self.requires_grad
            def _backward():
                if self.requires_grad:
                    K.k_scale_backward(res.grad.arr, 1.0 / val, self.grad.arr, self.total_size)
            res._backward_fn = _backward
            return res
        res = self.clone()
        K.k_div(res.arr, other.arr, res.total_size, other.total_size)
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
                temp_grad = res.grad.clone()
                K.k_div(temp_grad.arr, other.arr, res.total_size, other.total_size)
                K.k_add(self.grad.arr, temp_grad.arr, self.total_size, self.total_size)
            if other.requires_grad:
                if other.grad is None: other.zero_grad()
                temp = res.clone()
                K.k_mul(temp.arr, res.grad.arr, res.total_size, res.total_size)
                K.k_scale(temp.arr, -1.0, temp.total_size)
                K.k_div(temp.arr, other.arr, res.total_size, other.total_size)
                K.k_add(other.grad.arr, temp.arr, res.total_size, other.total_size)
        res._backward_fn = _backward
        return res

    def __matmul__(self, other): return self.matmul(other)
    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)

    def mean(self, dim=None, keepdim=False):
        if dim == -1 or dim == len(self.shape) - 1:
            N = self.shape[-1]
            M = self.total_size // N
            out = Tensor(None, shape=self.shape[:-1] + (1,) if keepdim else self.shape[:-1])
            K.k_mean_last_dim(self.arr, out.arr, M, N)
            return out
        return Tensor(float(np.mean(self.to_numpy())))

    def sqrt(self):
        res = self.clone()
        K.k_sqrt(res.arr, res.total_size)
        return res

    def tanh(self):
        res = self.clone()
        K.k_tanh(res.arr, res.total_size)
        return res

    def t(self):
        if len(self.shape) != 2: return self
        H, W = self.shape
        res = Tensor(None, shape=(W, H))
        K.k_transpose_2d(self.arr, res.arr, H, W)
        return res
