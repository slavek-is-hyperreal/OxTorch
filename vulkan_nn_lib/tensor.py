import taichi as ti
import numpy as np
from . import kernels as K
# Circular dependency avoidance: import SOE locally within methods

class Tensor:
    """A wrapper around ti.ndarray for Vulkan NN operations with Autograd."""
    _tensor_store = None
    _tensor_counter = 0

    @classmethod
    def setup_ssd_storage(cls, path="/vectorlegis_ssd_pool/vnn_cache"):
        from .tensor_store import TensorStore
        cls._tensor_store = TensorStore(path)

    def _get_available_ram(self):
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) * 1024
        except:
            return 8 * 1024 * 1024 * 1024

    def __init__(self, data=None, shape=None, requires_grad=False, device='auto', dtype=None, external_path=None):
        self._shape = shape
        self._requires_grad = False
        self._prev = set()
        self._backward_fn = None
        self.grad = None
        
        # Dtype handling (PyTorch compatibility)
        if dtype is None:
            if isinstance(data, np.ndarray): dtype = data.dtype
            elif isinstance(data, Tensor): dtype = data.dtype
            else: dtype = np.float32 # Default
        
        # Map strings or torch-like dtypes to numpy
        dtype_map = {
            'float32': np.float32, 'float16': np.float16,
            'int32': np.int32, 'int16': np.int16, 'int8': np.int8,
            'float': np.float32, 'double': np.float64,
            'long': np.int64, 'int': np.int32,
            np.float32: np.float32, np.float16: np.float16,
            np.int32: np.int32, np.int16: np.int16, np.int8: np.int8
        }
        self.dtype = dtype_map.get(dtype, dtype)
        
        # Item size for memory calculation
        item_size = np.dtype(self.dtype).itemsize

        # Initial size estimation
        if data is not None:
            if isinstance(data, np.ndarray):
                self._shape = data.shape if shape is None else shape
            elif isinstance(data, (list, tuple)):
                self._shape = (len(data),) if shape is None else shape
            elif isinstance(data, Tensor):
                self._shape = data.shape
            elif isinstance(data, ti.Ndarray):
                self._shape = data.shape if shape is None else shape
            else: # Scalar
                self._shape = (1,) if shape is None else shape
        
        n = self.total_size
        size_bytes = n * item_size

        # Auto-device selection
        if device == 'auto':
            if external_path:
                device = 'ssd'
            # 1. Try Vulkan (only for float32 for now)
            elif self.dtype == np.float32 and size_bytes <= 128 * 1024 * 1024:
                device = 'vulkan'
            else:
                avail = self._get_available_ram()
                # 90% of available as threshold
                if size_bytes <= avail * 0.9: 
                    device = 'cpu'
                else:
                    device = 'ssd'

        self.device = device
        
        # Re-calculate shape just in case (though it should be set)
        if data is not None:
            if isinstance(data, np.ndarray): self._shape = data.shape if shape is None else shape
            elif isinstance(data, (list, tuple)): self._shape = (len(data),) if shape is None else shape
            elif isinstance(data, Tensor): self._shape = data.shape
            elif isinstance(data, ti.Ndarray): self._shape = data.shape if shape is None else shape
            else: self._shape = (1,) if shape is None else shape
        
        n = self.total_size

        if device == 'ssd':
            if Tensor._tensor_store is None:
                Tensor.setup_ssd_storage()
            name = f"t{Tensor._tensor_counter}"
            Tensor._tensor_counter += 1
            # Use external binary file if provided
            self.arr = Tensor._tensor_store.zeros(name, shape=(n,), dtype=self.dtype, external_path=external_path)
            if data is not None:
                # Optimized copy to SSD
                print(f"  [Tensor] Initializing SSD tensor {name} ({n*item_size/1e6:.1f}MB, {self.dtype})...")
                if isinstance(data, Tensor):
                    self.arr[:] = data.to_numpy().flatten().astype(self.dtype)
                elif isinstance(data, (np.ndarray, list, tuple)):
                    self.arr[:] = np.array(data, dtype=self.dtype).flatten()
                else: # scalar
                    self.arr.fill(data)
        elif data is not None:
            if isinstance(data, np.ndarray):
                self.np_arr = data.astype(self.dtype).flatten()
            elif isinstance(data, (list, tuple)):
                self.np_arr = np.array(data, dtype=self.dtype).flatten()
            elif isinstance(data, Tensor):
                self.arr = data.arr
                self.device = data.device
                if self.device in ['ram', 'cpu', 'ssd']: 
                    self.np_arr = data.arr.flatten()
            elif isinstance(data, ti.Ndarray):
                self.np_arr = data.to_numpy().flatten().astype(self.dtype)
                self._shape = data.shape if shape is None else shape
                self.device = 'vulkan' # Default to vulkan if from ti.ndarray
            else: # Scalar
                self.np_arr = np.array([data], dtype=self.dtype).flatten()
            
            if self.device == 'vulkan':
                if not hasattr(self, 'arr'):
                    self.arr = ti.ndarray(dtype=ti.f32, shape=(n,))
                if hasattr(self, 'np_arr'):
                    self.arr.from_numpy(self.np_arr.astype(np.float32))
                    ti.sync()
            else: # device == 'ram' or 'cpu'
                if hasattr(self, 'np_arr'): 
                    # Reuse np_arr directly to avoid copy
                    self.arr = self.np_arr
                else: 
                    self.arr = np.zeros(n, dtype=self.dtype)
        else:
            if self.device == 'vulkan':
                self.arr = ti.ndarray(dtype=ti.f32, shape=(n,))
            else: # ram/cpu
                self.arr = np.zeros(n, dtype=self.dtype)

        self.requires_grad = requires_grad

    @property
    def requires_grad(self): return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, val):
        self._requires_grad = val
        if val and self.grad is None: self.zero_grad()

    def zero_(self):
        """In-place zeroing of the tensor."""
        if self.device == 'vulkan':
            K.k_zero(self.arr, self.total_size)
            ti.sync()
        else:
            self.arr.fill(0.0)

    def _as_np(self):
        """Helper to get a numpy-reshaped view for safe manipulation."""
        if self.device == 'vulkan': return self.to_numpy()
        return self.arr.reshape(self.shape)

    def __getitem__(self, idx):
        # Handle slicing and indexing
        if self.device == 'vulkan':
            # For simplicity, move to CPU for indexing if on Vulkan
            return Tensor(self.to_numpy()[idx])
        
        # CPU or SSD (both are numpy-backed)
        val = self._as_np()[idx]
        if isinstance(val, (np.ndarray, np.memmap)):
            return Tensor(val, device=self.device, dtype=self.dtype)
        else:
            # Scalar result
            return Tensor(np.array([val], dtype=self.dtype), shape=(), device=self.device, dtype=self.dtype)

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            value = value.to_numpy()
        
        if self.device == 'vulkan':
            # Synchronize for setitem
            tmp = self.to_numpy()
            tmp[idx] = value
            self.load_from_numpy(tmp)
        else:
            # CPU or SSD
            view = self.arr.reshape(self.shape)
            view[idx] = value

    def zero_grad(self):
        if self.grad is None:
            self.grad = Tensor(None, shape=self.shape, device=self.device)
        self.grad.zero_()

    def backward(self, grad=None):
        if grad is None:
            if self.total_size != 1:
                raise RuntimeError("backward() can only be called on scalar outputs or with explicit grad.")
            grad = Tensor([1.0], shape=self.shape)
        elif not isinstance(grad, Tensor):
            grad = Tensor(grad, shape=self.shape)
        
        if self.grad is None: self.grad = grad
        else:
            K.k_add(self.grad.arr, grad.arr, self.total_size, grad.total_size)
            if self.device == 'vulkan': ti.sync()

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

    def load_from_numpy(self, np_arr):
        if self.device == 'vulkan':
            self.arr.from_numpy(np_arr.flatten())
            ti.sync()
            _ = self.arr.to_numpy() # Force sync
        else:
            self.arr = np_arr.astype(np.float32).flatten()
            if hasattr(self, 'np_arr'): self.np_arr = self.arr

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
        new_t = Tensor(None, shape=self.shape, device=self.device, dtype=self.dtype)
        if self.device == 'ssd':
            # Tiled copy for SSD to avoid Taichi/VRAM allocation errors
            n = self.total_size
            chunk = 128 * 1024 * 1024 # 128M elements
            for i in range(0, n, chunk):
                end = min(i + chunk, n)
                new_t.arr[i:end] = self.arr[i:end]
            return new_t
        K.k_copy(self.arr, new_t.arr, self.total_size)
        if self.device == 'vulkan': ti.sync()
        return new_t
    
    def transpose(self, dim0, dim1):
        if len(self.shape) < 2: return self
        if self.device == 'vulkan':
            if len(self.shape) == 2 and self.total_size < 100 * 1024 * 1024:
                H, W = self.shape
                res = Tensor(None, shape=(W, H), device='vulkan')
                K.k_transpose_2d(self.arr, res.arr, H, W)
                ti.sync()
                return res
            return Tensor(self.to_numpy().swapaxes(dim0, dim1))
        
        # CPU/SSD: use numpy swapaxes and force contiguous if SSD
        np_swap = self._as_np().swapaxes(dim0, dim1)
        if self.device == 'ssd':
            res = Tensor(None, shape=np_swap.shape, device='ssd', dtype=self.dtype)
            res.arr[:] = np_swap.flatten()
            return res
        return Tensor(np_swap.copy())

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)): dims = dims[0]
        if self.device == 'vulkan':
            return Tensor(self.to_numpy().transpose(dims))
            
        np_p = self._as_np().transpose(dims)
        if self.device == 'ssd':
            res = Tensor(None, shape=np_p.shape, device='ssd', dtype=self.dtype)
            print(f"  [Tensor.permute] Re-laying out SSD tensor ({self.total_size*4/1e6:.1f}MB)...")
            res.arr[:] = np_p.flatten()
            return res
        return Tensor(np_p.copy())

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)): shape = shape[0]
        np_e = np.broadcast_to(self._as_np(), shape)
        if self.device == 'ssd':
            res = Tensor(None, shape=shape, device='ssd', dtype=self.dtype)
            res.arr[:] = np_e.flatten()
            return res
        return Tensor(np_e.copy())

    def flatten(self, start_dim=0, end_dim=-1):
        curr_shape = list(self.shape)
        if end_dim == -1: end_dim = len(curr_shape) - 1
        
        new_shape = curr_shape[:start_dim]
        mid = 1
        for i in range(start_dim, end_dim + 1):
            mid *= curr_shape[i]
        new_shape.append(mid)
        new_shape.extend(curr_shape[end_dim + 1:])
        return self.reshape(tuple(new_shape))

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
                if self.device == 'vulkan': ti.sync()
        res._backward_fn = _backward
        return res

    def relu(self):
        res = Tensor(None, shape=self.shape)
        K.k_copy(self.arr, res.arr, self.total_size)
        K.k_relu_1d(res.arr, self.total_size)
        if self.device == 'vulkan': ti.sync()
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
                K.k_relu_backward(self.arr, res.grad.arr, self.grad.arr, self.total_size)
                if self.device == 'vulkan': ti.sync()
        res._backward_fn = _backward
        return res

    def matmul(self, other):
        if self.device == 'ssd' or other.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.matmul(self, other)
        M, K_dim = self.shape
        K2, N = other.shape
        res = Tensor(None, shape=(M, N))
        K.k_matmul(self.arr, other.arr, res.arr, M, N, K_dim)
        if self.device == 'vulkan': ti.sync()
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
                if self.device == 'vulkan': ti.sync()
        res._backward_fn = _backward
        return res

    def __add__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, shape=self.shape, dtype=self.dtype)
        if self.device == 'ssd' or other.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, other, 'add')
        res = Tensor(None, shape=self.shape, dtype=self.dtype)
        K.k_copy(self.arr, res.arr, self.total_size)
        K.k_add(res.arr, other.arr, res.total_size, other.total_size)
        if self.device == 'vulkan': ti.sync()
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
            if self.device == 'vulkan': ti.sync()
            # print(f"DEBUG: va.grad sum after kernel: {self.grad.to_numpy().sum()}")
        res._backward_fn = _backward
        return res

    def __sub__(self, other):
        if not isinstance(other, Tensor): other = Tensor(other, shape=self.shape, dtype=self.dtype)
        if self.device == 'ssd' or other.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, other, 'sub')
        res = Tensor(None, shape=self.shape, dtype=self.dtype)
        K.k_copy(self.arr, res.arr, self.total_size)
        K.k_sub(res.arr, other.arr, res.total_size, other.total_size)
        if self.device == 'vulkan': ti.sync()
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
            if self.device == 'vulkan': ti.sync()
        res._backward_fn = _backward
        return res

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            val = float(other)
            if self.device == 'ssd':
                from . import streaming_ops as SOE
                return SOE.SOE.elementwise_op(self, val, 'mul')
            res = self.clone()
            K.k_scale(res.arr, val, res.total_size)
            if self.device == 'vulkan': ti.sync()
            res._prev = {self}
            res.requires_grad = self.requires_grad
            def _backward():
                if self.requires_grad:
                    if self.grad is None: self.zero_grad()
                    K.k_scale_backward(res.grad.arr, val, self.grad.arr, self.total_size)
                    if self.device == 'vulkan': ti.sync()
            res._backward_fn = _backward
            return res
        if self.device == 'ssd' or other.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, other, 'mul')
        res = self.clone()
        K.k_mul(res.arr, other.arr, res.total_size, other.total_size)
        if self.device == 'vulkan': ti.sync()
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
            if self.device == 'vulkan': ti.sync()
        res._backward_fn = _backward
        return res

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            val = float(other)
            if self.device == 'ssd':
                from . import streaming_ops as SOE
                return SOE.SOE.elementwise_op(self, val, 'div')
            res = self.clone()
            K.k_scale(res.arr, 1.0 / val, res.total_size)
            if self.device == 'vulkan': ti.sync()
            res._prev = {self}
            res.requires_grad = self.requires_grad
            def _backward():
                if self.requires_grad:
                    if self.grad is None: self.zero_grad()
                    K.k_scale_backward(res.grad.arr, 1.0 / val, self.grad.arr, self.total_size)
                    if self.device == 'vulkan': ti.sync()
            res._backward_fn = _backward
            return res
        if self.device == 'ssd' or other.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, other, 'div')
        res = self.clone()
        K.k_div(res.arr, other.arr, res.total_size, other.total_size)
        if self.device == 'vulkan': ti.sync()
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
            if self.device == 'vulkan': ti.sync()
        res._backward_fn = _backward
        return res

    def __matmul__(self, other): return self.matmul(other)
    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)

    def _comp_op(self, other, op_type):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=np.float32), device='cpu')
            
        if self.device == 'ssd' or other.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, other, op_type)
            
        res = Tensor(None, shape=self.shape, device=self.device)
        kernel = getattr(K, f"k_{op_type}")
        kernel(self.arr, other.arr, res.arr, self.total_size, other.total_size)
        if self.device == 'vulkan': ti.sync()
        return res

    def __gt__(self, other): return self._comp_op(other, 'gt')
    def __lt__(self, other): return self._comp_op(other, 'lt')
    def __ge__(self, other): return self._comp_op(other, 'ge')
    def __le__(self, other): return self._comp_op(other, 'le')
    def __eq__(self, other): return self._comp_op(other, 'eq')
    def __ne__(self, other): return self._comp_op(other, 'ne')

    def mean(self, dim=None, keepdim=False):
        if dim == -1 or dim == len(self.shape) - 1:
            N = self.shape[-1]
            M = self.total_size // N
            out = Tensor(None, shape=self.shape[:-1] + (1,) if keepdim else self.shape[:-1])
            K.k_mean_last_dim(self.arr, out.arr, M, N)
            if self.device == 'vulkan': ti.sync()
            return out
        return Tensor(float(np.mean(self.to_numpy())))

    def sqrt(self):
        if self.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, None, 'sqrt')
        res = self.clone()
        K.k_sqrt(res.arr, res.total_size)
        if self.device == 'vulkan': ti.sync()
        return res

    def exp(self):
        if self.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, None, 'exp')
        return Tensor(np.exp(self.to_numpy()), device=self.device)

    def log(self):
        if self.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, None, 'log')
        return Tensor(np.log(self.to_numpy()), device=self.device)

    def pow(self, val):
        if self.device == 'ssd':
            from . import streaming_ops as SOE
            return SOE.SOE.elementwise_op(self, val, 'pow')
        return Tensor(np.power(self.to_numpy(), val), device=self.device)

    def masked_fill(self, mask, value):
        if self.device == 'ssd':
            from . import streaming_ops as SOE
            # Pass (mask, value) as 'b' argument
            m_np = mask.to_numpy() if isinstance(mask, Tensor) else mask
            return SOE.SOE.elementwise_op(self, (m_np, value), 'masked_fill')
        res = self.to_numpy()
        m = mask.to_numpy() if isinstance(mask, Tensor) else mask
        # Ensure mask is boolean for numpy indexing
        res[m > 0.5] = value
        return Tensor(res, device=self.device)

    def gather(self, dim, index):
        # Implementation for 1:1 parity
        if self.device == 'ssd':
            # For SSD, we do a greedy gather. 
            # Note: This is simplified and might be slow for complex indices
            idx_np = index.to_numpy()
            out_shape = index.shape
            print(f"  [Tensor.gather] SSD gathering to {out_shape}...")
            # Use numpy but keep it in blocks if needed. 
            # For now, let's use the simplest greedy approach.
            res_np = np.take_along_axis(self.arr.reshape(self.shape), idx_np, axis=dim)
            return Tensor(res_np, device='ssd' if self.total_size > 1e8 else 'auto')
        
        return Tensor(np.take_along_axis(self.to_numpy(), index.to_numpy(), axis=dim), device=self.device)

    def tanh(self):
        res = self.clone()
        K.k_tanh(res.arr, res.total_size)
        if self.device == 'vulkan': ti.sync()
        return res

    def t(self):
        if len(self.shape) != 2: return self
        H, W = self.shape
        res = Tensor(None, shape=(W, H))
        K.k_transpose_2d(self.arr, res.arr, H, W)
        if self.device == 'vulkan': ti.sync()
        return res
    def sum(self):
        # Result is a scalar tensor
        res = Tensor(float(np.sum(self.to_numpy())), shape=(1,))
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                if self.grad is None: self.zero_grad()
                # Gradient of sum is 1.0 everywhere (multiplied by incoming gradient)
                # We use a kernel to add the scalar value to all elements of the gradient
                if res.grad is not None:
                    # Incoming grad is scalar (1,). We broadcast it to self.shape
                    K.k_add(self.grad.arr, res.grad.arr, self.total_size, 1)
                    if self.device == 'vulkan': ti.sync()
        res._backward_fn = _backward
        return res

    def unsqueeze(self, dim):
        new_shape = list(self.shape)
        if dim < 0: dim += len(new_shape) + 1
        new_shape.insert(dim, 1)
        return self.reshape(*new_shape)
