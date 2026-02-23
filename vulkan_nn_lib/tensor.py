import taichi as ti
import numpy as np
from . import kernels as K
import torch
from .memory import MemoryManager
from . import config

def _get_torch():
    return torch

class Tensor:
    """A wrapper around ti.ndarray for Vulkan NN operations with Autograd."""
    _tensor_store = None
    _tensor_counter = 0

    @classmethod
    def setup_ssd_storage(cls, path=None):
        if path is None:
            path = config.get_ssd_path()
        from .tensor_store import TensorStore
        cls._tensor_store = TensorStore(path)

    @staticmethod
    def from_numpy(data, requires_grad=False):
        """Zero-copy: Share memory with existing numpy array."""
        return Tensor(data, requires_grad=requires_grad, device='cpu')

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
            'int4': 'int4', # Special case
            'float': np.float32, 'double': np.float64,
            'long': np.int64, 'int': np.int32,
            np.float32: np.float32, np.float16: np.float16,
            np.int32: np.int32, np.int16: np.int16, np.int8: np.int8
        }
        self.dtype = dtype_map.get(dtype, dtype)
        
        # Storage dtype handling
        if self.dtype == 'int4':
            self.storage_dtype = np.uint8
        else:
            self.storage_dtype = self.dtype

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
        size_bytes = int(n * self.item_size)

        # Auto-device selection
        if device == 'auto':
            if external_path:
                device = 'ssd'
            else:
                if MemoryManager.should_offload_to_ssd(size_bytes):
                    device = 'ssd'
                elif self.dtype == np.float32 and size_bytes <= 128 * 1024 * 1024:
                    # Small float32 tensors default to vulkan in 'auto' mode
                    device = 'vulkan'
                else:
                    device = 'cpu'
        
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
            num_elements = int(n * self.item_size) if self.dtype == 'int4' else n
            self.arr = Tensor._tensor_store.zeros(name, shape=(num_elements,), dtype=self.storage_dtype, external_path=external_path)
            if data is not None:
                # Optimized copy to SSD
                size_bytes = int(n * self.item_size)
                size_str = f"{size_bytes/1e6:.1f}MB" if size_bytes >= 1e6 else f"{size_bytes/1024:.1f}KB"
                print(f"  [Tensor] Initializing SSD tensor {name} ({size_str}, {self.dtype})...")
                
                if self.dtype == 'int4':
                    # Handle int4 packing
                    val = np.array(data, dtype=np.int8).flatten()
                    packed = (val[0::2] & 0x0F) | ((val[1::2] & 0x0F) << 4)
                    self.arr[:len(packed)] = packed
                elif isinstance(data, Tensor):
                    self.arr[:] = data.to_numpy().flatten().astype(self.dtype)
                elif isinstance(data, (np.ndarray, list, tuple)):
                    self.arr[:] = np.array(data, dtype=self.dtype).flatten()
                else: # scalar
                    self.arr.fill(data)
                
                if hasattr(self.arr, 'flush'): 
                    self.arr.flush()
            
            # CRITICAL: Do not fall through to CPU/VULKAN init!
            self.requires_grad = requires_grad
            self._prev = set()
            self._backward_fn = None
            return
        elif data is not None:
            if self.dtype == 'int4':
                # Packed in-RAM
                val = (np.array(data, dtype=np.float32).flatten() + 8.0).clip(0, 15).astype(np.uint8)
                self.np_arr = (val[0::2] & 0x0F) | ((val[1::2] & 0x0F) << 4)
            elif isinstance(data, np.ndarray):
                if data.dtype == self.dtype:
                    self.np_arr = data.ravel()
                else:
                    self.np_arr = data.astype(self.dtype).ravel()
            elif isinstance(data, (list, tuple)):
                self.np_arr = np.array(data, dtype=self.dtype).ravel()
            elif isinstance(data, Tensor):
                self.arr = data.arr
                self.device = data.device
                if self.device in ['ram', 'cpu', 'ssd']: 
                    self.np_arr = data.arr.ravel()
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
                    if self.dtype == 'int4':
                        # Unpack to f32 for Vulkan
                        unpacked = np.zeros(n, dtype=np.float32)
                        unpacked[0::2] = (self.np_arr & 0x0F).astype(np.float32) - 8.0
                        unpacked[1::2] = ((self.np_arr >> 4) & 0x0F).astype(np.float32) - 8.0
                        self.arr.from_numpy(unpacked)
                    else:
                        self.arr.from_numpy(self.np_arr.astype(np.float32))
                    ti.sync()
            else: # device == 'ram' or 'cpu'
                if hasattr(self, 'np_arr'): 
                    # Reuse np_arr directly to avoid copy
                    self.arr = self.np_arr
                else: 
                    # Use storage_dtype directly. If it's not uint8 (int4), use n. 
                    num_elements = int(n * self.item_size) if self.dtype == 'int4' else n
                    self.arr = np.zeros(num_elements, dtype=self.storage_dtype)
        else:
            if self.device == 'vulkan':
                self.arr = ti.ndarray(dtype=ti.f32, shape=(n,))
            else: # ram/cpu
                num_elements = int(n * self.item_size) if self.dtype == 'int4' else n
                self.arr = np.zeros(num_elements, dtype=self.storage_dtype)

        self.requires_grad = requires_grad
        self._prev = set()
        self._backward_fn = None

    def __hash__(self):
        return id(self)
        
    def __eq__(self, other):
        return self is other

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
            # CPU or SSD
            self.arr.fill(0.0)

    def _as_np(self):
        """Helper to get a numpy-reshaped view for safe manipulation."""
        if self.device == 'vulkan': return self.to_numpy()
        return self.arr.reshape(self.shape)

    def __getitem__(self, idx):
        # Handle slicing and indexing
        if self.device == 'vulkan':
            if self.total_size * self.item_size > 1e8: # 100MB
                 # OOM-Safe: Move only the slice if possible, or use numpy directly via to_numpy(copy=False)
                 # Actually, Taichi doesn't support slicing Ndarrays well without GPU->CPU. 
                 # Best we can do is download to a reusable buffer or tile. 
                 # For sampling in tests, we'll implement a 'sample' method or fix it in test suite.
                 # For now, let's at least prevent the crash by checking size.
                 pass 
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
            self.grad = Tensor(None, shape=self.shape, device=self.device, dtype=self.dtype)
        self.grad.zero_()

    def _acc_grad(self, grad):
        """Unified gradient accumulation across backends (OOM-safe)."""
        if not self.requires_grad: return
        
        if self.grad is None:
            # First time: Initialize gradient on the same device as parameter
            self.grad = Tensor(None, shape=self.shape, device=self.device, dtype=self.dtype)
            self.grad.zero_()
            
        if self.device == 'ssd':
            from . import streaming_ops as SOE
            self.grad = SOE.SOE.elementwise_op(self.grad, grad, 'add', out_device='ssd')
        elif self.device == 'vulkan':
            K.k_add(self.grad.arr, grad.arr, self.total_size, grad.total_size)
            ti.sync()
        else:
            # CPU/RAM: Numpy handles broadcasting
            if isinstance(grad, Tensor):
                g_val = grad.to_numpy()
            else:
                g_val = np.array(grad)
            
            # Use rank-agnostic ellipsis for in-place update (handles 0D scalars)
            # Ensure g_val matches target shape to avoid (1,) vs () broadcast errors
            if g_val.shape != self.shape and g_val.size == self.total_size:
                g_val = g_val.reshape(self.shape)
                
            self.grad.arr.reshape(self.shape)[...] += g_val
            
    def backward(self, grad=None):
        if grad is None:
            if self.total_size != 1:
                raise RuntimeError("backward() can only be called on scalar outputs or with explicit grad.")
            grad = Tensor([1.0], shape=self.shape)
        elif not isinstance(grad, Tensor):
            grad = Tensor(grad, shape=self.shape)
        
        # Initial accumulation into the root
        self._acc_grad(grad)

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

    def get_samples(self, indices):
        """OOM-Safe retrieval of specific elements (for testing/parity)."""
        if self.device in ['cpu', 'ssd']:
            return self.arr[indices]
        # For Vulkan, we still have to to_numpy() for now, but we'll flag it
        return self.to_numpy().flatten()[indices]

    def to_numpy(self):
        """Unified data retrieval with robust shape handling."""
        def safe_reshape(arr, target_shape):
            try:
                return arr.reshape(target_shape if target_shape else (int(np.prod(arr.shape)),))
            except:
                return arr.flatten()

        if self.device == 'vulkan':
            if hasattr(self, 'np_arr') and self.np_arr is not None:
                return safe_reshape(self.np_arr, self.shape)
            ti.sync()
            return safe_reshape(self.arr.to_numpy(), self.shape)

        # CPU/SSD Mode: self.arr is already a numpy-like array
        if self.dtype == 'int4':
            # Unpack bytes to f32
            n = self.total_size
            packed = self.arr
            unpacked = np.zeros(n, dtype=np.float32)
            unpacked[0::2] = (packed & 0x0F).astype(np.float32) - 8.0
            unpacked[1::2] = ((packed >> 4) & 0x0F).astype(np.float32) - 8.0
            return safe_reshape(unpacked, self.shape)
            
        return safe_reshape(self.arr, self.shape)

    def numpy(self):
        """PyTorch parity alias."""
        return self.to_numpy()
    def item(self):
        """Returns the value of this tensor as a standard Python number."""
        if self.total_size != 1:
            raise ValueError("item() can only be called on tensors with 1 element.")
        return self.to_numpy().item()

    def load_from_numpy(self, np_arr):
        if self.device == 'vulkan':
            self.arr.from_numpy(np_arr.flatten())
            ti.sync()
            _ = self.arr.to_numpy() # Force sync
        else:
            self.arr = np_arr.astype(self.dtype).flatten()
            if hasattr(self, 'np_arr'): self.np_arr = self.arr

    @property
    def total_size(self):
        sz = 1
        for s in self.shape: sz *= s
        return sz

    @property
    def item_size(self):
        if self.dtype == 'int4': return 0.5
        return np.dtype(self.dtype).itemsize

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
    
    @classmethod
    def should_tile(cls, size_bytes):
        """Returns True if the operation should be tiled even on CPU."""
        budget = MemoryManager.get_safe_budget()
        # High-performance threshold: 40% of safe budget (e.g. 6.4GB on 16GB budget)
        # Allows most 'RAM-resident' operations to skip tiling overhead
        return size_bytes > (budget * 0.4)

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
        return Tensor(np_e.copy(), device=self.device)

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
                if self.device == 'vulkan':
                    K.k_add(self.grad.arr, res.grad.arr, self.total_size, res.total_size)
                    ti.sync()
                else:
                    self.grad.arr += res.grad.arr # Simplified for reshape backward
        res._backward_fn = _backward
        return res

    def relu(self):
        from . import functional as F
        return F.relu(self)
            
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                if self.device == 'cpu':
                    mask = self.to_numpy() > 0
                    self._acc_grad(res.grad * mask)
                else:
                    from . import streaming_ops as SOE
                    # Use SOE for masked gradient to avoid heavy Autograd chain
                    mask = (self > 0)
                    self._acc_grad(res.grad * mask)
        res._backward_fn = _backward
        return res

    def leaky_relu(self, alpha=0.01):
        from . import functional as F
        return F.leaky_relu(self, alpha)

    def silu(self):
        from . import functional as F
        return F.silu(self)

    def gelu_tanh(self):
        from . import functional as F
        return F.gelu_tanh(self)

    def softmax(self, dim=-1):
        from . import functional as F
        return F.softmax(self, dim)

    def matmul(self, other):
        """Matrix multiplication with ARAS/Vulkan support."""
        M, K_dim = self.shape
        K2, N = other.shape
        if self.device == 'ssd' or other.device == 'ssd':
            from . import streaming_ops as SOE
            res = SOE.SOE.matmul(self, other)
        else:
            # KAGGLE MODE REDIRECT
            from .config import get_kaggle_enabled, get_kaggle_threshold
            if get_kaggle_enabled() and (self.total_size * self.item_size + other.total_size * other.item_size > get_kaggle_threshold()):
                from .kaggle_executor import KaggleExecutor
                executor = KaggleExecutor()
                return executor.submit_operation("matmul", self, other)

            if self.device == 'cpu' and other.device == 'cpu' and (not MemoryManager.should_tile(self.total_size * self.item_size)):
                a_t = torch.from_numpy(self.to_numpy())
                b_t = torch.from_numpy(other.to_numpy())
                res_t = a_t @ b_t
                res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=(M, N))
            else:
                res = Tensor(None, shape=(M, N), device=self.device)
                K.k_matmul(self.arr, other.arr, res.arr, M, N, K_dim)
                if self.device == 'vulkan': ti.sync()
            
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        
        def _backward():
            if self.requires_grad:
                self._acc_grad(res.grad.matmul(other.transpose(0, 1)))
            if other.requires_grad:
                other._acc_grad(self.transpose(0, 1).matmul(res.grad))
                
        res._backward_fn = _backward
        return res

    def __add__(self, other):
        if not isinstance(other, Tensor): 
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
            
        from .memory import MemoryManager
        size_bytes = self.total_size * self.item_size
        
        # KAGGLE MODE REDIRECT
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (size_bytes > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            return executor.submit_operation('add', self, other)

        if self.device in ['ssd', 'vulkan', 'hybrid', 'kaggle'] or (isinstance(other, Tensor) and other.device in ['ssd', 'vulkan', 'hybrid', 'kaggle']) or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, other, 'add')
        else:
            # Fast in-RAM path for small CPU tensors
            a_t = torch.from_numpy(self.to_numpy())
            b_t = torch.from_numpy(other.to_numpy())
            res_t = a_t + b_t
            res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=self.shape)
            
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad: self._acc_grad(res.grad)
            if other.requires_grad: other._acc_grad(res.grad)
        res._backward_fn = _backward
        return res

    def __sub__(self, other):
        if not isinstance(other, Tensor): 
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
            
        from .memory import MemoryManager
        size_bytes = self.total_size * self.item_size
        
        # KAGGLE MODE REDIRECT
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (size_bytes > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            return executor.submit_operation('sub', self, other)

        if self.device in ['ssd', 'vulkan', 'hybrid', 'kaggle'] or (isinstance(other, Tensor) and other.device in ['ssd', 'vulkan', 'hybrid', 'kaggle']) or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, other, 'sub')
        else:
            # Fast in-RAM path for small CPU tensors
            a_t = torch.from_numpy(self.to_numpy())
            b_t = torch.from_numpy(other.to_numpy())
            res_t = a_t - b_t
            res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=self.shape)
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad: self._acc_grad(res.grad)
            if other.requires_grad: other._acc_grad(res.grad * -1.0)
        res._backward_fn = _backward
        return res

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
            
        from .memory import MemoryManager
        size_bytes = self.total_size * self.item_size
        
        # KAGGLE MODE REDIRECT
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (size_bytes > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            return executor.submit_operation('mul', self, other)

        if self.device in ['ssd', 'vulkan', 'hybrid', 'kaggle'] or (isinstance(other, Tensor) and other.device in ['ssd', 'vulkan', 'hybrid', 'kaggle']) or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, other, 'mul')
        else:
            # Fast in-RAM path for small CPU tensors
            a_t = torch.from_numpy(self.to_numpy())
            b_t = torch.from_numpy(other.to_numpy())
            res_t = a_t * b_t
            res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=self.shape)
            
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad: self._acc_grad(res.grad * other)
            if other.requires_grad: other._acc_grad(res.grad * self)
        res._backward_fn = _backward
        return res

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
            
        from .memory import MemoryManager
        size_bytes = self.total_size * self.item_size
        
        # KAGGLE MODE REDIRECT
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (size_bytes > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            return executor.submit_operation('div', self, other)

        if self.device in ['ssd', 'vulkan', 'hybrid', 'kaggle'] or (isinstance(other, Tensor) and other.device in ['ssd', 'vulkan', 'hybrid', 'kaggle']) or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, other, 'div')
        else:
            # Fast in-RAM path for small CPU tensors
            a_t = torch.from_numpy(self.to_numpy())
            b_t = torch.from_numpy(other.to_numpy())
            res_t = a_t / b_t
            res = Tensor(res_t.numpy(), device='cpu', dtype=np.float32, shape=self.shape)
            
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad: self._acc_grad(res.grad / other)
            if other.requires_grad: other._acc_grad(res.grad * self * -1.0 / (other * other))
        res._backward_fn = _backward
        return res

    def __matmul__(self, other): return self.matmul(other)
    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)
    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
        return other - self
    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
        return other / self
    def __pow__(self, other): return self.pow(other)
    def __rpow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
        return other.pow(self)

    def pow(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=self.dtype), shape=(), device=self.device)
            
        size_bytes = self.total_size * self.item_size
        if self.device == 'cpu' and other.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            res_t = torch.pow(torch.from_numpy(self.to_numpy()), torch.from_numpy(other.to_numpy()))
            res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=self.shape)
        elif self.device == 'ssd' or other.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, other, 'pow')
        else:
            res = Tensor(None, shape=self.shape, device=self.device)
            K.k_pow(self.arr, other.arr, res.arr, self.total_size)
            if self.device == 'vulkan': ti.sync()
            
        res._prev = {self, other}
        res.requires_grad = self.requires_grad or other.requires_grad
        def _backward():
            if self.requires_grad:
                self._acc_grad(res.grad * other * (self ** (other - 1.0)))
            if other.requires_grad:
                self._acc_grad(res.grad * res * self.log())
        res._backward_fn = _backward
        return res

    def _comp_op(self, other, op_type):
        if not isinstance(other, Tensor):
            other = Tensor(np.array([other], dtype=np.float32), device=self.device)
            
        size_bytes = self.total_size * self.item_size
        if (self.device == 'ssd' or other.device == 'ssd') or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, other, op_type)
        else:
            res = Tensor(None, shape=self.shape, device=self.device)
            kernel = getattr(K, f"k_{op_type}")
            kernel(self.arr, other.arr, res.arr, self.total_size, other.total_size)
            if self.device == 'vulkan': ti.sync()
            
        res._prev = {self, other}
        return res

    def __gt__(self, other): return self._comp_op(other, 'gt')
    def __lt__(self, other): return self._comp_op(other, 'lt')
    def __ge__(self, other): return self._comp_op(other, 'ge')
    def __le__(self, other): return self._comp_op(other, 'le')
    def __eq__(self, other): return self._comp_op(other, 'eq')
    def __ne__(self, other): return self._comp_op(other, 'ne')

    def mean(self, dim=None, keepdim=False):
        size_bytes = self.total_size * self.item_size
        
        # 1. Dispatching
        if dim is not None:
            if (dim == -1 or dim == len(self.shape) - 1) and self.device == 'vulkan':
                N = self.shape[-1]
                M = self.total_size // N
                res = Tensor(None, shape=self.shape[:-1] + (1,) if keepdim else self.shape[:-1], device=self.device)
                K.k_mean_last_dim(self.arr, res.arr, M, N)
                ti.sync()
            else:
                a_t = torch.from_numpy(self.to_numpy().reshape(self.shape))
                res_t = a_t.mean(dim=dim, keepdim=keepdim)
                res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=res_t.shape)
        elif self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            s = float(torch.mean(torch.from_numpy(self.to_numpy())).item())
            res = Tensor([s], shape=(), device='cpu')
        elif self.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.mean(self)
        else:
            # KAGGLE MODE REDIRECT
            from .config import get_kaggle_enabled, get_kaggle_threshold
            if get_kaggle_enabled() and (size_bytes > get_kaggle_threshold()):
                from .kaggle_executor import KaggleExecutor
                executor = KaggleExecutor()
                return executor.submit_operation('mean', self)
            res = Tensor(float(np.mean(self.to_numpy())), shape=(), device=self.device)
            
        # 2. Autograd Attachment
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                if dim is None:
                    self._acc_grad(res.grad.expand(*self.shape) / float(self.total_size))
                else:
                    self._acc_grad(res.grad.expand(*self.shape) / float(self.shape[dim]))
        res._backward_fn = _backward
        return res

    def sqrt(self):
        size_bytes = self.total_size * self.item_size
        if self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            res_t = torch.sqrt(torch.from_numpy(self.to_numpy()))
            res = Tensor(res_t.numpy(), device='cpu', dtype=np.float32, shape=self.shape)
        elif self.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, None, 'sqrt')
        else:
            res = Tensor(None, shape=self.shape, device=self.device)
            K.k_sqrt(self.arr, res.arr, self.total_size)
            if self.device == 'vulkan': ti.sync()
            
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self._acc_grad(res.grad * 0.5 / (res + 1e-10))
        res._backward_fn = _backward
        return res

    def exp(self):
        size_bytes = self.total_size * self.item_size
        if self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            res_t = torch.exp(torch.from_numpy(self.to_numpy()))
            res = Tensor(res_t.numpy(), device='cpu', dtype=np.float32, shape=self.shape)
        elif self.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, None, 'exp')
        else:
            res = Tensor(None, shape=self.shape, device=self.device)
            K.k_exp(self.arr, res.arr, self.total_size)
            if self.device == 'vulkan': ti.sync()
            
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self._acc_grad(res.grad * res)
        res._backward_fn = _backward
        return res

    def log(self):
        size_bytes = self.total_size * self.item_size
        if self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            res_t = torch.log(torch.from_numpy(self.to_numpy()))
            res = Tensor(res_t.numpy(), device='cpu', dtype=np.float32, shape=self.shape)
        elif self.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, None, 'log')
        else:
            res = Tensor(None, shape=self.shape, device=self.device)
            K.k_log(self.arr, res.arr, self.total_size)
            if self.device == 'vulkan': ti.sync()
            
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self._acc_grad(res.grad / (self + 1e-10))
        res._backward_fn = _backward
        return res

    def tanh(self):
        size_bytes = self.total_size * self.item_size
        if self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            res_t = torch.tanh(torch.from_numpy(self.to_numpy()))
            res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=self.shape)
        elif self.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, None, 'tanh')
        else:
            res = Tensor(None, shape=self.shape, device=self.device)
            K.k_tanh(self.arr, res.arr, self.total_size)
            if self.device == 'vulkan': ti.sync()
            
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self._acc_grad(res.grad * (1.0 - res * res))
        res._backward_fn = _backward
        return res

    def sigmoid(self):
        size_bytes = self.total_size * self.item_size
        if self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            res_t = torch.sigmoid(torch.from_numpy(self.to_numpy()))
            res = Tensor(res_t.numpy(), device='cpu', dtype=self.dtype, shape=self.shape)
        elif self.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, None, 'sigmoid')
        else:
            res = Tensor(None, shape=self.shape, device=self.device)
            K.k_copy(self.arr, res.arr, self.total_size)
            K.k_sigmoid_1d(res.arr, self.total_size)
            if self.device == 'vulkan': ti.sync()
            
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                from . import streaming_ops as SOE
                # Unified sigmoid backward kernel
                grad_val = SOE.SOE.elementwise_op(self, None, 'sigmoid_backward', extra=res.grad)
                self._acc_grad(grad_val)
        res._backward_fn = _backward
        return res

    def masked_fill(self, mask, value):
        if self.device == 'ssd':
            from . import streaming_ops as SOE
            res = SOE.SOE.elementwise_op(self, mask, 'masked_fill', extra=value)
        else:
            res_np = self.to_numpy()
            m = mask.to_numpy() if isinstance(mask, Tensor) else mask
            res_np[m > 0.5] = value
            res = Tensor(res_np, device=self.device)
            
        res._prev = {self, mask} if isinstance(mask, Tensor) else {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                m = mask.to_numpy() if isinstance(mask, Tensor) else mask
                float_mask = Tensor((m <= 0.5).astype(np.float32), device=res.device)
                self._acc_grad(res.grad * float_mask)
        res._backward_fn = _backward
        return res

    def gather(self, dim, index):
        if self.device == 'ssd':
            idx_np = index.to_numpy()
            res_np = np.take_along_axis(self.arr.reshape(self.shape), idx_np, axis=dim)
            return Tensor(res_np, device='ssd' if self.total_size > 1e8 else 'auto')
        return Tensor(np.take_along_axis(self.to_numpy(), index.to_numpy(), axis=dim), device=self.device)

    def t(self):
        if len(self.shape) != 2: return self
        size_bytes = self.total_size * self.item_size
        if self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            res_t = torch.from_numpy(self.to_numpy().reshape(self.shape)).t()
            return Tensor(res_t.numpy().copy(), device='cpu', dtype=self.dtype, shape=res_t.shape)
        if size_bytes > 1e8:
             return Tensor(self.to_numpy().T, device='auto')
        H, W = self.shape
        res = Tensor(None, shape=(W, H), device=self.device)
        K.k_transpose_2d(self.arr, res.arr, H, W)
        if self.device == 'vulkan': ti.sync()
        return res

    def sum(self):
        size_bytes = self.total_size * self.item_size
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (size_bytes > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            return executor.submit_operation('sum', self)

        if self.device == 'cpu' and not MemoryManager.should_tile(size_bytes):
            torch = _get_torch()
            s = float(torch.sum(torch.from_numpy(self.arr.reshape(self.shape)).to(torch.float64)).item())
            res = Tensor([s], shape=(), device='cpu')
        elif self.device == 'ssd' or MemoryManager.should_tile(size_bytes):
            from . import streaming_ops as SOE
            res = SOE.SOE.sum(self)
        else:
            res = Tensor(float(np.sum(self.to_numpy())), shape=(), device=self.device)
            
        res._prev = {self}
        res.requires_grad = self.requires_grad
        def _backward():
            if self.requires_grad:
                self._acc_grad(res.grad.expand(*self.shape))
        res._backward_fn = _backward
        return res

    def fused_sum(self, other=None, op='mul'):
        if self.device != 'ssd':
            if not MemoryManager.should_tile(self.total_size * self.item_size):
                if other is None: return self.sum()
                if op == 'mul': return (self * other).sum()
                if op == 'add': return (self + other).sum()
                if op == 'sub': return (self - other).sum()
                if op == 'div': return (self / other).sum()
                return self.sum()
        from . import streaming_ops as SOE
        return SOE.SOE.elementwise_reduce(self, other, op, 'sum')

    def unsqueeze(self, dim):
        new_shape = list(self.shape)
        if dim < 0: dim += len(new_shape) + 1
        new_shape.insert(dim, 1)
        return self.reshape(*new_shape)
