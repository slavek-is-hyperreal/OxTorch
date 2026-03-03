import numpy as np
from .tensor import Tensor
from . import optimizers as optim
from . import functional as F
from .modules import base, layers, tiled
from concurrent.futures import ThreadPoolExecutor
import time

# Dtype Aliases
float32 = np.float32
float16 = np.float16
int32 = np.int32
int16 = np.int16
int8 = np.int8
long = np.int64
int = np.int32
double = np.float64
int32 = np.int32
int16 = np.int16
int4 = 'int4'
# Device Shim
class cuda:
    @staticmethod
    def is_available(): return True # We use Vulkan, but torch scripts ask this
    @staticmethod
    def device_count(): return 1

# nn subpackage
class NN:
    Module = base.Module
    ModuleList = base.ModuleList
    Sequential = base.Sequential
    Linear = layers.Linear
    ReLU = layers.ReLU
    SiLU = layers.SiLU
    RMSNorm = layers.RMSNorm
    Softmax = layers.Softmax
    Embedding = layers.Embedding
    # Tiled variants
    TiledLinear = tiled.TiledLinear
    TiledEmbedding = tiled.TiledEmbedding

nn = NN()

# Factory Functions
def tensor(data, dtype=None, device='auto', requires_grad=False):
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def from_numpy(np_array):
    return Tensor(np_array)

# Re-export Tensor functions

def _should_stream(shape, dtype, requested_device):
    if requested_device == 'ssd': return True
    if requested_device == 'auto':
        n = 1
        for s in shape: n *= s
        item_size = np.dtype(dtype if dtype else np.float32).itemsize
        size_bytes = n * item_size
        
        from .memory import MemoryManager
        return MemoryManager.should_offload_to_ssd(size_bytes)
    return False

def zeros(*size, dtype=None, device='auto', requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (list, tuple)): size = size[0]
    if _should_stream(size, dtype, device):
        t = Tensor(None, shape=size, dtype=dtype, device='ssd', requires_grad=requires_grad)
        # Tensor(None) is already zeroed by TensorStore.zeros()
        return t
    return Tensor(np.zeros(size, dtype=np.float32 if dtype is None else dtype), 
                  dtype=dtype, device=device, requires_grad=requires_grad)

def ones(*size, dtype=None, device='auto', requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (list, tuple)): size = size[0]
    if _should_stream(size, dtype, device):
        t = Tensor(None, shape=size, dtype=dtype, device='ssd', requires_grad=requires_grad)
        n = t.total_size
        item_size = np.dtype(t.dtype).itemsize
        
        # Conservative Allocation to prevent OS freezing (128MB sequential)
        chunk_size_bytes = 128 * 1024 * 1024
        chunk_len = chunk_size_bytes // item_size
        
        size_str = f"{n*item_size/1e6:.1f}MB" if n*item_size >= 1e6 else f"{n*item_size/1024:.1f}KB"
        print(f"  [Factory] Initializing SSD 'ones' tensor ({size_str}, {t.dtype})...")
        t0 = time.perf_counter()
        
        for start in range(0, n, chunk_len):
            end = min(start + chunk_len, n)
            t.arr[start:end] = np.ones(end - start, dtype=t.dtype)
        
        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return t
    return Tensor(np.ones(size, dtype=np.float32 if dtype is None else dtype), 
                  dtype=dtype, device=device, requires_grad=requires_grad)

def randn(*size, dtype=None, device='auto', requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (list, tuple)): size = size[0]
    if _should_stream(size, dtype, device):
        t = Tensor(None, shape=size, dtype=dtype, device='ssd', requires_grad=requires_grad)
        n = t.total_size
        item_size = np.dtype(t.dtype).itemsize
        
        # Conservative Allocation to prevent OS freezing (128MB sequential)
        chunk_size_bytes = 128 * 1024 * 1024
        chunk_len = chunk_size_bytes // item_size
        
        print(f"  [Factory] SSD 'randn' ({n*item_size/1e6:.1f}MB)...")
        t0 = time.perf_counter()
        
        for start in range(0, n, chunk_len):
            end = min(start + chunk_len, n)
            t.arr[start:end] = np.random.randn(end - start).astype(t.dtype)
        
        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return t
    return Tensor(np.random.randn(*size).astype(np.float32 if dtype is None else dtype), 
                  dtype=dtype, device=device, requires_grad=requires_grad)

def arange(end, dtype=None, device='auto'):
    return Tensor(np.arange(end, dtype=np.float32 if dtype is None else dtype),
                  dtype=dtype, device=device)

def from_binary(path, shape, dtype=np.float32, requires_grad=False):
    """Zero-copy load: Mount an existing binary file as an SSD-native Tensor."""
    return Tensor(None, shape=shape, dtype=dtype, device='ssd', requires_grad=requires_grad, external_path=path)

def cat(tensors, dim=0):
    if len(tensors) == 0: return None
    # Check if any tensor is on SSD
    is_ssd = any([t.device == 'ssd' for t in tensors])
    
    # Calculate output shape
    out_shape = list(tensors[0].shape)
    for t in tensors[1:]:
        out_shape[dim] += t.shape[dim]
    
    if is_ssd or _should_stream(out_shape, tensors[0].dtype, 'auto'):
        # SSD-Native Concatenation
        res = Tensor(None, shape=out_shape, device='ssd', dtype=tensors[0].dtype)
        # We need to copy slices. Since we have __setitem__ now, we can use it!
        # But for SSD it's faster to do it in SOE for better tile management.
        # For now, let's use the greedy __setitem__ logic.
        current_pos = 0
        print(f"  [torch.cat] Concatenating {len(tensors)} tensors to SSD ({np.prod(out_shape)*4/1e6:.1f}MB)...")
        for t in tensors:
            length = t.shape[dim]
            # Create a full-dim slice object
            slc = [slice(None)] * len(out_shape)
            slc[dim] = slice(current_pos, current_pos + length)
            res[tuple(slc)] = t
            current_pos += length
        return res
    else:
        # Standard CPU cat
        np_arrs = [t.to_numpy() for t in tensors]
        return Tensor(np.concatenate(np_arrs, axis=dim))

def stack(tensors, dim=0):
    # Unsqueeze each and cat
    unsqueezed = [t.unsqueeze(dim) for t in tensors]
    return cat(unsqueezed, dim=dim)

def split(tensor, split_size_or_sections, dim=0):
    # Basic implementation using slices
    if isinstance(split_size_or_sections, int):
        sections = range(0, tensor.shape[dim], split_size_or_sections)
        indices = [slice(i, min(i + split_size_or_sections, tensor.shape[dim])) for i in sections]
    else:
        # List of sizes
        curr = 0
        indices = []
        for s in split_size_or_sections:
            indices.append(slice(curr, curr + s))
            curr += s
            
    res = []
    for slc in indices:
        full_slc = [slice(None)] * len(tensor.shape)
        full_slc[dim] = slc
        res.append(tensor[tuple(full_slc)])
    return res

def chunk(tensor, chunks, dim=0):
    size = (tensor.shape[dim] + chunks - 1) // chunks
    return split(tensor, size, dim=dim)

def sum(input):
    return input.sum()

def matmul(input, other):
    return input.matmul(other)

# Global settings
def manual_seed(s):
    np.random.seed(s)

def patch_pytorch():
    """Nuclear option: Replace torch in sys.modules."""
    import sys
    sys.modules['torch'] = sys.modules['vulkan_nn_lib.torch_shim']
    sys.modules['torch.nn'] = sys.modules['vulkan_nn_lib.torch_shim'].nn
    sys.modules['torch.nn.functional'] = sys.modules['vulkan_nn_lib.functional']
    sys.modules['torch.optim'] = sys.modules['vulkan_nn_lib.optimizers']
