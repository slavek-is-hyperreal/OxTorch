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
def _should_stream(size, dtype, device):
    if device == 'ssd': return True
    if device == 'auto':
        n = 1
        for s in size: n *= s
        item_size = np.dtype(dtype if dtype else np.float32).itemsize
        # If > 512MB, check if we should go to SSD
        if n * item_size > 512 * 1024 * 1024:
            from .tensor import Tensor
            # We don't want to import Tensor at top level due to circularity if not careful
            # but here it's fine.
            return True
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
        # Parallel chunked fill with 1.0
        n = t.total_size
        chunk = 128 * 1024 * 1024 # 128M elements
        print(f"  [Factory] Parallel initializing SSD 'ones' ({n*np.dtype(t.dtype).itemsize/1e6:.1f}MB)...")
        t0 = time.perf_counter()
        
        def fill_chunk(start, end):
            t.arr[start:end] = 1.0

        with ThreadPoolExecutor(max_workers=4) as executor:
            offsets = range(0, n, chunk)
            for start in offsets:
                end = min(start + chunk, n)
                executor.submit(fill_chunk, start, end)
        
        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return t
    return Tensor(np.ones(size, dtype=np.float32 if dtype is None else dtype), 
                  dtype=dtype, device=device, requires_grad=requires_grad)

def randn(*size, dtype=None, device='auto', requires_grad=False):
    if len(size) == 1 and isinstance(size[0], (list, tuple)): size = size[0]
    if _should_stream(size, dtype, device):
        t = Tensor(None, shape=size, dtype=dtype, device='ssd', requires_grad=requires_grad)
        n = t.total_size
        chunk = 32 * 1024 * 1024
        print(f"  [Factory] Parallel initializing SSD 'randn' ({n*np.dtype(t.dtype).itemsize/1e6:.1f}MB)...")
        t0 = time.perf_counter()
        
        def fill_chunk(start, end):
            t.arr[start:end] = np.random.randn(end-start).astype(t.dtype)

        # Note: np.random is not truly thread-safe for high speed, 
        # but for initialization it's okay or we use one seed per thread.
        with ThreadPoolExecutor(max_workers=4) as executor:
            offsets = range(0, n, chunk)
            for start in offsets:
                end = min(start + chunk, n)
                executor.submit(fill_chunk, start, end)
        
        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return t
    return Tensor(np.random.randn(*size).astype(np.float32 if dtype is None else dtype), 
                  dtype=dtype, device=device, requires_grad=requires_grad)

def arange(end, dtype=None, device='auto'):
    return Tensor(np.arange(end, dtype=np.float32 if dtype is None else dtype),
                  dtype=dtype, device=device)

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
