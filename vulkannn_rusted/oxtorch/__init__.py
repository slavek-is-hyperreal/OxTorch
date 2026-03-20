import sys
import vulkannn_rusted as vnn
from .tensor import Tensor

# Re-export DataType and native Tensor for low-level access if needed
from vulkannn_rusted import DataType

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def __getattr__(name):
    # Normalize case for common ops (GELU -> gelu, ReLU -> relu)
    if name in ["GELU", "ReLU", "Softmax"]:
        name = name.lower()
    """
    Module-level dynamic dispatcher.
    Allows 'oxtorch.randn' or 'oxtorch.zeros' to either use native VNN implementations
    or fallback to original PyTorch.
    """
    # 1. Native factories & Global Ops (proxied from vnn.Tensor)
    if hasattr(vnn.Tensor, name):
        attr = getattr(vnn.Tensor, name)
        if callable(attr):
            def global_wrapper(*args, **kwargs):
                # Check if it's a factory (zeros, ones, rand)
                if name in ['zeros', 'ones', 'rand']:
                    # Standardize shape arg
                    if len(args) > 0 and isinstance(args[0], (list, tuple)):
                        shape = list(args[0])
                    elif len(args) > 0 and isinstance(args[0], int):
                        shape = [arg for arg in args if isinstance(arg, int)]
                    else:
                        shape = args[0] if len(args) > 0 else kwargs.get('shape', [])
                    
                    dtype = kwargs.get('dtype', DataType.F32)
                    device = kwargs.get('device', 'cpu')
                    result = attr(shape, dtype, device)
                else:
                    # It might be an instance method called as a global (e.g. torch.relu(tensor))
                    if len(args) > 0 and isinstance(args[0], Tensor):
                        # Call as instance method: args[0].relu(*args[1:], **kwargs)
                        instance = args[0]
                        method = getattr(instance, name)
                        result = method(*args[1:], **kwargs)
                    else:
                        # Direct static call
                        result = attr(*args, **kwargs)

                if isinstance(result, vnn.Tensor):
                    return Tensor(result)
                return result
            return global_wrapper

    # 2. PyTorch Globals Fallback (e.g., torch.exp, torch.randn, etc.)
    if HAS_TORCH:
        if hasattr(torch, name):
            pt_attr = getattr(torch, name)
            if callable(pt_attr):
                def pt_global_wrapper(*args, **kwargs):
                    # Convert inputs to PT
                    pt_args = [a.to_torch() if isinstance(a, Tensor) else a for a in args]
                    pt_kwargs = {k: (v.to_torch() if isinstance(v, Tensor) else v) for k, v in kwargs.items()}
                    result = pt_attr(*pt_args, **pt_kwargs)
                    if isinstance(result, torch.Tensor):
                        return Tensor.from_torch(result)
                    elif isinstance(result, (list, tuple)):
                        return type(result)(Tensor.from_torch(r) if isinstance(r, torch.Tensor) else r for r in result)
                    return result
                return pt_global_wrapper
            return pt_attr

    raise AttributeError(f"module 'oxtorch' has no attribute '{name}'")

# Explicit factories
def tensor(data, *args, **kwargs):
    return Tensor(data=data, *args, **kwargs)

def as_tensor(data, *args, **kwargs):
    if isinstance(data, Tensor):
        return data
    return Tensor(data=data, *args, **kwargs)

def from_numpy(data):
    return Tensor(data=data)

def bmm(input, mat2):
    return input.bmm(mat2)

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    return input.layer_norm(normalized_shape, weight, bias, eps)

def rms_norm(input, normalized_shape, weight=None, eps=1e-5):
    return input.rms_norm(normalized_shape, weight, eps)

def zeros(*args, **kwargs):
    # Try native vnn.Tensor.zeros first
    try:
        # VNN zeros takes (shape, dtype, device)
        # torch.zeros (*size, out=None, dtype=None, device=None, layout=None, requires_grad=False)
        # This is complex to map perfectly without a sig-matcher, but let's do common cases.
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            shape = list(args[0])
        else:
            shape = list(args)
        
        dtype = kwargs.get('dtype', DataType.F32)
        device = kwargs.get('device', 'cpu')
        return Tensor(vnn.Tensor.zeros(shape, dtype, device))
    except:
        if HAS_TORCH:
            return Tensor.from_torch(torch.zeros(*args, **kwargs))
        raise

def randn(*args, **kwargs):
    if HAS_TORCH:
        return Tensor.from_torch(torch.randn(*args, **kwargs))
    # No native RandN in VNN yet (we have Rand Uniform)
    raise NotImplementedError("randn requires PyTorch fallback.")

# DType aliases for PyTorch compatibility
f32 = DataType.F32
float32 = DataType.F32
f16 = DataType.F16
float16 = DataType.F16
bf16 = DataType.BF16
bfloat16 = DataType.BF16
int8 = DataType.Int8

# Branding
__version__ = "3.7.0"
__name__ = "oxtorch"
