import numpy as np
import vulkannn_rusted as vnn

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class Tensor:
    """
    OxTorch Tensor Proxy.
    Prioritizes native high-performance optimized kernels from 'vulkannn_rusted'.
    Automatically falls back to original PyTorch for any unimplemented operations.
    """
    def __init__(self, data=None, shape=None, dtype=vnn.DataType.F32, device="cpu", name="tensor"):
        if isinstance(dtype, str):
            dtype_map = {
                "f32": vnn.DataType.F32,
                "f16": vnn.DataType.F16,
                "bf16": vnn.DataType.BF16,
                "int8": vnn.DataType.Int8,
                "bitnet2": vnn.DataType.BitNet2,
                "bitnet1.6": vnn.DataType.BitNet1_6,
            }
            dtype = dtype_map.get(dtype.lower(), vnn.DataType.F32)

        if isinstance(data, vnn.Tensor):
            self._vnn = data
        elif isinstance(data, (np.ndarray, list)):
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            self._vnn = vnn.Tensor(data=data, shape=shape, dtype=dtype, device=device, name=name)
        elif shape is not None:
            self._vnn = vnn.Tensor(shape=shape, dtype=dtype, device=device, name=name)
        else:
            raise ValueError("OxTorch Tensor requires data or shape.")

    @property
    def shape(self):
        return self._vnn.shape

    @property
    def device(self):
        return self._vnn.device

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Tensor(self._vnn.reshape(list(shape)))

    def unsqueeze(self, dim):
        return Tensor(self._vnn.unsqueeze(dim))

    def squeeze(self, dim=None):
        return Tensor(self._vnn.squeeze(dim))

    def cat(self, others, dim=0):
        if not isinstance(others, (list, tuple)):
            others = [others]
        vnn_tensors = [self._vnn] + [t._vnn if isinstance(t, Tensor) else t for t in others]
        return Tensor(vnn.Tensor.cat(vnn_tensors, dim))
    
    def stack(self, others, dim=0):
        if not isinstance(others, (list, tuple)):
            others = [others]
        vnn_tensors = [self._vnn] + [t._vnn if isinstance(t, Tensor) else t for t in others]
        return Tensor(vnn.Tensor.stack(vnn_tensors, dim))
    def split(self, split_size, dim=0):
        vnn_results = self._vnn.split(split_size, dim)
        return [Tensor(v) for v in vnn_results]

    def chunk(self, chunks, dim=0):
        vnn_results = self._vnn.chunk(chunks, dim)
        return [Tensor(v) for v in vnn_results]

    def relu(self):
        return Tensor(self._vnn.relu())

    @property
    def dtype(self):
        return self._vnn.dtype

    def to_numpy(self):
        return self._vnn.to_numpy()

    def to_torch(self):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Fallback unavailable.")
        return torch.from_numpy(self.to_numpy())

    @staticmethod
    def from_torch(torch_tensor):
        np_data = torch_tensor.detach().cpu().numpy()
        return Tensor(data=np_data)

    @staticmethod
    def from_ssd(path, shape, dtype=vnn.DataType.F32):
        return Tensor(vnn.Tensor.from_ssd(path, shape, dtype))

    @staticmethod
    def new_ssd(path, shape, dtype=vnn.DataType.F32):
        return Tensor(vnn.Tensor.new_ssd(path, shape, dtype))

    def __repr__(self):
        return f"OxTorch.Tensor({self.shape}, dtype={self.dtype}, device={self.device})"

    def to(self, *args, **kwargs):
        # If it's a device change that VNN supports (cpu/vulkan/hybrid)
        if len(args) > 0 and isinstance(args[0], str):
            device = args[0]
            if device in ["cpu", "vulkan", "hybrid", "ssd", "vga"]:
                return Tensor(self._vnn.to(device))
        
        # Fallback to PyTorch for anything else (dtype changes, etc.)
        pt_res = self.to_torch().to(*args, **kwargs)
        if isinstance(pt_res, torch.Tensor):
            return Tensor.from_torch(pt_res)
        return pt_res

    def to_bitnet(self, dtype):
        if isinstance(dtype, str):
            dtype_map = {
                "bitnet2": vnn.DataType.BitNet2,
                "bitnet1.6": vnn.DataType.BitNet1_6,
            }
            dtype = dtype_map.get(dtype.lower(), vnn.DataType.BitNet2)
        return Tensor(self._vnn.to_bitnet(dtype))

    def cpu(self):
        return self # We are already on CPU or mapped

    def detach(self):
        return self # No autograd yet

    def numpy(self):
        return self.to_numpy()

    def msts_pytorch_apply(self, func):
        """
        Executes a PyTorch function tile-by-tile on an SSD tensor.
        Prevents OOM for massive tensors by streaming through a 1MB ring buffer.
        """
        if self.device != "ssd":
            # For non-SSD tensors, just use regular torch fallback
            return Tensor.from_torch(func(self.to_torch()))
            
        def tile_callback(np_tile):
            with torch.no_grad():
                tt = torch.from_numpy(np_tile)
                # Handle F16/BF16 views
                if self.dtype == vnn.DataType.F16:
                    tt = tt.view(torch.float16)
                elif self.dtype == vnn.DataType.BF16:
                    tt = tt.view(torch.bfloat16)
                
                # Execute the PyTorch function
                res = func(tt)
                
                # Ensure same dtype and device
                if res.dtype != tt.dtype:
                    res = res.to(tt.dtype)
                
                out_np = res.numpy().flatten()
                
                # Convert back to uint16 representation for F16/BF16
                if self.dtype in [vnn.DataType.F16, vnn.DataType.BF16]:
                    out_np = out_np.view(np.uint16)
                    
                return out_np

        # Call the Rust engine with our callback
        return Tensor(self._vnn.msts_pytorch_apply(tile_callback))

    def item(self):
        np_data = self.to_numpy()
        return np_data.flatten()[0] if np_data.size == 1 else np_data.item()

    def __getattr__(self, name):
        name_lower = name.lower()
        # 1. Check if it's a native method in vulkannn_rusted
        if hasattr(self._vnn, name_lower):
            attr = getattr(self._vnn, name_lower)
            if callable(attr):
                def wrapper(*args, **kwargs):
                    # Convert arguments back to native VNN tensors if they are proxies
                    vnn_args = [a._vnn if isinstance(a, Tensor) else a for a in args]
                    result = attr(*vnn_args, **kwargs)
                    if isinstance(result, vnn.Tensor):
                        return Tensor(result)
                    return result
                return wrapper
            return attr

        # 2. SSD Fallback: Auto-apply PyTorch for missing ops on SSD device (TILED)
        if self.device == "ssd" and HAS_TORCH:
            import torch.nn.functional as F
            # Check F then torch
            op = getattr(F, name_lower, None) or getattr(torch, name_lower, None)
            if op and callable(op):
                return lambda *args, **kwargs: self.msts_pytorch_apply(lambda x: op(x, *args, **kwargs))

        # 3. Standard Fallback to PyTorch (PULLS TO RAM!)
        if HAS_TORCH:
            if self.device == "ssd":
                # PROTECTION: Do not pull massive SSD tensors to RAM silently!
                raise MemoryError(
                    f"OxTorch: Operation '{name}' is not supported natively on SSD and tiling fallback failed. "
                    f"Refusing to pull SSD tensor ({self.shape}) into RAM to prevent OOM. "
                    f"Call '.to_ram()' explicitly if this is intended."
                )
            
            pt_tensor = self.to_torch()
            if hasattr(pt_tensor, name_lower) or hasattr(pt_tensor, name):
                target_name = name if hasattr(pt_tensor, name) else name_lower
                pt_attr = getattr(pt_tensor, target_name)
                if callable(pt_attr):
                    def fallback_wrapper(*args, **kwargs):
                        # Convert all proxy arguments to Torch tensors for the call
                        pt_args = []
                        for a in args:
                            if isinstance(a, Tensor):
                                pt_args.append(a.to_torch())
                            else:
                                pt_args.append(a)
                        
                        result = pt_attr(*pt_args, **kwargs)
                        
                        # Convert result back to OxTorch
                        if isinstance(result, torch.Tensor):
                            return Tensor.from_torch(result)
                        elif isinstance(result, (list, tuple)):
                            return type(result)(Tensor.from_torch(r) if isinstance(r, torch.Tensor) else r for r in result)
                        return result
                    return fallback_wrapper
                return pt_attr

        raise AttributeError(f"'OxTorch.Tensor' object has no attribute '{name}'")

    # Magic methods for arithmetic
    def __add__(self, other):
        other_vnn = other._vnn if isinstance(other, Tensor) else other
        return Tensor(self._vnn.__add__(other_vnn))

    def __sub__(self, other):
        other_vnn = other._vnn if isinstance(other, Tensor) else other
        return Tensor(self._vnn.__sub__(other_vnn))

    def __mul__(self, other):
        other_vnn = other._vnn if isinstance(other, Tensor) else other
        return Tensor(self._vnn.__mul__(other_vnn))

    def __rmul__(self, other):
        return Tensor(self._vnn.__rmul__(other))

    def __truediv__(self, other):
        other_vnn = other._vnn if isinstance(other, Tensor) else other
        return Tensor(self._vnn.__truediv__(other_vnn))

    def __rtruediv__(self, other):
         # x / tensor -> fallback to PT for now as we don't have scalar / tensor native yet
         return self.from_torch(other / self.to_torch())

    def __radd__(self, other):
        return Tensor(self._vnn.__radd__(other))

    def __rsub__(self, other):
        # x - tensor
        return self.from_torch(other - self.to_torch())

    def __matmul__(self, other):
        other_vnn = other._vnn if isinstance(other, Tensor) else other
        return Tensor(self._vnn.__matmul__(other_vnn))

    def bmm(self, other):
        other_vnn = other._vnn if isinstance(other, Tensor) else other
        return Tensor(self._vnn.bmm(other_vnn))

    def layer_norm(self, normalized_shape, weight=None, bias=None, eps=1e-5):
        w_vnn = weight._vnn if weight is not None else None
        b_vnn = bias._vnn if bias is not None else None
        return Tensor(self._vnn.layer_norm(list(normalized_shape), w_vnn, b_vnn, eps))

    def rms_norm(self, normalized_shape, weight=None, eps=1e-5):
        w_vnn = weight._vnn if weight is not None else None
        return Tensor(self._vnn.rms_norm(list(normalized_shape), w_vnn, eps))

    def index_select(self, dim, index):
        idx_tensor = Tensor(index) if not isinstance(index, Tensor) else index
        return Tensor(self._vnn.index_select(dim, idx_tensor._vnn))

    def __getitem__(self, key):
        # Indexing is complex; fallback to PyTorch
        pt_res = self.to_torch().__getitem__(key)
        if isinstance(pt_res, torch.Tensor):
            return Tensor.from_torch(pt_res)
        return pt_res

    def __setitem__(self, key, value):
        # Advanced assignment; fallback to PyTorch
        pt_tensor = self.to_torch()
        val = value.to_torch() if isinstance(value, Tensor) else value
        pt_tensor.__setitem__(key, val)
        # Update our underlying storage
        self._vnn = Tensor.from_torch(pt_tensor)._vnn

    def bit_linear(self, weight, scale, bias=None):
        w_vnn = weight._vnn if isinstance(weight, Tensor) else weight
        s_vnn = scale._vnn if isinstance(scale, Tensor) else scale
        b_vnn = bias._vnn if (bias is not None and isinstance(bias, Tensor)) else bias
        return Tensor(self._vnn.bit_linear(w_vnn, s_vnn, b_vnn))
