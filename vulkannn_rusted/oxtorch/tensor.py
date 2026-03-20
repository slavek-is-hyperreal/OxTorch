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
        if isinstance(data, vnn.Tensor):
            self._vnn = data
        elif isinstance(data, (np.ndarray, list)):
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            self._vnn = vnn.Tensor(data=data, dtype=dtype, device=device, name=name)
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
        # If it's a device change that VNN supports (cpu/vulkan)
        if len(args) > 0 and isinstance(args[0], str):
            if args[0] in ["cpu", "vulkan", "hybrid"]:
                # VNN currently handles device via .device property (settable)
                # But it's better to create a copy with the new device if needed.
                # For now, let's just update the proxy if consistent.
                # Actually, native VNN Tensor has no .to() method yet.
                # Let's fallback to PT for complex device/dtype casting.
                pass
        return self.to_torch().to(*args, **kwargs) # Fallback

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
        # 1. Check if it's a native method in vulkannn_rusted
        if hasattr(self._vnn, name):
            attr = getattr(self._vnn, name)
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

        # 2. SSD Fallback: Auto-apply PyTorch for missing ops on SSD device
        if self.device == "ssd" and HAS_TORCH:
            import torch.nn.functional as F
            # Check F then torch
            op = getattr(F, name, None) or getattr(torch, name, None)
            if op and callable(op):
                return lambda *args, **kwargs: self.msts_pytorch_apply(lambda x: op(x, *args, **kwargs))

        # 3. Standard Fallback to PyTorch (PULLS TO RAM!)
        if HAS_TORCH:
            if self.device == "ssd":
                # LOG A WARNING: pulling SSD to RAM
                print(f"WARNING: OxTorch is pulling SSD tensor '{self.name}' ({self.shape}) into RAM for fallback op '{name}'. This may OOM.")
            
            pt_tensor = self.to_torch()
            if hasattr(pt_tensor, name):
                pt_attr = getattr(pt_tensor, name)
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
