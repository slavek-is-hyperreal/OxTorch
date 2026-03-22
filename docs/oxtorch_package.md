# OxTorch Python Package & Fallback Mechanism

The `oxtorch` package is a high-level Python wrapper that ensures OxTorch is 100% compatible with the PyTorch API as a drop-in replacement (`import oxtorch as torch`).

---

## 1. Wrapper Architecture

The system consists of two layers:
1.  **`vulkannn_rusted`**: A binary module compiled in Rust (PyO3). It contains the low-level `Tensor` class and SIMD/Vulkan kernels.
2.  **`oxtorch`**: A pure Python module (`vulkannn_rusted/oxtorch/`). It implements dispatch logic, proxying, and fallback to the original PyTorch.

---

## 2. Dynamic Dispatch (Module-level)

In `oxtorch/__init__.py`, there is a `__getattr__` function that intercepts global calls (e.g., `torch.relu(t)`):

1.  **Native Check**: It checks if `vulkannn_rusted.Tensor` has the given method. If it does, it calls it natively.
2.  **PyTorch Fallback**: If the operation is missing in Rust, OxTorch:
    - Converts the `OxTorchTensor` arguments to `torch.Tensor`.
    - Calls the original function from the installed `torch` package.
    - Wraps the result back into an `OxTorchTensor`.

---

## 3. Proxy Tensor (`oxtorch/tensor.py`)

The `Tensor` class in `oxtorch` is a wrapper storing the native object in the `self._vnn` field. The key mechanism is class-level `__getattr__`:

```python
def __getattr__(self, name):
    # 1. Native?
    if hasattr(self._vnn, name):
        # ... call native kernel ...
    
    # 2. SSD Fallback?
    if self.device == "ssd":
        return self.msts_pytorch_apply(...)
    
    # 3. Standard PyTorch Fallback (PULLS TO RAM!)
```

---

## 4. Intelligent SSD Fallback (`msts_pytorch_apply`)

This is a unique feature of OxTorch. If you execute an `erf()` operation on a 100GB tensor located on an SSD:
1.  OxTorch knows there is no native `erf` kernel for SSD.
2.  Instead of throwing an error or pulling 100GB into RAM (which would cause an OOM), it calls `msts_pytorch_apply`.
3.  The data is streamed in tiles (1MB) through MSTS orchestration.
4.  Each tile is temporarily cast to PyTorch, processed, and flushed back to the SSD.

**Result**: Full PyTorch functionality on massive data with minimal RAM consumption.

---

## 5. Environment Configuration

Since `oxtorch` is delivered as a folder within the repository (or inside a wheel) rather than a standalone `pip` package, paths must be set correctly:

```bash
# Developer (local) version
export PYTHONPATH=$PYTHONPATH:/path/to/vulkannn_rusted
```

In your Python code, you can then simply use:
```python
# The entire change
import oxtorch as torch
x = torch.randn(10, 10, device="vulkan") # Native OxTorch
y = torch.erf(x)                        # Fallback to PyTorch
```
