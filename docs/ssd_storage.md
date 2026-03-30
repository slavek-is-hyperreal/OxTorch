# OxTorch SSD Storage Guide

OxTorch supports Multi-Source Tensor Streaming (MSTS), allowing you to process datasets that are much larger than your available RAM. Large tensors can be mapped directly to files on NVMe/SSD.

## Binary Storage Format

To ensure maximum performance (via `io_uring` and `O_DIRECT`), OxTorch uses a **strictly raw** binary format:

1.  **No Headers**: The file must contain only the raw numerical data. No metadata, JSON, or magic numbers.
2.  **C-order (Row-Major)**: Data must be stored in contiguous row-major order.
3.  **Alignment**: While MSTS handles offsets internally, for best performance, the files should be stored on systems with at least 512-byte block alignment.
4.  **Data Types**:
    - `float32`: 4-byte IEEE float (Little Endian).
    - `bf16`: 2-byte Brain Floating Point.
    - `int8`: 1-byte signed integer.

## Preparing Data in Python

The easiest way to prepare SSD-compatible files is using **NumPy** or **PyTorch**.

### Using NumPy
```python
import numpy as np
import oxtorch

# Create a large array
data = np.random.randn(8192, 8192).astype(np.float32)

# Save as raw binary
data.tofile("large_tensor.bin")

# Load in OxTorch
t_ssd = oxtorch.from_ssd("large_tensor.bin", shape=(8192, 8192), dtype=oxtorch.f32)
```

### Using PyTorch
```python
import torch
import oxtorch

# Create a torch tensor
t_pt = torch.randn(4096, 4096).to(torch.float32)

# Save raw bytes
with open("torch_data.bin", "wb") as f:
    f.write(t_pt.numpy().tobytes())

# Load in OxTorch
t_ssd = oxtorch.from_ssd("torch_data.bin", shape=(4096, 4096), dtype=oxtorch.f32)
```

## Hybrid Computation

Once a tensor is loaded via `from_ssd`, you can use it in standard operations. OxTorch will automatically stream the disk data into CPU registers in high-performance tiles.

```python
import oxtorch

a_ram = oxtorch.randn(8192, 8192)
b_ssd = oxtorch.from_ssd("large_tensor.bin", (8192, 8192), oxtorch.f32)

# This triggers the MSTS Tiled Hybrid Path
result = a_ram + b_ssd
```

> [!IMPORTANT]
> **O_DIRECT Requirement**: SSD-backed tensors use `O_DIRECT` for zero-copy I/O. This means the file must exist and have a size exactly matching `shape * dtype_size`. If the file is smaller, OxTorch will throw an I/O error.
