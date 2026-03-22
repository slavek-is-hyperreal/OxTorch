# SSD Binary Format (.ssd)

OxTorch supports mapping massive tensors (up to terabytes) directly from SSD storage using the **SSD Streaming (MSTS)** architecture. These tensors are stored as raw binary files with the `.ssd` extension.

## File Structure

An `.ssd` file is a flat binary blob containing only the tensor data. There is no header or metadata within the file itself. Metadata (shape, strides, dtype) is managed by the OxTorch `Tensor` object and typically saved in a separate sidecar file or provided during initialization.

### Data Layout by Type
The data is stored in the file in row-major (C-style) order. Each element occupies a fixed width based on its `DataType`:

| DataType | Bytes per Element | Format |
| :--- | :--- | :--- |
| `F32` | 4 | IEEE 754 Single-precision float |
| `F16` | 2 | IEEE 754 Half-precision float |
| `BF16` | 2 | Brain Floating Point (Google format) |
| `Int8` | 1 | 8-bit signed integer |
| `Ternary` | 1 | 8-bit signed integer (unpacked values: -1, 0, 1) |

## I/O Requirements

OxTorch's **DirectIoEngine** uses Linux-specific `O_DIRECT` for zero-copy, zero-overhead I/O. To be compatible with this engine, `.ssd` files must meet several strict requirements:

### 1. Page Alignment
All I/O operations (reads/writes) must be performed on buffers and file offsets that are aligned to the system's **physical sector size** (typically 4096 bytes). 
- **Internal Handling**: OxTorch's `AlignedBuffer` ensures that the memory buffers used for `io_uring` satisfy this requirement.
- **File Offsets**: The engine handles non-aligned file offsets by reading the surrounding page and masking the target data, but for maximum performance, tensors should be aligned to 4KB boundaries.

### 2. File Pre-allocation
When creating a new SSD tensor (e.g., via `ones` or `zeros` with `device="ssd"`), OxTorch pre-allocates the file to its full size using `set_len`. This ensures that the filesystem (ext4, ZFS, XFS) reserves the necessary blocks on disk to avoid fragmentation and ensure sequential access speed.

### 3. Filesystem Support
- **ext4/XFS**: Fully supported and recommended for general use.
- **ZFS**: Requires careful tuning of `recordsize` to match OxTorch's tile size (typically 1MB) for optimal streaming.
- **Btrfs**: Supported, but `O_DIRECT` may have higher overhead depending on the mount options.

## Usage Guide

### Creating SSD Tensors
You can create an SSD-mapped tensor in Python:
```python
import oxtorch
# Create a 10GB zeros tensor on disk
t = oxtorch.zeros((1024, 1024, 256, 10), dtype=oxtorch.float32, device="ssd")
```

### Loading Existing Data
Existing binary data can be mapped as an OxTorch tensor:
```python
# Map an existing 5GB weights file
weights = oxtorch.from_ssd("model_weights.ssd", shape=(5120, 1048576), dtype=oxtorch.float16)
```

## Related Documentation
- [MSTS Logic](msts_logic.md): Deep dive into the streaming scheduler.
- [Architecture](architecture.md): Overview of the entire OxTorch stack.
