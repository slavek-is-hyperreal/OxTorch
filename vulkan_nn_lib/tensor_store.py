import os
import numpy as np

class TensorStore:
    """SSD-backed tensor storage using numpy memmap on ZFS.
    
    Tensors live on SSD as raw binary files (no header for absolute speed).
    Access via numpy.memmap gives transparent RAM caching through 
    Linux page cache / ZFS ARC.
    """
    
    def __init__(self, base_path="/vectorlegis_ssd_pool/vnn_cache"):
        """base_path: directory on ZFS dataset."""
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
            
    def _get_path(self, name):
        return os.path.join(self.base_path, f"{name}.bin")
        
    def zeros(self, name, shape, dtype=np.float32, external_path=None):
        """Create or open a zero-initialized memmap tensor."""
        path = external_path if external_path else self._get_path(name)
        # Calculate total size in bytes
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        
        # Always ensure file is fresh and correct size
        with open(path, "wb") as f:
            if size > 0:
                f.seek(size - 1)
                f.write(b"\0")
            else:
                f.truncate(0)
        
        # Open as memmap
        m = np.memmap(path, dtype=dtype, mode='r+', shape=shape)
        return m
        
    def open(self, name, shape, dtype=np.float32, external_path=None):
        """Open existing memmap tensor."""
        path = external_path if external_path else self._get_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tensor {name} not found at {path}")
        return np.memmap(path, dtype=dtype, mode='r+', shape=shape)
        
    def cleanup(self):
        """Remove all stored tensors and directory."""
        if os.path.exists(self.base_path):
            for f in os.listdir(self.base_path):
                if f.endswith(".bin"):
                    try:
                        os.remove(os.path.join(self.base_path, f))
                    except:
                        pass
            try:
                os.removedirs(self.base_path)
            except:
                pass

    def delete(self, name):
        """Delete specific tensor."""
        path = self._get_path(name)
        if os.path.exists(path):
            os.remove(path)
