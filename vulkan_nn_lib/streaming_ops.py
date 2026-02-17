import numpy as np
from . import kernels as K
from concurrent.futures import ThreadPoolExecutor
import time

def get_tensor_class():
    from .tensor import Tensor
    return Tensor

def _get_available_ram():
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except:
        return 8 * 1024 * 1024 * 1024

class SOE:
    """Streaming Operator Engine for tiled execution."""
    
    # Tile size for parallel ops (smaller for better cache locality)
    TILE_SIZE_BYTES = 128 * 1024 * 1024 
    MAX_THREADS = 4 # Balance between RAM usage and I/O speed

    @staticmethod
    def elementwise_op(a, b, op_type, out_device='auto'):
        """Parallel component-wise op using tiling."""
        n = a.total_size
        item_size = np.dtype(a.dtype).itemsize
        tile_len = SOE.TILE_SIZE_BYTES // item_size
        
        # Decide output device
        if out_device == 'auto':
            avail = _get_available_ram()
            # If inputs are on SSD and size is significant, default to SSD
            inputs_on_ssd = (getattr(a, 'device', '') == 'ssd' or 
                             (isinstance(b, Tensor) and getattr(b, 'device', '') == 'ssd'))
            
            if n * item_size > 1024*1024*1024 and inputs_on_ssd:
                out_device = 'ssd'
            elif n * item_size > avail * 0.7: 
                out_device = 'ssd'
            else: 
                out_device = 'cpu'
            
        Tensor = get_tensor_class()
        res = Tensor(None, shape=a.shape, device=out_device, dtype=a.dtype)
        
        def process_tile(start, end):
            a_tile = a.arr[start:end]
            b_tile = b.arr[start:end] if isinstance(b, Tensor) else b
            
            # Use in-place to avoid extra copies
            out_buff = res.arr[start:end]
            if op_type == 'add':
                np.add(a_tile, b_tile, out=out_buff)
            elif op_type == 'sub':
                np.subtract(a_tile, b_tile, out=out_buff)
            elif op_type == 'mul':
                np.multiply(a_tile, b_tile, out=out_buff)
            elif op_type == 'div':
                np.divide(a_tile, b_tile, out=out_buff)

        print(f"  [SOE] Parallel {op_type.upper()} on {n*item_size/1e6:.1f}MB ({out_device})...")
        t0 = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            offsets = range(0, n, tile_len)
            total_tiles = len(offsets)
            for i, start in enumerate(offsets):
                end = min(start + tile_len, n)
                executor.submit(process_tile, start, end)
                # Show progress every 10%
                if i % max(1, total_tiles // 10) == 0:
                    print(f"    Progress: {i/total_tiles*100:.1f}%")
                    
        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return res

    @staticmethod
    def matmul(a, b, out_device='auto'):
        """Parallel Block-Based Matrix Multiplication."""
        M, K_dim = a.shape
        K2, N = b.shape
        item_size = np.dtype(a.dtype).itemsize
        
        # Decide output device
        out_size_bytes = M * N * item_size
        if out_device == 'auto':
            avail = _get_available_ram()
            if out_size_bytes > avail * 0.8: out_device = 'ssd'
            else: out_device = 'cpu'
            
        Tensor = get_tensor_class()
        res = Tensor(None, shape=(M, N), device=out_device, dtype=a.dtype)
        
        # Block size for M (how many rows to process at once)
        # We want to fit at least one row of A and full B (if possible)
        # or tiles of both.
        block_rows = max(1, 128 * 1024 * 1024 // (K_dim * item_size))
        
        print(f"  [SOE] Parallel Matmul {M}x{K_dim}x{N} ({out_device})...")
        t0 = time.perf_counter()
        
        # We pre-load B if it's small enough, otherwise B will be page-faulted.
        b_val = b.to_numpy() if b.total_size * item_size < 1024 * 1024 * 1024 else b.arr

        def process_block(start_m, end_m):
            a_block = a.arr[start_m * K_dim : end_m * K_dim].reshape(end_m - start_m, K_dim)
            res.arr[start_m * N : end_m * N] = np.matmul(a_block, b_val).flatten()

        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            m_offsets = range(0, M, block_rows)
            for i, start_m in enumerate(m_offsets):
                end_m = min(start_m + block_rows, M)
                executor.submit(process_block, start_m, end_m)
                if i % max(1, len(m_offsets) // 10) == 0:
                    print(f"    Progress: {i/len(m_offsets)*100:.1f}%")

        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return res
