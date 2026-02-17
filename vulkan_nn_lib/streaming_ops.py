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
    
    # Base tile size (will be scaled adaptively)
    BASE_TILE_SIZE_BYTES = 256 * 1024 * 1024 
    MAX_THREADS = 8 # Higher for concurrent prefetching
    RAM_SAFETY_MARGIN = 0.7 # Use up to 70% of available RAM

    @staticmethod
    def elementwise_op(a, b, op_type, out_device='auto'):
        """Adaptive component-wise op using explicit RAM prefetching."""
        n = a.total_size
        item_size = np.dtype(a.dtype).itemsize
        total_size_bytes = n * item_size
        
        # 1. Adaptive Budgeting
        avail_ram = _get_available_ram()
        safe_budget = int(avail_ram * SOE.RAM_SAFETY_MARGIN)
        
        # We want to fill the budget with active tiles.
        # Each "active" tile needs A_ram + B_ram + Out_ram.
        tile_len = max(1000, min(n, (safe_budget // (SOE.MAX_THREADS * 3)) // item_size))
        
        # Decide output device
        if out_device == 'auto':
            if total_size_bytes > safe_budget or a.device == 'ssd':
                out_device = 'ssd'
            else:
                out_device = 'cpu'
            
        Tensor = get_tensor_class()
        res = Tensor(None, shape=a.shape, device=out_device, dtype=a.dtype)
        
        print(f"  [ARAS] Greedy {op_type.upper()} on {total_size_bytes/1e6:.1f}MB ({out_device})")
        print(f"         RAM Budget: {safe_budget/1e9:.1f}GB | Tile: {tile_len*item_size/1e6:.1f}MB x {SOE.MAX_THREADS} threads")
        
        t0 = time.perf_counter()
        
        def process_tile(start, end):
            # 2. FORCE data into RAM by creating explicit copies
            # This makes the process use its own memory budget instead of just ZFS ARC.
            a_ram = a.arr[start:end].copy()
            
            if isinstance(b, Tensor):
                b_ram = b.arr[start:end].copy()
            else:
                b_ram = b
            
            # Workspace for result in RAM
            out_ram = np.empty_like(a_ram)
            
            if op_type == 'add': np.add(a_ram, b_ram, out=out_ram)
            elif op_type == 'sub': np.subtract(a_ram, b_ram, out=out_ram)
            elif op_type == 'mul': np.multiply(a_ram, b_ram, out=out_ram)
            elif op_type == 'div': np.divide(a_ram, b_ram, out=out_ram)
            elif op_type == 'exp': np.exp(a_ram, out=out_ram)
            elif op_type == 'log': np.log(a_ram, out=out_ram)
            elif op_type == 'sqrt': np.sqrt(a_ram, out=out_ram)
            elif op_type == 'sin': np.sin(a_ram, out=out_ram)
            elif op_type == 'cos': np.cos(a_ram, out=out_ram)
            elif op_type == 'pow': np.power(a_ram, b_ram, out=out_ram)
            elif op_type == 'gt': np.greater(a_ram, b_ram, out=out_ram)
            elif op_type == 'lt': np.less(a_ram, b_ram, out=out_ram)
            elif op_type == 'ge': np.greater_equal(a_ram, b_ram, out=out_ram)
            elif op_type == 'le': np.less_equal(a_ram, b_ram, out=out_ram)
            elif op_type == 'eq': np.equal(a_ram, b_ram, out=out_ram)
            elif op_type == 'ne': np.not_equal(a_ram, b_ram, out=out_ram)
            elif op_type == 'masked_fill':
                # b_ram is (mask, value) tuple or just mask if value is fixed
                mask, val = b_ram
                out_ram[:] = a_ram
                # Explicitly cast mask to bool for numpy indexing
                out_ram[mask > 0.5] = val
            
            # 3. Write back to SSD/Disk (memmap)
            res.arr[start:end] = out_ram

        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            # We use a sliding window of futures to control memory precisely
            offsets = range(0, n, tile_len)
            total_tiles = len(offsets)
            futures = []
            
            for i, start in enumerate(offsets):
                end = min(start + tile_len, n)
                # Submit tile processing
                futures.append(executor.submit(process_tile, start, end))
                
                # Progress every 10%
                if i % max(1, total_tiles // 10) == 0:
                    print(f"    Progress: {i/total_tiles*100:.1f}% | RAM: {i*tile_len*item_size*3/1e9:.1f}GB processed")

            for f in futures:
                f.result() # Wait for completion
                    
        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return res

    @staticmethod
    def matmul(a, b, out_device='auto'):
        """Adaptive Parallel Block Matrix Multiplication."""
        M, K_dim = a.shape
        K2, N = b.shape
        item_size = np.dtype(a.dtype).itemsize
        
        # 1. Adaptive Budgeting
        avail_ram = _get_available_ram()
        safe_budget = int(avail_ram * SOE.RAM_SAFETY_MARGIN)
        
        # Decide output device
        out_size_bytes = M * N * item_size
        if out_device == 'auto':
            if out_size_bytes > safe_budget or a.device == 'ssd' or b.device == 'ssd':
                out_device = 'ssd'
            else:
                out_device = 'cpu'
            
        Tensor = get_tensor_class()
        res = Tensor(None, shape=(M, N), device=out_device, dtype=a.dtype)
        
        # 2. Strategy: Load full B into RAM if it fits 50% of budget
        # Otherwise, B will stay as memmap (OS-level paging)
        b_size_bytes = b.total_size * item_size
        b_val = b.to_numpy() if b_size_bytes < safe_budget * 0.5 else b.arr
        
        # 3. Scale block size based on remaining budget
        remaining = safe_budget - (b_size_bytes if b_size_bytes < safe_budget * 0.5 else 0)
        # Each block of A takes (block_rows * K_dim) + (block_rows * N) space
        raw_block_size = remaining // (SOE.MAX_THREADS * 2)
        block_rows = max(1, min(M, raw_block_size // ((K_dim + N) * item_size)))
        
        print(f"  [ARAS] Matmul {M}x{K_dim}x{N} ({out_device})")
        print(f"         Budget: {safe_budget/1e9:.1f}GB | B-Cache: {'Yes' if b_size_bytes < safe_budget * 0.5 else 'No'}")
        print(f"         Block Rows: {block_rows} | Threads: {SOE.MAX_THREADS}")
        
        t0 = time.perf_counter()

        def process_block(start_m, end_m):
            # 2. Force A-block into resident RAM
            a_block_ram = a.arr[start_m * K_dim : end_m * K_dim].reshape(end_m - start_m, K_dim).copy()
            
            # 3. Compute in RAM
            # b_val is already in RAM if it fits the budget (see step 1)
            res_block_ram = np.matmul(a_block_ram, b_val)
            
            # 4. Write back to SSD (memmap)
            res.arr[start_m * N : end_m * N] = res_block_ram.flatten()

        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            m_offsets = range(0, M, block_rows)
            futures = []
            for i, start_m in enumerate(m_offsets):
                end_m = min(start_m + block_rows, M)
                futures.append(executor.submit(process_block, start_m, end_m))
                if i % max(1, len(m_offsets) // 10) == 0:
                    print(f"    Progress: {i/len(m_offsets)*100:.1f}% | RAM: {(i*block_rows*(K_dim+N)*item_size)/1e9:.1f}GB processed")

            for f in futures: f.result()

        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return res
