import numpy as np
from . import kernels as K
from concurrent.futures import ThreadPoolExecutor
import threading
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

import queue

class TilePrefetcher:
    """Producer-consumer prefetcher for SSD tensors.
    Maximizes throughput by issuing parallel sequential background reads.
    """
    def __init__(self, tensor, tile_len, look_ahead=8):
        self.tensor = tensor
        self.tile_len = tile_len
        self.n = tensor.total_size
        self.look_ahead = look_ahead
        self.queue = queue.Queue(maxsize=look_ahead)
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=min(look_ahead, 8))
        self.thread = threading.Thread(target=self._orchestrator)
        self.thread.daemon = True
        self.thread.start()

    def _orchestrator(self):
        """Orchestrates parallel read jobs."""
        futures = []
        try:
            for start in range(0, self.n, self.tile_len):
                if self.stop_event.is_set(): break
                end = min(start + self.tile_len, self.n)
                
                # Submit read job
                future = self.executor.submit(self._read_tile, start, end)
                futures.append(future)
                
                # If we've hit look-ahead, wait for the oldest future
                if len(futures) >= self.look_ahead:
                    oldest = futures.pop(0)
                    tile_data = oldest.result()
                    if tile_data is not None:
                        self.queue.put(tile_data)
                        
            # Drain remaining
            for f in futures:
                tile_data = f.result()
                if tile_data is not None:
                    self.queue.put(tile_data)
                    
        except Exception as e:
            print(f"  [TilePrefetcher] ERROR: {e}")
        finally:
            self.queue.put(None) # Sentinel

    def _read_tile(self, start, end):
        try:
            # Explicitly copy to resident RAM
            return (start, end, self.tensor.arr[start:end].copy())
        except Exception as e:
            print(f"  [TilePrefetcher] Read Error at {start}: {e}")
            return None

    def get_tile(self):
        item = self.queue.get()
        return item

    def stop(self):
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        self.thread.join(timeout=1.0)

class SOE:
    """Streaming Operator Engine for tiled execution."""
    
    # Base tile size (will be scaled adaptively)
    BASE_TILE_SIZE_BYTES = 512 * 1024 * 1024 
    MAX_THREADS = 12 # Higher for concurrent prefetching on high-end CPUs
    RAM_SAFETY_MARGIN = 0.85 # Use up to 85% of available RAM for maximum greed

    @staticmethod
    def mean(a):
        """OOM-safe tiled mean using prefetched sum."""
        s = SOE.sum(a)
        m = s.to_numpy()[0] / a.total_size
        Tensor = get_tensor_class()
        return Tensor([m], shape=(), device='cpu')

    @staticmethod
    def elementwise_op(a, b, op_type, out_device='auto', extra=None):
        """Adaptive component-wise op using TilePrefetcher for look-ahead."""
        n = a.total_size
        item_size = np.dtype(a.dtype).itemsize
        total_size_bytes = n * item_size
        avail_ram = _get_available_ram()
        safe_budget = int(avail_ram * SOE.RAM_SAFETY_MARGIN)
        
        tile_len = max(1000, min(n, (safe_budget // (SOE.MAX_THREADS * 2)) // item_size))
        
        if out_device == 'auto':
            if total_size_bytes > safe_budget or a.device == 'ssd':
                out_device = 'ssd'
            else:
                out_device = 'cpu'
            
        Tensor = get_tensor_class()
        res = Tensor(None, shape=a.shape, device=out_device, dtype=a.dtype)
        
        # B-Cache logic
        b_val = b
        b_is_cached = False
        if isinstance(b, Tensor):
            b_size_bytes = b.total_size * item_size
            if b_size_bytes < safe_budget * 0.5 and b.total_size > 1:
                b_val = b.to_numpy().flatten()
                b_is_cached = True
            else:
                b_val = b.arr

        prefetcher_a = TilePrefetcher(a, tile_len)
        t0 = time.perf_counter()
        
        def process_tile(start, end, a_ram):
            if isinstance(b, Tensor):
                if b.total_size == 1:
                    b_ram = float(b.to_numpy().flatten()[0])
                elif b_is_cached:
                    b_ram = b_val[start:end]
                else:
                    b_ram = b_val[start:end].copy()
            else:
                b_ram = b_val
            
            out_ram = np.empty_like(a_ram)
            if op_type == 'add': np.add(a_ram, b_ram, out=out_ram)
            elif op_type == 'sub': np.subtract(a_ram, b_ram, out=out_ram)
            elif op_type == 'mul': np.multiply(a_ram, b_ram, out=out_ram)
            elif op_type == 'div': np.divide(a_ram, b_ram, out=out_ram)
            elif op_type == 'exp': np.exp(a_ram, out=out_ram)
            elif op_type == 'log': np.log(a_ram, out=out_ram)
            elif op_type == 'sqrt': np.sqrt(a_ram, out=out_ram)
            elif op_type == 'pow': np.power(a_ram, b_ram, out=out_ram)
            elif op_type == 'masked_fill':
                out_ram[:] = a_ram
                out_ram[b_ram > 0.5] = extra
                
            res.arr[start:end] = out_ram

        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            while True:
                item = prefetcher_a.get_tile()
                if item is None: break
                start, end, a_ram = item
                executor.submit(process_tile, start, end, a_ram)
            
        sys_stdout = __import__('sys').stdout
        sys_stdout.flush()
        return res

    @staticmethod
    def elementwise_reduce(a, b, op_type, reduction_type='sum'):
        """Fused element-wise op + reduction (e.g., (a * b).sum()).
        Saves one SSD write cycle by reducing in RAM.
        """
        n = a.total_size
        item_size = np.dtype(a.dtype).itemsize
        avail_ram = _get_available_ram()
        safe_budget = int(avail_ram * SOE.RAM_SAFETY_MARGIN * 0.8)
        
        # Parallel strategy: 
        # I/O depth = 8, Workers = 12. Total overlap ~20 tiles.
        look_ahead = 8
        total_overlap = SOE.MAX_THREADS + look_ahead
        tile_len = max(1000, min(n, (safe_budget // total_overlap) // item_size))
        
        prefetcher_a = TilePrefetcher(a, tile_len, look_ahead=look_ahead)
        
        # B-Cache logic
        b_val = b
        b_is_cached = False
        if isinstance(b, get_tensor_class()):
            if b.total_size * item_size < avail_ram * 0.2:
                b_val = b.to_numpy().flatten()
                b_is_cached = True
            else:
                b_val = b.arr

        total_sum = 0.0
        lock = threading.Lock()
        processed_bytes = 0
        t_start = time.perf_counter()
        t_last = t_start

        def process_tile_reduce(start, end, a_ram):
            nonlocal total_sum, processed_bytes, t_last
            if isinstance(b, get_tensor_class()):
                if b.total_size == 1:
                    b_ram = float(b.to_numpy().flatten()[0])
                elif b_is_cached:
                    b_ram = b_val[start:end]
                else:
                    b_ram = b_val[start:end].copy()
            else:
                b_ram = b_val
            
            # Workspace for intermediate op result in RAM
            if op_type == 'add': r = a_ram + b_ram
            elif op_type == 'sub': r = a_ram - b_ram
            elif op_type == 'mul': r = a_ram * b_ram
            elif op_type == 'div': r = a_ram / b_ram
            else: r = a_ram
            
            tile_sum = np.sum(r)
            with lock:
                total_sum += tile_sum
                processed_bytes += (end - start) * item_size
                
                # Heartbeat every 0.5s or 10%
                now = time.perf_counter()
                if now - t_last > 0.5:
                    pct = (processed_bytes / (n * item_size)) * 100
                    speed = (processed_bytes / (now - t_start)) / 1e6
                    print(f"    Progress: {pct:5.1f}% | Speed: {speed:7.1f} MB/s | RAM: {avail_ram/1e9:.1f}GB", end='\r')
                    t_last = now

        print(f"  [ARAS] Greedy Fused {op_type}+{reduction_type} on {n*item_size/1e6:.1f}MB SSD tensor...")
        print(f"         Budget: {safe_budget/1e9:.1f}GB | Tile: {tile_len*item_size/1e6:.1f}MB | Overlap: {total_overlap}")
        
        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            while True:
                item = prefetcher_a.get_tile()
                if item is None: break
                start, end, a_ram = item
                executor.submit(process_tile_reduce, start, end, a_ram)
                
        if reduction_type == 'mean':
            total_sum /= n
            
        t_final = time.perf_counter()
        final_speed = (n * item_size / (t_final - t_start)) / 1e6
        print(f"\n    Done in {t_final-t_start:.2f}s | Avg Speed: {final_speed:.1f} MB/s | Result: {total_sum}")
        Tensor = get_tensor_class()
        return Tensor([total_sum], shape=(), device='cpu')

    @staticmethod
    def sum(a):
        """OOM-safe tiled summation using TilePrefetcher."""
        n = a.total_size
        item_size = np.dtype(a.dtype).itemsize
        avail_ram = _get_available_ram()
        tile_len = max(1000, min(n, (int(avail_ram * 0.5) // SOE.MAX_THREADS) // item_size))
        
        prefetcher = TilePrefetcher(a, tile_len)
        total_sum = 0.0
        lock = threading.Lock()
        
        def reduce_tile(tile_data):
            nonlocal total_sum
            tile_sum = np.sum(tile_data)
            with lock:
                total_sum += tile_sum

        t0 = time.perf_counter()
        print(f"  [ARAS] Prefetched Sum on {n*item_size/1e6:.1f}MB SSD tensor...")
        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            while True:
                item = prefetcher.get_tile()
                if item is None: break
                _, _, a_ram = item
                executor.submit(reduce_tile, a_ram)
                
        print(f"    Done in {time.perf_counter()-t0:.2f}s | Sum: {total_sum}")
        Tensor = get_tensor_class()
        return Tensor([total_sum], shape=(), device='cpu')

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
    @staticmethod
    def sgd_update(p, g, lr):
        """OOM-safe tiled SGD update."""
        n = p.total_size
        item_size = np.dtype(p.dtype).itemsize
        avail_ram = _get_available_ram()
        # Each tile needs p_ram and g_ram (we update p in-place)
        tile_len = max(1000, min(n, (int(avail_ram * SOE.RAM_SAFETY_MARGIN) // (SOE.MAX_THREADS * 2)) // item_size))
        
        def process_tile(start, end):
            p_ram = p.arr[start:end].copy()
            g_ram = g.arr[start:end].copy()
            p_ram -= lr * g_ram
            p.arr[start:end] = p_ram

        offsets = range(0, n, tile_len)
        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            for start in offsets:
                end = min(start + tile_len, n)
                executor.submit(process_tile, start, end)

    @staticmethod
    def adam_update(p, g, m, v, lr, b1, b2, eps, t):
        """OOM-safe tiled Adam update."""
        n = p.total_size
        item_size = np.dtype(p.dtype).itemsize
        avail_ram = _get_available_ram()
        # Each tile needs p, g, m, v
        tile_len = max(1000, min(n, (int(avail_ram * SOE.RAM_SAFETY_MARGIN) // (SOE.MAX_THREADS * 4)) // item_size))
        
        b1_corr = 1.0 - pow(b1, float(t))
        b2_corr = 1.0 - pow(b2, float(t))
        step_size = lr * (np.sqrt(b2_corr)) / b1_corr

        def process_tile(start, end):
            p_ram = p.arr[start:end].copy()
            g_ram = g.arr[start:end].copy()
            m_ram = m[start:end].copy()
            v_ram = v[start:end].copy()
            
            m_ram = b1 * m_ram + (1.0 - b1) * g_ram
            v_ram = b2 * v_ram + (1.0 - b2) * (g_ram * g_ram)
            
            p_ram -= step_size * m_ram / (np.sqrt(v_ram) + eps)
            
            # Commit back
            p.arr[start:end] = p_ram
            m[start:end] = m_ram
            v[start:end] = v_ram

        offsets = range(0, n, tile_len)
        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            for start in offsets:
                end = min(start + tile_len, n)
                executor.submit(process_tile, start, end)
