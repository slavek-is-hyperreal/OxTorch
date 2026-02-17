import numpy as np
from . import kernels as K
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import queue
import sys
from .memory import MemoryManager

def get_tensor_class():
    from .tensor import Tensor
    return Tensor

def _get_torch():
    try:
        import torch
        return torch
    except:
        return None

class TilePrefetcher:
    """Producer-consumer prefetcher for SSD tensors.
    Maximizes throughput by issuing parallel sequential background reads.
    """
    def __init__(self, tensor, tile_len, look_ahead=12, num_consumers=1):
        self.tensor = tensor
        self.tile_len = tile_len
        self.n = tensor.total_size
        self.look_ahead = look_ahead
        self.num_consumers = num_consumers
        self.queue = queue.Queue(maxsize=look_ahead)
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=min(look_ahead, 12))
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
                
                # DRAS: Wait if system RAM is critical before submitting NEW future
                MemoryManager.wait_for_ram()
                
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
            # Emit sentinels for all consumers
            for _ in range(self.num_consumers):
                self.queue.put(None)

    def _read_tile(self, start, end):
        try:
            # Explicitly copy to resident RAM
            raw = self.tensor.arr[start:end].copy()
            if self.tensor.dtype == 'int4':
                # Dequantize on-the-fly during prefetch
                n = end - start
                unpacked = np.zeros(n, dtype=np.float32)
                unpacked[0::2] = (raw & 0x0F).astype(np.float32) - 8.0
                unpacked[1::2] = ((raw >> 4) & 0x0F).astype(np.float32) - 8.0
                return (start, end, unpacked)
            return (start, end, raw)
        except Exception as e:
            print(f"  [TilePrefetcher] Read Error at {start}: {e}")
            return None

    def get_tile(self):
        item = self.queue.get()
        return item

    def stop(self):
        self.stop_event.set()
        self.executor.shutdown(wait=False)
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

class SOE:
    """Streaming Operator Engine for tiled execution."""
    
    MAX_THREADS = 8 

    @staticmethod
    def mean(a):
        """OOM-safe tiled mean using prefetched sum."""
        s = SOE.sum(a)
        m = s.to_numpy()[0] / a.total_size
        Tensor = get_tensor_class()
        return Tensor([m], shape=(), device='cpu')

    @staticmethod
    def elementwise_op(a, b, op_type, out_device='auto', extra=None):
        """Adaptive component-wise op using ParallelTilePrefetcher and PyTorch backend."""
        n = a.total_size
        item_size = a.item_size
        total_size_bytes = int(n * item_size)
        
        safe_budget = MemoryManager.get_safe_budget()
        
        if out_device == 'auto':
            if total_size_bytes > safe_budget or a.device == 'ssd':
                out_device = 'ssd'
            else:
                out_device = 'cpu'
            
        Tensor = get_tensor_class()
        # Output is usually float32 for compute safety unless explicitly requested
        res_dtype = a.dtype if a.dtype != 'int4' else np.float32
        res = Tensor(None, shape=a.shape, device=out_device, dtype=res_dtype)
        
        look_ahead = 12
        total_overlap = SOE.MAX_THREADS + look_ahead
        tile_len = max(1000, min(n, (safe_budget // total_overlap) // item_size))

        b_val = b
        b_is_cached = False
        if isinstance(b, Tensor):
            if b.total_size * b.item_size < safe_budget * 0.2:
                b_val = b.to_numpy().flatten()
                b_is_cached = True
            else:
                b_val = b.arr

        prefetcher_a = TilePrefetcher(a, tile_len, look_ahead=look_ahead)
        torch = _get_torch()
        
        # Automatic Heterogeneous Acceleration
        vram_budget = MemoryManager.get_vram_budget()
        can_vulkan = vram_budget > (tile_len * item_size * 3)
        
        # Decide if we should go Hybrid
        use_vulkan = False
        use_hybrid = False
        
        # Determine effective device
        eff_device = out_device if out_device != 'auto' else a.device
        
        if eff_device == 'hybrid' or (eff_device == 'auto' and can_vulkan and total_size_bytes > 1e9):
            use_hybrid = True
        elif eff_device == 'vulkan' or (eff_device == 'auto' and can_vulkan):
            use_vulkan = True

        ti_a = None
        ti_b = None
        ti_out = None
        if use_vulkan or use_hybrid:
            import taichi as ti
            ti_a = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
            ti_out = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
            if isinstance(b, get_tensor_class()) and b.total_size > 1:
                ti_b = ti.ndarray(dtype=ti.f32, shape=(tile_len,))

        prefetcher_a = TilePrefetcher(a, tile_len, look_ahead=look_ahead, num_consumers=(2 if use_hybrid else 1))

        def process_tile_vulkan(start, end, a_ram):
            try:
                if isinstance(b, get_tensor_class()):
                    if b.total_size == 1: b_ram = float(b.to_numpy().flatten()[0])
                    elif b_is_cached: b_ram = b_val[start:end]
                    else: b_ram = b_val[start:end].copy()
                else: b_ram = b_val
                
                ti_a.from_numpy(a_ram)
                if ti_b is not None:
                    ti_b.from_numpy(b_ram)
                    if op_type == 'add': K.k_add(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'sub': K.k_sub(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'mul': K.k_mul(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'div': K.k_div(ti_a, ti_b, end-start, end-start)
                    import taichi as ti
                    ti.sync()
                    K.k_copy(ti_a, ti_out, end-start)
                else:
                    if op_type == 'add': K.k_add_scalar(ti_a, b_ram, end-start)
                    elif op_type == 'mul': K.k_scale(ti_a, b_ram, end-start)
                    elif op_type == 'exp': K.k_exp(ti_a, end-start)
                    elif op_type == 'log': K.k_log(ti_a, end-start)
                    elif op_type == 'sqrt': K.k_sqrt(ti_a, end-start)
                    import taichi as ti
                    ti.sync()
                    K.k_copy(ti_a, ti_out, end-start)
                
                ti.sync()
                res.arr[start:end] = ti_out.to_numpy()
            except Exception as e:
                print(f"    [SOE] Vulkan Panic in elementwise_op: {e} | Falling back to CPU.")
                process_tile_cpu(start, end, a_ram)

        def process_tile_cpu(start, end, a_ram):
            if isinstance(b, get_tensor_class()):
                if b.total_size == 1: b_ram = float(b.to_numpy().flatten()[0])
                elif b_is_cached: b_ram = b_val[start:end]
                else: b_ram = b_val[start:end].copy()
            else: b_ram = b_val
            
            if torch:
                a_t = torch.from_numpy(a_ram)
                b_t = torch.from_numpy(b_ram) if isinstance(b_ram, np.ndarray) else b_ram
                if op_type == 'add': r_t = a_t + b_t
                elif op_type == 'sub': r_t = a_t - b_t
                elif op_type == 'mul': r_t = a_t * b_t
                elif op_type == 'div': r_t = a_t / b_t
                elif op_type == 'exp': r_t = torch.exp(a_t)
                elif op_type == 'log': r_t = torch.log(a_t)
                elif op_type == 'sqrt': r_t = torch.sqrt(a_t)
                elif op_type == 'pow': r_t = torch.pow(a_t, b_t)
                elif op_type == 'masked_fill':
                    r_t = a_t.clone()
                    r_t[b_t > 0.5] = extra
                else: r_t = a_t
                res.arr[start:end] = r_t.numpy()
            else:
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

        print(f"  [ARAS] Elementwise {op_type} on {total_size_bytes/1e6:.1f}MB tensor ({'Hybrid' if use_hybrid else 'Vulkan' if use_vulkan else 'CPU'})")
        
        gpu_lock = threading.Lock()
        def gpu_worker():
            while True:
                item = prefetcher_a.get_tile()
                if item is None: break
                start, end, a_ram = item
                with gpu_lock:
                    process_tile_vulkan(start, end, a_ram)

        if use_hybrid:
            # 1 GPU thread + (MAX_THREADS - 1) CPU threads
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS - 1) as cpu_executor:
                g_thread = threading.Thread(target=gpu_worker)
                g_thread.start()
                
                while True:
                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    cpu_executor.submit(process_tile_cpu, start, end, a_ram)
                
                # prefetcher_a.get_tile() willEventually return None for the gpu_worker too
                g_thread.join()
        elif use_vulkan:
            with ThreadPoolExecutor(max_workers=1) as executor: # GPU is serial per card
                while True:
                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    executor.submit(process_tile_vulkan, start, end, a_ram)
        else:
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
                while True:
                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    executor.submit(process_tile_cpu, start, end, a_ram)
            
        return res

    @staticmethod
    def elementwise_reduce(a, b, op_type, reduction_type='sum'):
        """High-performance fused op + reduction for SSD tensors."""
        n = a.total_size
        item_size = a.item_size
        total_size_bytes = int(n * item_size)
        
        safe_budget = MemoryManager.get_safe_budget()
        vram_budget = MemoryManager.get_vram_budget()
        
        look_ahead = 12
        total_overlap = SOE.MAX_THREADS + look_ahead
        
        # Base tile fits in RAM
        tile_len = max(1000, min(n, (safe_budget // total_overlap) // item_size))
        
        # If we want to use Vulkan, we must cap tile_len to VRAM budget 
        # (needs Space for A, B, Output in workspace)
        max_vram_tile_len = (vram_budget // 3) // item_size
        can_vulkan = max_vram_tile_len > 1000
        
        if can_vulkan:
            tile_len = min(tile_len, max_vram_tile_len)
        
        b_val = b
        b_is_cached = False
        if isinstance(b, get_tensor_class()):
            if b.total_size * b.item_size < safe_budget * 0.2:
                b_val = b.to_numpy().flatten()
                b_is_cached = True
            else:
                b_val = b.arr

        # Determine effective device
        eff_device = a.device if a.device in ['cpu', 'vulkan', 'hybrid'] else 'auto'
        
        # Decide if we should go Hybrid
        use_vulkan = False
        use_hybrid = False
        if eff_device == 'hybrid' or (eff_device == 'auto' and can_vulkan and total_size_bytes > 1e9):
            use_hybrid = True
        elif eff_device == 'vulkan' or (eff_device == 'auto' and can_vulkan):
            use_vulkan = True

        prefetcher_a = TilePrefetcher(a, tile_len, look_ahead=look_ahead, num_consumers=(2 if use_hybrid else 1))
        torch = _get_torch()

        ti_a = None
        ti_b = None
        ti_res_scalar = None
        if use_vulkan or use_hybrid:
            import taichi as ti
            ti_a = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
            ti_res_scalar = ti.ndarray(dtype=ti.f64, shape=(1,)) # HIGH PRECISION
            if isinstance(b, get_tensor_class()) and b.total_size > 1:
                ti_b = ti.ndarray(dtype=ti.f32, shape=(tile_len,))

        total_sum = 0.0
        lock = threading.Lock()
        
        processed_bytes = 0
        t_start = time.perf_counter()
        t_last = t_start

        def process_tile_vulkan(start, end, a_ram):
            nonlocal total_sum, processed_bytes, t_last
            try:
                if isinstance(b, get_tensor_class()):
                    if b.total_size == 1: b_ram = float(b.to_numpy().flatten()[0])
                    elif b_is_cached: b_ram = b_val[start:end]
                    else: b_ram = b_val[start:end].copy()
                else: b_ram = b_val
                
                ti_a.from_numpy(a_ram)
                if ti_b is not None:
                    ti_b.from_numpy(b_ram)
                    if op_type == 'add': K.k_add(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'sub': K.k_sub(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'mul': K.k_mul(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'div': K.k_div(ti_a, ti_b, end-start, end-start)
                else:
                    if op_type == 'add': K.k_add_scalar(ti_a, b_ram, end-start)
                    elif op_type == 'mul': K.k_scale(ti_a, b_ram, end-start)
                
                import taichi as ti
                ti.sync() # Synchronize before reading result to prevent Device Lost
                K.k_reduce_sum(ti_a, ti_res_scalar, end-start)
                ti.sync()
                tile_sum = float(ti_res_scalar.to_numpy()[0])
                with lock:
                    total_sum += tile_sum
                    processed_bytes += (end - start) * item_size
            except Exception as e:
                print(f"    [SOE] Vulkan Panic: {e} | Falling back to CPU for this tile.")
                process_tile_cpu(start, end, a_ram)

        def process_tile_cpu(start, end, a_ram):
            nonlocal total_sum, processed_bytes, t_last
            if isinstance(b, get_tensor_class()):
                if b.total_size == 1: b_ram = float(b.to_numpy().flatten()[0])
                elif b_is_cached: b_ram = b_val[start:end]
                else: b_ram = b_val[start:end].copy()
            else: b_ram = b_val
            
            if torch:
                a_t = torch.from_numpy(a_ram)
                b_t = torch.from_numpy(b_ram) if isinstance(b_ram, np.ndarray) else b_ram
                if op_type == 'add': r_t = a_t + b_t
                elif op_type == 'sub': r_t = a_t - b_t
                elif op_type == 'mul': r_t = a_t * b_t
                elif op_type == 'div': r_t = a_t / b_t
                else: r_t = a_t
                tile_sum = float(torch.sum(r_t.to(torch.float64)).item())
            else:
                if op_type == 'add': r = a_ram + b_ram
                elif op_type == 'sub': r = a_ram - b_ram
                elif op_type == 'mul': r = a_ram * b_ram
                elif op_type == 'div': r = a_ram / b_ram
                else: r = a_ram
                tile_sum = np.sum(r)
                
            with lock:
                total_sum += tile_sum
                processed_bytes += (end - start) * item_size

        mode_name = 'Hybrid' if use_hybrid else 'Vulkan' if use_vulkan else 'CPU'
        print(f"  [ARAS] Greedy Fused {op_type}+{reduction_type} on {total_size_bytes/1e6:.1f}MB ({mode_name})")
        
        def gpu_worker():
            while True:
                item = prefetcher_a.get_tile()
                if item is None: break
                start, end, a_ram = item
                process_tile_vulkan(start, end, a_ram)

        if use_hybrid:
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS - 1) as cpu_executor:
                g_thread = threading.Thread(target=gpu_worker)
                g_thread.start()
                while True:
                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    cpu_executor.submit(process_tile_cpu, start, end, a_ram)
                g_thread.join()
        elif use_vulkan:
            with ThreadPoolExecutor(max_workers=1) as executor:
                while True:
                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    executor.submit(process_tile_vulkan, start, end, a_ram)
        else:
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
                while True:
                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    executor.submit(process_tile_cpu, start, end, a_ram)
                
        if reduction_type == 'mean':
            total_sum /= n
            
        t_final = time.perf_counter()
        final_speed = (n * item_size / (t_final - t_start)) / 1e6
        print(f"\n    Done in {t_final-t_start:.2f}s | Avg Speed: {final_speed:.1f} MB/s | Result: {total_sum}")
        Tensor = get_tensor_class()
        return Tensor([total_sum], shape=(), device='cpu')

    @staticmethod
    def sum(a):
        """OOM-safe tiled summation."""
        n = a.total_size
        item_size = a.item_size
        safe_budget = MemoryManager.get_safe_budget()
        tile_len = max(1000, min(n, (safe_budget // (SOE.MAX_THREADS * 2)) // item_size))
        
        prefetcher = TilePrefetcher(a, tile_len)
        total_sum = 0.0
        lock = threading.Lock()
        
        # Vulkan Acceleration for Sum
        ti_data = None
        ti_res_scalar = None
        if a.device == 'vulkan':
            import taichi as ti
            ti_data = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
            ti_res_scalar = ti.ndarray(dtype=ti.f32, shape=(1,))

        def reduce_tile(tile_data):
            nonlocal total_sum
            if ti_data is not None:
                ti_data.from_numpy(tile_data)
                K.k_reduce_sum(ti_data, ti_res_scalar, len(tile_data))
                tile_sum = float(ti_res_scalar.to_numpy()[0])
            else:
                torch = _get_torch()
                if torch:
                    t_tile = torch.from_numpy(tile_data)
                    tile_sum = float(torch.sum(t_tile.to(torch.float64)).item())
                else:
                    tile_sum = np.sum(tile_data)
            with lock:
                total_sum += tile_sum

        t0 = time.perf_counter()
        print(f"  [ARAS] Tiled Sum on {n*item_size/1e6:.1f}MB tensor...")
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
        item_size = a.item_size
        
        safe_budget = MemoryManager.get_safe_budget()
        out_size_bytes = M * N * item_size
        if out_device == 'auto':
            if out_size_bytes > safe_budget or a.device == 'ssd' or b.device == 'ssd':
                out_device = 'ssd'
            
        Tensor = get_tensor_class()
        res = Tensor(None, shape=(M, N), device=out_device, dtype=a.dtype)
        
        b_size_bytes = b.total_size * b.item_size
        b_val = b.to_numpy() if b_size_bytes < safe_budget * 0.25 else b.arr
        
        remaining = safe_budget - (b_size_bytes if b_size_bytes < safe_budget * 0.25 else 0)
        raw_block_size = remaining // (SOE.MAX_THREADS * 2)
        block_rows = max(1, min(M, raw_block_size // ((K_dim + N) * item_size)))
        
        print(f"  [ARAS] Matmul {M}x{K_dim}x{N} ({out_device}) | Budget: {safe_budget/1e9:.1f}GB")
        t0 = time.perf_counter()

        def process_block(start_m, end_m):
            a_block_ram = a.arr[start_m * K_dim : end_m * K_dim].reshape(end_m - start_m, K_dim).copy()
            res_block_ram = np.matmul(a_block_ram, b_val)
            res.arr[start_m * N : end_m * N] = res_block_ram.flatten()

        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            m_offsets = range(0, M, block_rows)
            futures = [executor.submit(process_block, sm, min(sm + block_rows, M)) for sm in m_offsets]
            for f in futures: f.result()

        print(f"    Done in {time.perf_counter()-t0:.2f}s")
        return res

    @staticmethod
    def sgd_update(p, g, lr):
        """OOM-safe tiled SGD update."""
        n = p.total_size
        item_size = p.item_size
        safe_budget = MemoryManager.get_safe_budget()
        tile_len = max(1000, min(n, (safe_budget // (SOE.MAX_THREADS * 2)) // item_size))
        
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
        item_size = p.item_size
        safe_budget = MemoryManager.get_safe_budget()
        tile_len = max(1000, min(n, (safe_budget // (SOE.MAX_THREADS * 4)) // item_size))
        
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
            p.arr[start:end] = p_ram
            m[start:end] = m_ram
            v[start:end] = v_ram

        offsets = range(0, n, tile_len)
        with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
            for start in offsets:
                end = min(start + tile_len, n)
                executor.submit(process_tile, start, end)
