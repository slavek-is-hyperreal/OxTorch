import numpy as np
from . import kernels as K
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import queue
import sys
from .memory import MemoryManager, SafetyViolationError

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
    def __init__(self, tensor, tile_len, look_ahead=None, num_consumers=1):
        self.tensor = tensor
        self.tile_len = tile_len
        self.n = tensor.total_size
        # For Greedy Mode, look_ahead is effectively dynamic based on RAM
        self.num_consumers = num_consumers
        # Set maxsize to look_ahead or a safe default to prevent unbounded RAM usage
        self.queue = queue.Queue(maxsize=look_ahead or 4) 
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=12) # Saturate ZFS RAID-0
        self.thread = threading.Thread(target=self._orchestrator)
        self.thread.daemon = True
        self.thread.start()

    def _orchestrator(self):
        """Orchestrates parallel read jobs."""
        futures = []
        try:
            # Wait for RAM before starting prefetch loop
            MemoryManager.wait_for_ram()

            for start in range(0, self.n, self.tile_len):
                if self.stop_event.is_set(): break
                end = min(start + self.tile_len, self.n)
                
                # DRAS v4: ADAPTIVE BACKOFF
                # Measure current risk and available budget
                budget = MemoryManager.get_safe_budget()
                tile_size = (end - start) * self.tensor.item_size
                # Limit outstanding I/O to 40% of currently safe budget
                max_outstanding = int((budget * 0.4) // tile_size)
                max_outstanding = max(2, min(50, max_outstanding))
                
                # If we've hit dynamic limit, wait for the oldest future
                while len(futures) >= max_outstanding:
                    f = futures.pop(0)
                    tile_data = f.result()
                    if tile_data is not None:
                        self.queue.put(tile_data)
                
                # Submit next read job
                future = self.executor.submit(self._read_tile, start, end)
                futures.append(future)
                
                # Slower submission if risk > 0.5 to let ZFS breathe
                risk = MemoryManager.get_usage_risk()
                if risk > 0.5: time.sleep(0.01 * (risk * 10))
                       
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
            # Unified storage access: if not SSD, we MUST use np_arr shadow
            if self.tensor.device == 'ssd':
                storage = self.tensor.arr
            else:
                # Restoration: ensure np_arr exists or fallback to full download
                if not hasattr(self.tensor, 'np_arr') or self.tensor.np_arr is None:
                     self.tensor.np_arr = self.tensor.to_numpy().flatten()
                storage = self.tensor.np_arr
            
            raw = storage[start:end].copy()
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
    
    MAX_THREADS = 12 # Increased for ZFS RAID-0 saturation

    @staticmethod
    def mean(a):
        """OOM-safe tiled mean using prefetched sum."""
        s = SOE.sum(a)
        val = s.to_numpy()
        m = (val[0] if val.ndim > 0 else val.item()) / a.total_size
        Tensor = get_tensor_class()
        return Tensor([m], shape=(), device='cpu')

    @staticmethod
    def elementwise_op(a, b, op_type, out_device='auto', extra=None):
        """Adaptive component-wise op using ParallelTilePrefetcher and PyTorch backend."""
        n_a = a.total_size
        n_b = b.total_size if isinstance(b, get_tensor_class()) else 1
        n = max(n_a, n_b)
        item_size = a.item_size
        total_size_bytes = int(n * item_size)
        
        safe_budget = MemoryManager.get_safe_budget()
        
        if out_device == 'auto':
            # Default to primary operand's device
            out_device = a.device
            # But if too large for RAM, force SSD
            if a.device != 'ssd' and total_size_bytes > safe_budget:
                out_device = 'ssd'
        
        # FINAL GUARD: If a is on SSD, result MUST be on SSD for component-wise
        if a.device == 'ssd': out_device = 'ssd'
            
        Tensor = get_tensor_class()
        
        # KAGGLE MODE REDIRECT
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (total_size_bytes > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            print(f"  [VNN] Offloading {op_type} to Kaggle...")
            return executor.submit_operation(op_type, a, b, extra)

        # Output is usually float32 for compute safety unless explicitly requested
        res_dtype = a.dtype if a.dtype != 'int4' else np.float32
        if op_type == 'div' or op_type == 'truediv': res_dtype = np.float32
        # Determine master tensor for tiling (prefetch the large one)
        master = a if n_a >= n_b else b
        
        look_ahead = 12
        total_overlap = SOE.MAX_THREADS + look_ahead
        tile_len = max(1000, min(n, (safe_budget // total_overlap) // item_size))
        
        # 1. Thaw operands: Ensure operands have valid RAM shadows BEFORE resolve and prefetch
        def thaw(t):
            if not isinstance(t, get_tensor_class()): return
            if t.device in ['vulkan', 'hybrid']:
                if not hasattr(t, 'np_arr') or t.np_arr is None:
                    t.np_arr = t.to_numpy().flatten()
        
        thaw(a); thaw(b); thaw(extra)

        # 2. Extract constant if one operand is scalar
        const_val = 0.0
        is_true_scalar = False
        if n_a != n_b:
            other = b if n_a > n_b else a
            if other is not None:
                if isinstance(other, get_tensor_class()) and other.total_size == 1:
                    const_val = float(other.to_numpy().flatten()[0])
                    is_true_scalar = True
                elif not isinstance(other, get_tensor_class()):
                    const_val = float(other)
                    is_true_scalar = True
            else:
                is_true_scalar = True

        # 3. Resolve Backend & Master
        eff_device = out_device if out_device != 'auto' else a.device
        if out_device == 'auto' and a.device == 'ssd': eff_device = 'ssd'
        
        use_vulkan = (eff_device == 'vulkan')
        use_hybrid = (eff_device == 'hybrid' or eff_device == 'ssd')
        use_kaggle = (eff_device == 'kaggle')
        
        supported_vulkan_ops = {'add', 'sub', 'mul', 'div', 'truediv', 'gt', 'lt', 'ge', 'le', 'eq', 'ne', 'relu', 'sigmoid', 'silu', 'sigmoid_backward', 'silu_backward_direct'}
        if op_type not in supported_vulkan_ops:
            use_vulkan = False
            use_hybrid = False
        
        master = a if n_a >= n_b else b
        swapped = (n_a < n_b)
        n_extra = extra.total_size if isinstance(extra, get_tensor_class()) else 1
        a_is_cached = (n_a < safe_budget * 0.4)
        b_is_cached = (n_b < safe_budget * 0.4)
        extra_is_cached = (n_extra < safe_budget * 0.4) if extra is not None else True

        # 4. Result Initialization (with RAM shadow for tiling)
        if eff_device in ['vulkan', 'hybrid']:
            res = Tensor(np.zeros(master.shape, dtype=res_dtype), device=eff_device)
        else:
            res = Tensor(None, shape=master.shape, device=eff_device, dtype=res_dtype)
        
        write_target = res.np_arr if hasattr(res, 'np_arr') else res.arr

        # 5. Resolve Storage for workers (Prioritize RAM shadow to avoid GPU sync)
        def get_storage(t):
            if t is None: return None
            # If not SSD, MUST have np_arr (ensured by thaw)
            if t.device != 'ssd':
                if not hasattr(t, 'np_arr') or t.np_arr is None:
                     from .tensor import Tensor
                     if isinstance(t, Tensor):
                         if isinstance(t.arr, np.ndarray):
                             t.np_arr = t.arr.flatten()
                         else:
                             t.np_arr = t.arr.to_numpy().flatten()
                return t.np_arr
            return t.arr

        b_val = b
        b_is_cached = False
        if isinstance(b, Tensor):
            if b.total_size * b.item_size < safe_budget * 0.2:
                # Small tensor: Cache as flat numpy for workers
                b_raw = get_storage(b)
                # Force f32 for Vulkan compatibility if needed, but here it's for workers
                b_val = b_raw.flatten().astype(np.float32) if use_vulkan or use_hybrid else b_raw.flatten()
                b_is_cached = True
            else:
                b_val = get_storage(b)
        
        prefetcher_m = TilePrefetcher(master, tile_len, look_ahead=look_ahead, num_consumers=(2 if use_hybrid else 1))
 
        ti_a = None; ti_b = None; ti_out = None; ti_extra = None
        if use_vulkan or use_hybrid:
            import taichi as ti
            ti_a = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
            ti_out = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
            if not is_true_scalar and getattr(b, 'total_size', 1) <= getattr(a, 'total_size', 1):
                ti_b = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
            if extra is not None:
                ti_extra = ti.ndarray(dtype=ti.f32, shape=(tile_len,))

        # 2. Extract constant logic moved up for cleanliness
 
        def process_tile_vulkan(start, end, m_ram):
            try:
                # 1. Prepare ti_a/ti_b based on broadcasting
                # We normalize: ti_a is ALWAYS the master (the one being prefetched)
                # If swapped, we use specialized 'reverse' kernels if needed
                
                # Padding last tile
                m_pad = m_ram
                if len(m_ram) < tile_len:
                    m_pad = np.zeros(tile_len, dtype=np.float32)
                    m_pad[:len(m_ram)] = m_ram
                ti_a.from_numpy(m_pad.astype(np.float32))
 
                if ti_b is not None:
                    # Same-shape or broadcasting
                    storage_b = get_storage(b) if n_a >= n_b else get_storage(a)
                    n_min = min(n_a, n_b)
                    
                    if n_a != n_b:
                        # Broadcast slicing
                        idx = np.arange(start, end) % n_min
                        other_ram = storage_b[idx]
                    else:
                        other_ram = storage_b[start:end].copy() if hasattr(storage_b, '__getitem__') else storage_b
                        
                    if len(other_ram) < tile_len:
                        p = np.zeros(tile_len, dtype=np.float32)
                        p[:len(other_ram)] = other_ram; other_ram = p
                    ti_b.from_numpy(other_ram.astype(np.float32))
                # const_val is already extracted in parent scope

                if ti_extra is not None:
                    if isinstance(extra, get_tensor_class()):
                        storage_extra = get_storage(extra)
                        extra_ram = storage_extra[start:end].copy()
                        if len(extra_ram) < tile_len:
                            p = np.zeros(tile_len, dtype=np.float32)
                            p[:len(extra_ram)] = extra_ram; extra_ram = p
                        ti_extra.from_numpy(extra_ram.astype(np.float32))
                    else:
                        # Extra is a scalar
                        e_pad = np.full(tile_len, float(extra), dtype=np.float32)
                        ti_extra.from_numpy(e_pad)

                # 3. Dispatch kernel
                is_swapped = (n_a < n_b) # True if b is master
                
                if ti_b is not None:
                    # Same shape array-array
                    if op_type == 'add': K.k_add(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'sub': 
                        if not is_swapped: K.k_sub(ti_a, ti_b, end-start, end-start)
                        else: 
                            # swapping handles res = ti_b - ti_a = a - b
                            K.k_sub(ti_b, ti_a, end-start, end-start)
                            # Copy from b to a to keep final copy logic unified if possible, or just copy to out
                            K.k_copy(ti_b, ti_out, end-start)
                    elif op_type == 'mul': K.k_mul(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'div' or op_type == 'truediv': 
                        if not is_swapped: K.k_div(ti_a, ti_b, end-start, end-start)
                        else:
                            K.k_div(ti_b, ti_a, end-start, end-start)
                            K.k_copy(ti_b, ti_out, end-start)
                    elif op_type == 'gt': K.k_gt(ti_a, ti_b, ti_out, end-start, end-start)
                    elif op_type == 'lt': K.k_lt(ti_a, ti_b, ti_out, end-start, end-start)
                    elif op_type == 'ge': K.k_ge(ti_a, ti_b, ti_out, end-start, end-start)
                    elif op_type == 'le': K.k_le(ti_a, ti_b, ti_out, end-start, end-start)
                    elif op_type == 'eq': K.k_eq(ti_a, ti_b, ti_out, end-start, end-start)
                    elif op_type == 'ne': K.k_ne(ti_a, ti_b, ti_out, end-start, end-start)
                    import taichi as ti; ti.sync()
                    if op_type in ['add', 'mul'] or (op_type in ['sub', 'div', 'truediv'] and not is_swapped):
                         K.k_copy(ti_a, ti_out, end-start)
                else:
                    # Scalar-Array broadcasting
                    if op_type == 'add': K.k_add_scalar(ti_a, const_val, end-start); K.k_copy(ti_a, ti_out, end-start)
                    elif op_type == 'mul': K.k_scale(ti_a, const_val, end-start); K.k_copy(ti_a, ti_out, end-start)
                    elif op_type == 'sub':
                        if not is_swapped: # a - const
                            K.k_add_scalar(ti_a, -const_val, end-start); K.k_copy(ti_a, ti_out, end-start)
                        else: # const - b
                            K.k_rsub_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'div' or op_type == 'truediv':
                        if not is_swapped: # a / const
                            K.k_scale(ti_a, 1.0/const_val, end-start); K.k_copy(ti_a, ti_out, end-start)
                        else: # const / b
                            K.k_rdiv_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'gt': K.k_gt_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'lt': K.k_lt_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'ge': K.k_ge_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'le': K.k_le_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'eq': K.k_eq_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'ne': K.k_ne_scalar(ti_a, const_val, ti_out, end-start)
                    elif op_type == 'relu': K.k_copy(ti_a, ti_out, end-start); K.k_relu_1d(ti_out, end-start)
                    elif op_type == 'sigmoid': K.k_sigmoid_1d(ti_a, end-start); K.k_copy(ti_a, ti_out, end-start)
                    elif op_type == 'silu': K.k_copy(ti_a, ti_out, end-start); K.k_silu_1d(ti_out, end-start)
                    elif op_type == 'sigmoid_backward': K.k_sigmoid_backward(ti_a, ti_extra, ti_out, end-start)
                    elif op_type == 'silu_backward_direct': K.k_silu_backward_direct(ti_a, ti_extra, ti_out, end-start)
                    import taichi as ti; ti.sync()
                
                ti.sync()
                res_tile = ti_out.to_numpy()
                write_target[start:end] = res_tile[:end-start]
            except Exception as e:
                print(f"    [SOE] Vulkan Panic in elementwise_op: {e}")
                raise

        def process_tile_cpu(start, end, m_ram):
            try:
                # m_ram is master, const_val/other_ram is slave
                _torch = _get_torch()
                if not is_true_scalar:
                    # other is an array (same shape or broadcast)
                    storage_other = get_storage(b) if not swapped else get_storage(a)
                    o_cached = b_is_cached if not swapped else a_is_cached
                    n_min = min(n_a, n_b)
                    
                    if n_a != n_b:
                        idx = np.arange(start, end) % n_min
                        other_ram = storage_other[idx]
                    else:
                        if not o_cached and hasattr(storage_other, '__getitem__'):
                            other_ram = storage_other[start:end].copy()
                        else:
                            other_ram = storage_other[start:end]
                            
                    a_t = _torch.from_numpy(m_ram if not swapped else other_ram)
                    b_t = _torch.from_numpy(other_ram if not swapped else m_ram)
                else:
                    # other is true scalar
                    a_t = _torch.from_numpy(m_ram) if not swapped else const_val
                    b_t = const_val if not swapped else _torch.from_numpy(m_ram)
                
                _torch = _get_torch()
                extra_t = None
                if extra is not None:
                    if hasattr(extra, 'arr') and hasattr(extra.arr, '__getitem__') and not extra_is_cached:
                        extra_t = _torch.from_numpy(extra.arr[start:end].copy())
                    elif isinstance(extra, (int, float)):
                        extra_t = extra
                    elif hasattr(extra, 'to_numpy'):
                         # If cached or small, it might benefit from full conversion or slice
                         if extra_is_cached:
                             val = extra.to_numpy().flatten()
                             # Handle case where extra is larger than tile (shouldn't happen if cached properly as scalar broadcast, but if same shape...)
                             # Actually extra is usually scalar or same shape.
                             # If cached and same shape, we need slicing.
                             if val.size > 1:
                                  extra_t = _torch.from_numpy(val[start:end])
                             else:
                                  extra_t = val[0]
                         else:
                             extra_t = _torch.from_numpy(extra.to_numpy())
                    else:
                        storage_extra = get_storage(extra)
                        if hasattr(storage_extra, '__getitem__'):
                             extra_t = _torch.from_numpy(storage_extra[start:end].copy())
                        else:
                             extra_t = _torch.from_numpy(storage_extra)

                if _torch is not None:
                    # a_t and b_t are now correctly ordered regardless of who is master
                    if op_type == 'add': r_t = a_t + b_t
                    elif op_type == 'sub': r_t = a_t - b_t
                    elif op_type == 'mul': r_t = a_t * b_t
                    elif op_type == 'div' or op_type == 'truediv': r_t = a_t / b_t
                    elif op_type == 'gt': r_t = (a_t > b_t).float()
                    elif op_type == 'lt': r_t = (a_t < b_t).float()
                    elif op_type == 'ge': r_t = (a_t >= b_t).float()
                    elif op_type == 'le': r_t = (a_t <= b_t).float()
                    elif op_type == 'eq': r_t = (a_t == b_t).float()
                    elif op_type == 'ne': r_t = (a_t != b_t).float()
                    elif op_type == 'exp': r_t = _torch.exp(a_t)
                    elif op_type == 'log': r_t = _torch.log(a_t + 1e-12)
                    elif op_type == 'sqrt': r_t = _torch.sqrt(a_t)
                    elif op_type == 'tanh': r_t = _torch.tanh(a_t)
                    elif op_type == 'pow': r_t = _torch.pow(a_t, b_t)
                    elif op_type == 'relu': r_t = _torch.relu(a_t)
                    elif op_type == 'leaky_relu': 
                        print(f"DEBUG: op_type={op_type}, extra={extra}")
                        r_t = _torch.nn.functional.leaky_relu(a_t, negative_slope=extra if extra is not None else 0.01)
                    elif op_type == 'sigmoid': r_t = _torch.sigmoid(a_t)
                    elif op_type == 'silu': r_t = _torch.nn.functional.silu(a_t)
                    elif op_type == 'sigmoid_backward':
                        sig = _torch.sigmoid(a_t)
                        r_t = extra_t * sig * (1.0 - sig)
                    elif op_type == 'silu_backward_direct':
                        # sig = 1 / (1 + exp(-x)), deriv = sig * (1 + x * (1 - sig))
                        sig = _torch.sigmoid(a_t)
                        r_t = extra_t * sig * (1.0 + a_t * (1.0 - sig))
                    elif op_type == 'leaky_relu_backward':
                        mask = (a_t > 0).float()
                        r_t = extra_t * (mask + (1.0 - mask) * extra) 
                    elif op_type == 'gelu_tanh_backward':
                        # Precise GELU derivative
                        r_t = extra_t * (0.5 * (1.0 + _torch.tanh(0.7978845608 * (a_t + 0.044715 * a_t**3))) + 
                                         0.5 * a_t * (1.0 - _torch.tanh(0.7978845608 * (a_t + 0.044715 * a_t**3))**2) * 
                                         0.7978845608 * (1.0 + 3.0 * 0.044715 * a_t**2))
                    elif op_type == 'gelu_tanh':
                        r_t = 0.5 * a_t * (1.0 + _torch.tanh(0.7978845608 * (a_t + 0.044715 * a_t**3)))
                    elif op_type == 'masked_fill':
                        r_t = a_t.clone()
                        r_t[b_t > 0.5] = extra
                    else: r_t = a_t
                res_np = r_t.numpy()
                write_target[start:end] = res_np.flatten()
                if hasattr(write_target, 'flush'): 
                    # Coherence: Flush each tile if writing directly to SSD memmap
                    write_target.flush()
            except Exception as e:
                print(f"    [SOE] process_tile_cpu ERROR: {e}")
                raise

        print(f"  [ARAS] Elementwise {op_type} on {total_size_bytes/1e6:.1f}MB tensor ({'Hybrid' if use_hybrid else 'Vulkan' if use_vulkan else 'CPU'})")
        
        gpu_lock = threading.Lock()
        def gpu_worker():
            while True:
                item = prefetcher_m.get_tile()
                if item is None: break
                s_v, e_v, m_v = item
                with gpu_lock:
                    process_tile_vulkan(s_v, e_v, m_v)

        t_start = time.perf_counter()
        if use_hybrid:
            # 1 GPU thread + (MAX_THREADS - 1) CPU threads
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS - 1) as cpu_executor:
                g_thread = threading.Thread(target=gpu_worker)
                g_thread.start()
                
                while True:
                    item_cpu = prefetcher_m.get_tile()
                    if item_cpu is None: break
                    sc, ec, mc = item_cpu
                    cpu_executor.submit(process_tile_cpu, sc, ec, mc)
                
                g_thread.join()
        elif use_vulkan:
            with ThreadPoolExecutor(max_workers=1) as executor: # GPU is serial per card
                while True:
                    item_v = prefetcher_m.get_tile()
                    if item_v is None: break
                    sv, ev, mv = item_v
                    executor.submit(process_tile_vulkan, sv, ev, mv)
        else:
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
                while True:
                    item_c = prefetcher_m.get_tile()
                    if item_c is None: break
                    s_c, e_c, m_c = item_c
                    executor.submit(process_tile_cpu, s_c, e_c, m_c)

        # Final Sync & Persistence
        if res.device == 'vulkan' and hasattr(res, 'np_arr') and res.np_arr is not None:
            # Convert to f32 for Taichi (default) and sync
            res.arr.from_numpy(res.np_arr.astype(np.float32))
            import taichi as ti
            ti.sync()
            
        if a.device == 'ssd' or out_device == 'ssd':
            if hasattr(res.arr, 'flush'):
                 res.arr.flush()
                 try: os.fsync(res.arr.fileno())
                 except: pass
            
        t_end = time.perf_counter()
        final_speed = (n * item_size / (t_end - t_start)) / 1e6
        print(f"    Done in {t_end-t_start:.2f}s | Avg Speed: {final_speed:.1f} MB/s")
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
        is_ssd_backed = (a.device == 'ssd')

        # Decide if we should go Hybrid
        use_vulkan = False
        use_hybrid = False
        if eff_device == 'hybrid' or (is_ssd_backed and can_vulkan and total_size_bytes > 1e9):
            use_hybrid = True
        elif eff_device == 'vulkan' or (is_ssd_backed and can_vulkan):
            use_vulkan = True

        prefetcher_a = TilePrefetcher(a, tile_len, num_consumers=(2 if use_hybrid else 1))
        torch = _get_torch()

        # KAGGLE MODE REDIRECT
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (total_size_bytes > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            print(f"  [VNN] Offloading {op_type}+{reduction_type} to Kaggle...")
            return executor.submit_operation(f"{op_type}_{reduction_type}", a, b, extra)

        ti_a = None
        ti_b = None
        ti_res_scalar = None
        if use_vulkan or use_hybrid:
            import taichi as ti
            @ti.kernel
            def k_reduce_sum(X: ti.types.ndarray(), Out: ti.types.ndarray(), Total: ti.i32):
                acc = 0.0 # Taichi f64 by default if initialized with float and no explicit cast
                for i in range(Total):
                    acc += ti.f64(X[i])
                Out[0] = acc 
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
                    if b.total_size == 1: 
                        b_val_current = float(b.to_numpy().flatten()[0])
                        is_b_scalar = True
                    elif b_is_cached: 
                        b_val_current = b_val[start:end]
                        is_b_scalar = False
                    else: 
                        b_val_current = b_val[start:end].copy()
                        is_b_scalar = False
                else: 
                    b_val_current = b_val # b_val is already the scalar
                    is_b_scalar = True
                
                ti_a.from_numpy(a_ram)
                
                if not is_b_scalar: # b is a tensor
                    ti_b.from_numpy(b_val_current)
                    if op_type == 'add': K.k_add(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'sub': K.k_sub(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'mul': K.k_mul(ti_a, ti_b, end-start, end-start)
                    elif op_type == 'div': K.k_div(ti_a, ti_b, end-start, end-start)
                else: # b is a scalar
                    if op_type == 'add': K.k_add_scalar(ti_a, b_val_current, end-start)
                    elif op_type == 'sub': K.k_sub_scalar(ti_a, b_val_current, end-start) # a - scalar
                    elif op_type == 'mul': K.k_scale(ti_a, b_val_current, end-start)
                    elif op_type == 'div': K.k_div_scalar(ti_a, b_val_current, end-start) # a / scalar
                
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
            futures = []
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS - 1) as cpu_executor:
                g_thread = threading.Thread(target=gpu_worker)
                g_thread.start()
                while True:
                    while len(futures) >= SOE.MAX_THREADS * 2:
                        f = futures.pop(0)
                        f.result()

                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    futures.append(cpu_executor.submit(process_tile_cpu, start, end, a_ram))
                
                for f in futures: f.result()
                g_thread.join()
        elif use_vulkan:
            futures = []
            with ThreadPoolExecutor(max_workers=1) as executor:
                while True:
                    while len(futures) >= 4:
                        f = futures.pop(0)
                        f.result()

                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    futures.append(executor.submit(process_tile_vulkan, start, end, a_ram))
                
                for f in futures: f.result()
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
                while True:
                    while len(futures) >= SOE.MAX_THREADS * 2:
                        f = futures.pop(0)
                        f.result()

                    item = prefetcher_a.get_tile()
                    if item is None: break
                    start, end, a_ram = item
                    futures.append(executor.submit(process_tile_cpu, start, end, a_ram))
                
                for f in futures: f.result()
                
        if reduction_type == 'mean':
            total_sum /= n
            
        t_final = time.perf_counter()
        final_speed = (n * item_size / (t_final - t_start)) / 1e6
        print(f"\n    Done in {t_final-t_start:.2f}s | Avg Speed: {final_speed:.1f} MB/s | Result: {total_sum}")
        Tensor = get_tensor_class()
        return Tensor([total_sum], shape=(), device='cpu')

    @staticmethod
    def sum(a):
        """OOM-safe tiled summation with Adaptive Restart."""
        retry_count = 0
        while True:
            try:
                n = a.total_size
                item_size = a.item_size
                safe_budget = MemoryManager.get_safe_budget()
                tile_len = max(1000, min(n, (safe_budget // (SOE.MAX_THREADS * 3)) // item_size))
                
                prefetcher = TilePrefetcher(a, tile_len)
                total_sum = 0.0
                lock = threading.Lock()
                
                ti_data = None
                ti_res_scalar = None
                if a.device == 'vulkan':
                    import taichi as ti
                    ti_data = ti.ndarray(dtype=ti.f32, shape=(tile_len,))
                    ti_res_scalar = ti.ndarray(dtype=ti.f64, shape=(1,)) # MATCH KERNEL f64

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
                print(f"  [DRAS v4] Tiled Sum on {n*item_size/1e6:.1f}MB (MaxUsage: {MemoryManager.MAX_TOTAL_USAGE_PCT*100:.0f}%)")
                futures = []
                with ThreadPoolExecutor(max_workers=SOE.MAX_THREADS) as executor:
                    while True:
                        # Manage backpressure on submitted tasks
                        while len(futures) >= SOE.MAX_THREADS * 2:
                            f = futures.pop(0)
                            f.result()

                        item = prefetcher.get_tile()
                        if item is None: break
                        _, _, a_ram = item
                        futures.append(executor.submit(reduce_tile, a_ram))
                
                # Wait for remaining
                for f in futures: f.result()
                
                t_end = time.perf_counter()
                final_speed = (n * item_size / (t_end - t0)) / 1e6
                print(f"    Done in {t_end-t0:.2f}s | Avg Speed: {final_speed:.1f} MB/s | Sum: {total_sum}")
                Tensor = get_tensor_class()
                return Tensor([total_sum], shape=(), device='cpu')
                
            except SafetyViolationError as e:
                prefetcher.stop()
                retry_count += 1
                MemoryManager.MAX_TOTAL_USAGE_PCT -= 0.05
                print(f"  [!] {e} - Reducing budget to {MemoryManager.MAX_TOTAL_USAGE_PCT*100:.0f}% and RESTARTING (Attempt {retry_count+1})...")
                time.sleep(1.0) # Let garbage collector breathe

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
        
        # KAGGLE MODE REDIRECT
        from .config import get_kaggle_enabled, get_kaggle_threshold
        if get_kaggle_enabled() and (a.total_size * a.item_size + b.total_size * b.item_size > get_kaggle_threshold()):
            from .kaggle_executor import KaggleExecutor
            executor = KaggleExecutor()
            print(f"  [VNN] Offloading MatMul to Kaggle...")
            return executor.submit_operation("matmul", a, b)

        b_size_bytes = b.total_size * b.item_size
        b_val = b.to_numpy() if b_size_bytes < safe_budget * 0.25 else b.arr.reshape(K2, N)
        
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

        t_end = time.perf_counter()
        final_speed = (M * K_dim * N * item_size / (t_end - t0)) / 1e6
        print(f"    Done in {t_end-t0:.2f}s | Avg Speed: {final_speed:.1f} MB/s")
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
