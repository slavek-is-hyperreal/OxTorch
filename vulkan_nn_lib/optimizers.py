import taichi as ti
import numpy as np
from . import kernels as K
import os

class Optimizer:
    def __init__(self, params, lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            if p.grad is None: continue
            K.k_sgd(p.arr, p.grad.arr, p.total_size, self.lr)
        ti.sync()

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, tile_size=4*1024*1024):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.tile_size = tile_size
        self.m = [ti.ndarray(ti.f32, shape=(p.total_size,)) for p in self.params]
        self.v = [ti.ndarray(ti.f32, shape=(p.total_size,)) for p in self.params]
        for mi, vi in zip(self.m, self.v):
            mi.from_numpy(np.zeros(mi.shape, dtype=np.float32))
            vi.from_numpy(np.zeros(vi.shape, dtype=np.float32))

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            K.k_adam(p.arr, p.grad.arr, self.m[i], self.v[i], 
                    self.lr, self.betas[0], self.betas[1], self.eps, self.t, p.total_size)
        ti.sync()

class HybridAdam(Optimizer):
    """Memory-efficient Adam that uses a VRAM tile to update RAM-resident params."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, tile_size=4*1024*1024):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.tile_size = tile_size
        self.m = [np.zeros(p.total_size, dtype=np.float32) for p in self.params]
        self.v = [np.zeros(p.total_size, dtype=np.float32) for p in self.params]
        # VRAM tile caches
        self.p_tile = ti.ndarray(ti.f32, shape=(tile_size,))
        self.g_tile = ti.ndarray(ti.f32, shape=(tile_size,))
        self.m_tile = ti.ndarray(ti.f32, shape=(tile_size,))
        self.v_tile = ti.ndarray(ti.f32, shape=(tile_size,))

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            n = p.total_size
            for start in range(0, n, self.tile_size):
                end = min(start + self.tile_size, n)
                curr_tile = end - start
                
                # Copy to VRAM
                self.p_tile.from_numpy(p.arr[start:end])
                self.g_tile.from_numpy(p.grad.arr[start:end])
                self.m_tile.from_numpy(self.m[i][start:end])
                self.v_tile.from_numpy(self.v[i][start:end])
                
                # Update
                K.k_adam_hybrid(self.p_tile, self.g_tile, self.m_tile, self.v_tile,
                               self.lr, self.betas[0], self.betas[1], self.eps, self.t, curr_tile)
                ti.sync()
                
                # Copy back
                p.arr[start:end] = self.p_tile.to_numpy()[:curr_tile]
                self.m[i][start:end] = self.m_tile.to_numpy()[:curr_tile]
                self.v[i][start:end] = self.v_tile.to_numpy()[:curr_tile]

class AutoSGD(Optimizer):
    """Hybrid SGD with Dynamic Budgeting and SSD Streaming."""
    def __init__(self, params, lr=1e-3, 
                 vram_budget=1500*1024*1024, # 1.5GB for a 2GB card
                 ram_budget='auto',
                 ssd_path="/vectorlegis_ssd_pool/vnn_cache"):
        from .tensor_store import TensorStore
        super().__init__(params, lr)
        
        if ram_budget == 'auto':
            from .streaming_ops import _get_available_ram
            avail = _get_available_ram()
            ram_budget = int(avail * 0.85)

        self.vram_budget = vram_budget
        self.ram_budget = ram_budget
        self.store = TensorStore(ssd_path) if ssd_path else None
        
        # GPU tile size (VBR Strategy)
        self.gpu_tile = int(vram_budget * 0.8 / (2 * 4)) # p + g
        self.gpu_tile = min(self.gpu_tile, 128 * 1024 * 1024)
        self.gpu_tile = max(self.gpu_tile, 1024 * 1024)
        
        self.strategies = []
        vram_used = 0
        has_hybrid = False
        
        print(f"[AutoSGD] Initializing smart states for {len(self.params)} params...")
        for i, p in enumerate(self.params):
            n = p.total_size
            vram_need = 2 * n * 4 # p + g
            
            if vram_used + vram_need <= vram_budget:
                self.strategies.append('vram')
                vram_used += vram_need
            else:
                self.strategies.append('hybrid')
                has_hybrid = True
        
        if has_hybrid:
            self.p_cache = ti.ndarray(ti.f32, shape=(self.gpu_tile,))
            self.g_cache = ti.ndarray(ti.f32, shape=(self.gpu_tile,))
        
        print(f"[AutoSGD] Memory: VRAM={vram_used/1e6:.1f}MB, GPU Tile: {self.gpu_tile/1e6:.1f}M elements")

    def _step_vram(self, p):
        K.k_sgd(p.arr, p.grad.arr, self.lr, p.total_size)

    def _step_hybrid(self, p):
        from . import streaming_ops as SOE
        n = p.total_size
        tile = min(n, self.gpu_tile)
        
        # 1. Dispatch GPU task (Non-blocking)
        self.p_cache.from_numpy(p.arr[:tile])
        self.g_cache.from_numpy(p.grad.arr[:tile])
        K.k_sgd(self.p_cache, self.g_cache, self.lr, tile)
        
        # 2. Parallel CPU/SSD task (OOM-Safe Tiled)
        # This runs while Vulkan is crunching its tile
        if n > tile:
            p_rem = p[tile:]
            g_rem = p.grad[tile:]
            SOE.SOE.sgd_update(p_rem, g_rem, self.lr)

        # 3. Final Synchronization
        ti.sync()
        p.arr[:tile] = self.p_cache.to_numpy()

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            if self.strategies[i] == 'vram':
                self._step_vram(p)
            else:
                self._step_hybrid(p)
        ti.sync()

class AutoAdam(Optimizer):
    """Ultimate Hybrid Optimizer with Dynamic Budgeting, SSD Streaming, and Parallel Execution.
    The goal is 'Max Productivity': utilize all VRAM, then all RAM, then SSD.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 vram_budget=1500*1024*1024, # 1.5GB for a 2GB card
                 ram_budget='auto',
                 ssd_path="/vectorlegis_ssd_pool/vnn_cache"):
        from .tensor_store import TensorStore
        super().__init__(params, lr)
        
        if ram_budget == 'auto':
            from .streaming_ops import _get_available_ram
            avail = _get_available_ram()
            ram_budget = int(avail * 0.85)

        self.betas = betas
        self.eps = eps
        self.t = 0
        self.vram_budget = vram_budget
        self.ram_budget = ram_budget
        self.store = TensorStore(ssd_path) if ssd_path else None
        
        # GPU tile size (VBR Strategy)
        self.gpu_tile = int(vram_budget * 0.8 / (4 * 4))
        self.gpu_tile = min(self.gpu_tile, 128 * 1024 * 1024)
        self.gpu_tile = max(self.gpu_tile, 1024 * 1024)
        
        self.strategies = []
        self.m = []
        self.v = []
        self.m_gpu = []
        self.v_gpu = []
        
        vram_used = 0
        ram_used = 0
        has_hybrid = False
        
        print(f"[AutoAdam] Initializing smart states for {len(self.params)} params...")
        for i, p in enumerate(self.params):
            n = p.total_size
            adam_vram_need = 3 * n * 4 # p, m, v (g is transient or same size)
            adam_ram_need = 2 * n * 4 # m, v
            
            if vram_used + adam_vram_need <= vram_budget:
                m_arr = ti.ndarray(ti.f32, shape=(n,))
                v_arr = ti.ndarray(ti.f32, shape=(n,))
                m_arr.from_numpy(np.zeros(n, dtype=np.float32))
                v_arr.from_numpy(np.zeros(n, dtype=np.float32))
                self.m_gpu.append(m_arr)
                self.v_gpu.append(v_arr)
                self.m.append(None)
                self.v.append(None)
                self.strategies.append('vram')
                vram_used += adam_vram_need
            else:
                self.m_gpu.append(None)
                self.v_gpu.append(None)
                
                if ram_used + adam_ram_need <= ram_budget:
                    self.m.append(np.zeros(n, dtype=np.float32))
                    self.v.append(np.zeros(n, dtype=np.float32))
                    self.strategies.append('hybrid')
                    ram_used += adam_ram_need
                elif self.store:
                    print(f"  > Param {i} ({n/1e6:.1f}M) -> SSD Overflow")
                    self.m.append(self.store.zeros(f"p{i}_m", shape=(n,)))
                    self.v.append(self.store.zeros(f"p{i}_v", shape=(n,)))
                    self.strategies.append('hybrid') # We use generic 'hybrid' now as it's tiled
                else:
                    self.m.append(np.zeros(n, dtype=np.float32))
                    self.v.append(np.zeros(n, dtype=np.float32))
                    self.strategies.append('hybrid')
                has_hybrid = True
        
        if has_hybrid:
            self.p_cache = ti.ndarray(ti.f32, shape=(self.gpu_tile,))
            self.g_cache = ti.ndarray(ti.f32, shape=(self.gpu_tile,))
            self.m_cache = ti.ndarray(ti.f32, shape=(self.gpu_tile,))
            self.v_cache = ti.ndarray(ti.f32, shape=(self.gpu_tile,))
        
        print(f"[AutoAdam] Memory: VRAM={vram_used/1e6:.1f}MB, RAM={ram_used/1e6:.1f}MB")
        print(f"[AutoAdam] GPU Tile: {self.gpu_tile/1e6:.1f}M elements")

    def _step_vram(self, i, p):
        K.k_adam(p.arr, p.grad.arr, self.m_gpu[i], self.v_gpu[i], 
                self.lr, self.betas[0], self.betas[1], self.eps, self.t, p.total_size)

    def _step_hybrid_vbr(self, i, p):
        from . import streaming_ops as SOE
        n = p.total_size
        tile = min(n, self.gpu_tile)
        
        # 1. Dispatch GPU task (Non-blocking)
        self.p_cache.from_numpy(p.arr[:tile])
        self.g_cache.from_numpy(p.grad.arr[:tile])
        self.m_cache.from_numpy(self.m[i][:tile])
        self.v_cache.from_numpy(self.v[i][:tile])
        
        K.k_adam_hybrid(self.p_cache, self.g_cache, self.m_cache, self.v_cache, 
                       self.lr, self.betas[0], self.betas[1], self.eps, self.t, tile)
        
        # 2. Parallel CPU/SSD task (OOM-Safe Tiled)
        # Tiled approach avoids PyTorch/NumPy mapping the whole 100GB at once
        if n > tile:
            p_rem = p[tile:]
            g_rem = p.grad[tile:]
            m_rem = self.m[i][tile:]
            v_rem = self.v[i][tile:]
            SOE.SOE.adam_update(p_rem, g_rem, m_rem, v_rem, self.lr, self.betas[0], self.betas[1], self.eps, self.t)

        # 3. Final Synchronization
        ti.sync()
        p.arr[:tile] = self.p_cache.to_numpy()
        self.m[i][:tile] = self.m_cache.to_numpy()
        self.v[i][:tile] = self.v_cache.to_numpy()

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            if self.strategies[i] == 'vram':
                self._step_vram(i, p)
            else:
                self._step_hybrid_vbr(i, p)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
