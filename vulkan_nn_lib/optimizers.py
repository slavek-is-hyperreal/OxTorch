import numpy as np
from . import kernels as K
from .tensor import Tensor

class Optimizer:
    def __init__(self, params, lr=1e-3):
        # Flatten parameters to handle nested lists from torch models
        self.params = []
        for p in params:
            if isinstance(p, (list, tuple)): self.params.extend(p)
            else: self.params.append(p)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: 
                p.grad.zero_grad()

    def step(self):
        raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, tile_size=16*1024*1024):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.tile_size = tile_size
        
        # Optimizer states (m, v) reside in SYSTEM RAM (Mainframe)
        self.m = [np.zeros(p.total_size, dtype=np.float32) for p in self.params]
        self.v = [np.zeros(p.total_size, dtype=np.float32) for p in self.params]
        
        # VRAM Compute Cache (small buffers used for streaming)
        self.p_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))
        self.g_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))
        self.m_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))
        self.v_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))

    def step(self):
        self.t += 1
        b1, b2 = self.betas
        
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            
            # Use the VRAM as a computational cache - stream tiles
            total = p.total_size
            grad_np = p.grad.to_numpy().flatten()
            param_np = p.to_numpy().flatten()
            
            for offset in range(0, total, self.tile_size):
                curr_size = min(self.tile_size, total - offset)
                
                # If the tile is full size, use the cache. Otherwise, use a temp ndarray.
                # (Taichi from_numpy requires exact shape match)
                if curr_size == self.tile_size:
                    p_v = self.p_cache
                    g_v = self.g_cache
                    m_v = self.m_cache
                    v_v = self.v_cache
                else:
                    p_v = K.ti.ndarray(K.ti.f32, shape=(curr_size,))
                    g_v = K.ti.ndarray(K.ti.f32, shape=(curr_size,))
                    m_v = K.ti.ndarray(K.ti.f32, shape=(curr_size,))
                    v_v = K.ti.ndarray(K.ti.f32, shape=(curr_size,))

                # 1. Page IN (RAM -> VRAM Cache)
                p_v.from_numpy(param_np[offset : offset + curr_size])
                g_v.from_numpy(grad_np[offset : offset + curr_size])
                m_v.from_numpy(self.m[i][offset : offset + curr_size])
                v_v.from_numpy(self.v[i][offset : offset + curr_size])
                
                # 2. Compute (GPU Acceleration)
                K.k_adam_step(p_v, g_v, m_v, v_v,
                             self.lr, b1, b2, self.eps, self.t, curr_size)
                K.ti.sync()
                
                # 3. Page OUT (VRAM Cache -> RAM Source of Truth)
                param_np[offset : offset + curr_size] = p_v.to_numpy()
                self.m[i][offset : offset + curr_size] = m_v.to_numpy()
                self.v[i][offset : offset + curr_size] = v_v.to_numpy()
            
            # Update the source tensor (works for both VRAM and RAM tensors)
            p.load_from_numpy(param_np.reshape(p.shape))

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, tile_size=16*1024*1024):
        super().__init__(params, lr)
        self.tile_size = tile_size
        self.p_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))
        self.g_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))

    def step(self):
        for p in self.params:
            if p.grad is None: continue
            
            total = p.total_size
            grad_np = p.grad.to_numpy().flatten()
            param_np = p.to_numpy().flatten()

            for offset in range(0, total, self.tile_size):
                curr_size = min(self.tile_size, total - offset)
                
                if curr_size == self.tile_size:
                    p_v = self.p_cache
                    g_v = self.g_cache
                else:
                    p_v = K.ti.ndarray(K.ti.f32, shape=(curr_size,))
                    g_v = K.ti.ndarray(K.ti.f32, shape=(curr_size,))

                p_v.from_numpy(param_np[offset : offset + curr_size])
                g_v.from_numpy(grad_np[offset : offset + curr_size])
                
                K.k_sgd_step(p_v, g_v, self.lr, curr_size)
                K.ti.sync()
                
                param_np[offset : offset + curr_size] = p_v.to_numpy()
                
            p.load_from_numpy(param_np.reshape(p.shape))

class HybridAdam(Optimizer):
    """Adam optimizer that uses CPU and GPU as two parallel computers.
    
    Even tiles -> GPU (Vulkan k_adam_step kernel)
    Odd tiles  -> CPU (NumPy, uses AVX, zero PCIe overhead)
    Both run simultaneously via threading.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, tile_size=16*1024*1024):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.tile_size = tile_size
        
        # All state in RAM (Source of Truth)
        self.m = [np.zeros(p.total_size, dtype=np.float32) for p in self.params]
        self.v = [np.zeros(p.total_size, dtype=np.float32) for p in self.params]
        
        # VRAM Cache (only for GPU tiles)
        self.p_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))
        self.g_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))
        self.m_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))
        self.v_cache = K.ti.ndarray(K.ti.f32, shape=(tile_size,))

    def _cpu_adam_tile(self, param_np, grad_np, m, v, offset, size):
        """Pure NumPy Adam update. Runs on CPU cores (AVX). No PCIe overhead."""
        s = slice(offset, offset + size)
        b1, b2 = self.betas
        m[s] = b1 * m[s] + (1.0 - b1) * grad_np[s]
        v[s] = b2 * v[s] + (1.0 - b2) * grad_np[s] * grad_np[s]
        b1_corr = 1.0 - b1**self.t
        b2_corr = 1.0 - b2**self.t
        alpha = self.lr * np.sqrt(b2_corr) / b1_corr
        param_np[s] -= alpha * m[s] / (np.sqrt(v[s]) + self.eps)

    def _gpu_adam_tile(self, param_np, grad_np, m, v, offset, size):
        """Vulkan GPU Adam update. Pages tile through VRAM cache."""
        if size == self.tile_size:
            p_v, g_v, m_v, v_v = self.p_cache, self.g_cache, self.m_cache, self.v_cache
        else:
            p_v = K.ti.ndarray(K.ti.f32, shape=(size,))
            g_v = K.ti.ndarray(K.ti.f32, shape=(size,))
            m_v = K.ti.ndarray(K.ti.f32, shape=(size,))
            v_v = K.ti.ndarray(K.ti.f32, shape=(size,))

        s = slice(offset, offset + size)
        # Page IN
        p_v.from_numpy(param_np[s])
        g_v.from_numpy(grad_np[s])
        m_v.from_numpy(m[s])
        v_v.from_numpy(v[s])
        
        # Compute on GPU
        b1, b2 = self.betas
        K.k_adam_step(p_v, g_v, m_v, v_v,
                     self.lr, b1, b2, self.eps, self.t, size)
        K.ti.sync()
        
        # Page OUT
        param_np[s] = p_v.to_numpy()
        m[s] = m_v.to_numpy()
        v[s] = v_v.to_numpy()

    def step(self):
        import threading
        
        self.t += 1
        
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            
            total = p.total_size
            grad_np = p.grad.to_numpy().flatten()
            param_np = p.to_numpy().flatten()
            
            # Collect all tile offsets
            offsets = []
            for offset in range(0, total, self.tile_size):
                curr_size = min(self.tile_size, total - offset)
                offsets.append((offset, curr_size))
            
            # Process tiles in pairs: even=GPU, odd=CPU (simultaneously)
            idx = 0
            while idx < len(offsets):
                gpu_offset, gpu_size = offsets[idx]
                
                if idx + 1 < len(offsets):
                    # We have a pair — run GPU and CPU in parallel
                    cpu_offset, cpu_size = offsets[idx + 1]
                    
                    gpu_thread = threading.Thread(
                        target=self._gpu_adam_tile,
                        args=(param_np, grad_np, self.m[i], self.v[i], gpu_offset, gpu_size)
                    )
                    cpu_thread = threading.Thread(
                        target=self._cpu_adam_tile,
                        args=(param_np, grad_np, self.m[i], self.v[i], cpu_offset, cpu_size)
                    )
                    
                    gpu_thread.start()
                    cpu_thread.start()
                    gpu_thread.join()
                    cpu_thread.join()
                    
                    idx += 2
                else:
                    # Odd tile out — run on GPU solo
                    self._gpu_adam_tile(param_np, grad_np, self.m[i], self.v[i], gpu_offset, gpu_size)
                    idx += 1
            
            p.load_from_numpy(param_np.reshape(p.shape))
