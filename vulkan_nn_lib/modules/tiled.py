import numpy as np
import taichi as ti
from ..tensor import Tensor
from .base import Module
from .. import kernels as K

class TiledLinear(Module):
    """A Linear layer that pages weights from RAM to VRAM in tiles."""
    def __init__(self, in_features, out_features, bias=True, tile_size=2048, quant_type=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_size = tile_size
        self.quant_type = quant_type
        self.weight = Tensor(np.zeros((in_features, out_features), dtype=np.float32), device='cpu', requires_grad=True)
        self.has_bias = bias
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
            
        self.weight_tile_vram = ti.ndarray(dtype=ti.f32, shape=(in_features * tile_size,))
        if quant_type == 'q4_0':
            # Q4_0 packs 32 elements into 18 bytes
            self.raw_tile_vram = ti.ndarray(dtype=ti.u8, shape=(in_features * tile_size // 32 * 18,))
            
        self.grad_tile_vram = ti.ndarray(dtype=ti.f32, shape=(in_features * tile_size,))
        self.grad_out_tile_vram = None # Will be allocated once

    def forward(self, x: Tensor, sub_in_features=None, sub_out_features=None) -> Tensor:
        orig_shape = x.shape
        N_out = sub_out_features if sub_out_features else self.out_features
        K_A = x.shape[-1]
        K_active = sub_in_features if sub_in_features else K_A
        M = 1
        for s in orig_shape[:-1]: M *= s
        
        x_flat = x.reshape(M, K_A)
        out = Tensor(None, shape=(M, N_out))
        K.k_zero(out.arr, out.total_size)
        ti.sync()
        
        for n_offset in range(0, N_out, self.tile_size):
            n_tile = min(self.tile_size, N_out - n_offset)
            
            if self.quant_type == 'q4_0':
                # Simplified dummy copy for Q4_0 Integration Testing.
                # In real execution, `self.weight.arr` would be the zero-copy uint8 mmap.
                # We slice the required blocks and send to VRAM.
                blocks_needed = K_active * n_tile // 32
                bytes_needed = blocks_needed * 18
                raw_bytes = self.weight.arr[:bytes_needed] # Simplified slice
                
                # Fill the physical Vulkan buffer
                padded_raw = np.zeros(self.raw_tile_vram.shape[0], dtype=np.uint8)
                padded_raw[:len(raw_bytes)] = raw_bytes
                self.raw_tile_vram.from_numpy(padded_raw)
                
                ti.sync()
                # Run the dequantizer Shader to decode bits directly on the GPU registers
                K.k_dequantize_q4_0(self.raw_tile_vram, self.weight_tile_vram, blocks_needed)
                ti.sync()
            else:
                full_tile = np.zeros((K_active, self.tile_size), dtype=np.float32)
                full_tile[:, :n_tile] = self.weight.to_numpy()[:K_active, n_offset : n_offset + n_tile]
                self.weight_tile_vram.from_numpy(full_tile.flatten())
                ti.sync()
                _ = self.weight_tile_vram.to_numpy() # Force sync
                
            K.k_matmul_tiled(x_flat.arr, self.weight_tile_vram, out.arr, M, N_out, K_A, K_active, n_offset, n_tile, self.tile_size)
            ti.sync()
            
        if self.has_bias: out = out + self.bias
        res = out.reshape(*(list(orig_shape[:-1]) + [N_out]))
        res._prev = {x, self.weight}
        if self.has_bias: res._prev.add(self.bias)
        
        def _backward():
            grad_out_flat = res.grad.reshape(M, N_out)
            if self.weight.requires_grad:
                if self.weight.grad is None: self.weight.zero_grad()
                if self.grad_out_tile_vram is None:
                    self.grad_out_tile_vram = ti.ndarray(ti.f32, shape=(M * self.tile_size,))
                
                for n_offset in range(0, N_out, self.tile_size):
                    n_tile = min(self.tile_size, N_out - n_offset)
                    K.k_zero(self.grad_tile_vram, self.in_features * self.tile_size)
                    ti.sync()
                    grad_out_tile_np = grad_out_flat.to_numpy()[:, n_offset : n_offset + n_tile]
                    self.grad_out_tile_vram.from_numpy(np.pad(grad_out_tile_np, ((0,0), (0, self.tile_size - n_tile))).flatten())
                    ti.sync()
                    K.k_matmul_tiled_grad_w(x_flat.arr, self.grad_out_tile_vram, self.grad_tile_vram, M, K_A, n_tile, self.tile_size)
                    ti.sync()
                    grad_w_tile_ram = self.grad_tile_vram.to_numpy().reshape(self.in_features, self.tile_size)
                    self.weight.grad.arr.reshape(self.in_features, N_out)[:K_active, n_offset : n_offset + n_tile] += grad_w_tile_ram[:K_active, :n_tile]
            
            if x.requires_grad:
                if x.grad is None: x.zero_grad()
                for n_offset in range(0, N_out, self.tile_size):
                    n_tile = min(self.tile_size, N_out - n_offset)
                    full_tile = np.zeros((K_active, self.tile_size), dtype=np.float32)
                    full_tile[:, :n_tile] = self.weight.to_numpy()[:K_active, n_offset : n_offset + n_tile]
                    self.weight_tile_vram.from_numpy(full_tile.flatten())
                    grad_out_tile_np = grad_out_flat.to_numpy()[:, n_offset : n_offset + n_tile]
                    grad_out_tile_vram = ti.ndarray(ti.f32, shape=(M * n_tile,))
                    grad_out_tile_vram.from_numpy(grad_out_tile_np.flatten())
                    ti.sync()
                    _ = self.weight_tile_vram.to_numpy() # Force sync
                    _ = grad_out_tile_vram.to_numpy() # Force sync
                    K.k_matmul_tiled_grad_x(grad_out_tile_vram, self.weight_tile_vram, x.grad.arr, M, K_A, n_tile, self.tile_size)
                    ti.sync()

            if self.has_bias and self.bias.requires_grad:
                if self.bias.grad is None: self.bias.zero_grad()
                current_grad = self.bias.grad.to_numpy()
                new_grad_sum = grad_out_flat.to_numpy().sum(axis=0)
                self.bias.grad.load_from_numpy(current_grad + new_grad_sum)
        res._backward_fn = _backward
        return res

class MatFormerLinear(TiledLinear):
    def forward(self, x: Tensor, sub_in_features=None, sub_out_features=None) -> Tensor:
        return super().forward(x, sub_in_features=sub_in_features, sub_out_features=sub_out_features)

class TiledEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_ram = np.zeros((num_embeddings, embedding_dim), dtype=np.float32)
        # Placeholder for state_dict
        self.weight = Tensor(None, shape=(1, embedding_dim), device='cpu')

    def forward(self, x: Tensor) -> Tensor:
        indices = x.to_numpy().flatten().astype(np.int32)
        B, L = x.shape
        D = self.embedding_dim
        vectors = self.weight_ram[indices].reshape(B, L, D)
        return Tensor(vectors)

    def load_weight(self, data):
        self.weight_ram = data
        self.weight.shape = data.shape
