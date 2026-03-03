import numpy as np
import taichi as ti
from .tensor import Tensor

class KVCachePool:
    """
    Manages the global pre-allocated physical VRAM blocks for all KV caches.
    Follows the vLLM PagedAttention paradigm.
    """
    def __init__(self, max_blocks: int, block_size: int, num_heads: int, head_dim: int, dtype=ti.f32):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # Physical Cache Arrays: [max_blocks, num_heads, block_size, head_dim]
        # In actual vLLM, it's often [max_blocks, num_heads, head_dim/x, block_size, x] for optimal memory access,
        # but for Vulkan Taichi, we keep it simple and rely on shader strides.
        self.physical_k = ti.ndarray(dtype=dtype, shape=(max_blocks, num_heads, block_size, head_dim))
        self.physical_v = ti.ndarray(dtype=dtype, shape=(max_blocks, num_heads, block_size, head_dim))
        
        # Stack of free physical indices
        self.free_blocks = list(range(max_blocks))
        
    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("KVCachePool is Out of Memory (OOM)! Cannot allocate another physical block.")
        return self.free_blocks.pop()
        
    def free_block(self, block_idx: int):
        self.free_blocks.append(block_idx)
        
    def clear(self):
        self.free_blocks = list(range(self.max_blocks))

class BlockTable:
    """
    Virtual-to-Physical translation table for a single sequence.
    Maps logical sequence chunks to scattered physical blocks from KVCachePool.
    """
    def __init__(self, pool: KVCachePool):
        self.pool = pool
        self.logical_to_physical = [] # List of physical block indices
        self.seq_len = 0
        
    def append_tokens(self, num_tokens: int):
        """Requests new memory for `num_tokens` if the last block is full."""
        tokens_in_last_block = self.seq_len % self.pool.block_size
        
        if self.seq_len == 0 or tokens_in_last_block == 0:
            space_in_last_block = 0
        else:
            space_in_last_block = self.pool.block_size - tokens_in_last_block
            
        remaining_tokens = num_tokens - space_in_last_block
        if remaining_tokens > 0:
            # We need to allocate new blocks
            new_blocks_needed = (remaining_tokens + self.pool.block_size - 1) // self.pool.block_size
            for _ in range(new_blocks_needed):
                phys_idx = self.pool.allocate_block()
                self.logical_to_physical.append(phys_idx)
                
        self.seq_len += num_tokens
        
    def get_physical_blocks(self) -> np.ndarray:
        return np.array(self.logical_to_physical, dtype=np.int32)
        
    def free(self):
        for phys_idx in self.logical_to_physical:
            self.pool.free_block(phys_idx)
        self.logical_to_physical = []
        self.seq_len = 0

class PagedKVCache:
    """
    A virtualization context for KV cache belonging to one Transformer layer/sequence.
    """
    def __init__(self, pool: KVCachePool):
        self.pool = pool
        self.block_table = BlockTable(pool)
        
    @property
    def seq_len(self):
        return self.block_table.seq_len
        
    def num_blocks(self):
        return len(self.block_table.logical_to_physical)
        
    def append_context(self, qk_len: int):
        self.block_table.append_tokens(qk_len)
        
    def free(self):
        self.block_table.free()
