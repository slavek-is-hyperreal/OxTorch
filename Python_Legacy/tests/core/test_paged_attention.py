import os
import pytest
import torch
import torch.nn.functional as F
import numpy as np
import taichi as ti
from Python_Legacy.vulkan_nn_lib.tensor import Tensor
from Python_Legacy.vulkan_nn_lib.modules.layers import PagedAttention
from Python_Legacy.vulkan_nn_lib.paged_attention import KVCachePool, PagedKVCache

pytestmark = pytest.mark.skipif(os.environ.get("VNN_USE_RUST") == "1", reason="PagedAttention kernels not yet ported to Rust WGPU MVP")

def test_kv_cache_pool_allocation():
    ti.init(arch=ti.vulkan)
    
    pool = KVCachePool(max_blocks=10, block_size=16, num_heads=4, head_dim=64)
    assert len(pool.free_blocks) == 10
    
    # Check physical cache arrays
    assert pool.physical_k.shape == (10, 4, 16, 64)
    assert pool.physical_v.shape == (10, 4, 16, 64)
    
    idx1 = pool.allocate_block()
    assert idx1 == 9 # LIFO stack
    assert len(pool.free_blocks) == 9
    
    pool.free_block(idx1)
    assert len(pool.free_blocks) == 10

def test_paged_kv_cache_dynamic_scaling():
    ti.init(arch=ti.vulkan)
    pool = KVCachePool(max_blocks=100, block_size=16, num_heads=4, head_dim=64)
    
    cache = PagedKVCache(pool)
    assert cache.seq_len == 0
    assert cache.num_blocks() == 0
    
    # 1. Append 10 tokens (fits in 1 block of 16)
    cache.append_context(10)
    assert cache.seq_len == 10
    assert cache.num_blocks() == 1
    
    # 2. Append 10 more tokens (crosses into 2nd block)
    cache.append_context(10)
    assert cache.seq_len == 20
    assert cache.num_blocks() == 2
    
    # 3. Append 32 more tokens (exactly 2 more blocks)
    cache.append_context(32)
    assert cache.seq_len == 52
    assert cache.num_blocks() == 4
    
    # 4. Free
    cache.free()
    assert cache.seq_len == 0
    assert cache.num_blocks() == 0
    assert len(pool.free_blocks) == 100

def test_kv_cache_pool_oom():
    ti.init(arch=ti.vulkan)
    pool = KVCachePool(max_blocks=2, block_size=16, num_heads=1, head_dim=64)
    
    cache = PagedKVCache(pool)
    cache.append_context(32) # Uses exactly 2 blocks
    assert cache.num_blocks() == 2
    
    with pytest.raises(MemoryError):
        cache.append_context(1) # Tries to allocate 3rd block

def test_paged_attention_kernel():
    ti.init(arch=ti.vulkan)
    
    batch_size = 1
    num_heads = 2
    head_dim = 16
    seq_len = 35 # Crosses multiple blocks (e.g. block size 16)
    
    # 1. Setup VNN Paged Cache
    pool = KVCachePool(max_blocks=10, block_size=16, num_heads=num_heads, head_dim=head_dim)
    kv_cache = PagedKVCache(pool)
    kv_cache.append_context(seq_len)
    
    # Populate K and V with random data
    k_np = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32) * 0.1
    v_np = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32) * 0.1
    
    # Manually copy to physical blocks
    pool_k_np = pool.physical_k.to_numpy()
    pool_v_np = pool.physical_v.to_numpy()
    
    logical_blocks = kv_cache.block_table.get_physical_blocks()
    for t in range(seq_len):
        logical_block_idx = t // pool.block_size
        token_in_block = t % pool.block_size
        phys_block = logical_blocks[logical_block_idx]
        
        for h in range(num_heads):
            pool_k_np[phys_block, h, token_in_block, :] = k_np[0, t, h, :]
            pool_v_np[phys_block, h, token_in_block, :] = v_np[0, t, h, :]
            
    pool.physical_k.from_numpy(pool_k_np)
    pool.physical_v.from_numpy(pool_v_np)
    
    # 2. VNN Forward Pass
    q_np = np.random.randn(batch_size, num_heads * head_dim).astype(np.float32) * 0.1
    vnn_q = Tensor(q_np, device='vulkan')
    vnn_attn = PagedAttention(num_heads, head_dim)
    
    vnn_out = vnn_attn(vnn_q, kv_cache)
    ti.sync()
    vnn_res = vnn_out.to_numpy().reshape(batch_size, num_heads, head_dim)
    
    # 3. PyTorch Forward Pass (Standard SDP Attention)
    # PyTorch SDP expects: [B, H, L, D]
    pt_q = torch.from_numpy(q_np).view(batch_size, num_heads, 1, head_dim)
    pt_k = torch.from_numpy(k_np).permute(0, 2, 1, 3) # [B, H, L, D]
    pt_v = torch.from_numpy(v_np).permute(0, 2, 1, 3)
    
    pt_out = F.scaled_dot_product_attention(pt_q, pt_k, pt_v)
    pt_res = pt_out.numpy().reshape(batch_size, num_heads, head_dim)
    
    # 4. Parity Check
    diff = np.max(np.abs(vnn_res - pt_res))
    print(f"Max Diff block_size=16, seq_len={seq_len}: {diff}")
    assert diff < 1e-4
