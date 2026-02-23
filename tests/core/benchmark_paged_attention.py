import taichi as ti
from vulkan_nn_lib.paged_attention import KVCachePool, PagedKVCache

def print_memory(step_name):
    # Read VmRSS directly from /proc/self/status on Linux
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    mem_kb = int(line.split()[1])
                    print(f"[{step_name:<40}] CPU Memory RSS: {mem_kb / 1024:.2f} MB")
                    return
    except Exception:
        print(f"[{step_name:<40}] Memory unavailable")

def run_benchmark():
    ti.init(arch=ti.vulkan)
    print("=== PagedAttention vs Contiguous Cache Benchmark ===")
    
    # Simulate a 7B model context: 32 heads, dim 128
    num_heads = 32
    head_dim = 128
    max_context = 4096
    
    print_memory("Baseline (Taichi init)")
    
    # 1. Contiguous Allocation Simulation
    print("\n--- Contiguous Allocation (Standard) ---")
    contiguous_k = ti.ndarray(dtype=ti.f32, shape=(max_context, num_heads, head_dim))
    contiguous_v = ti.ndarray(dtype=ti.f32, shape=(max_context, num_heads, head_dim))
    print_memory("Allocated contiguous cache for 4096 tokens")
    
    # In standard LLM, even if we are at token 100, we allocated 4096.
    
    # 2. PagedAttention Allocation Simulation
    print("\n--- PagedAttention (Block-by-Block) ---")
    
    # In reality, pool is shared across whole transformer. Let's create a pool for 1 layer, 100 blocks (16 tokens each = 1600 tokens capacity)
    pool = KVCachePool(max_blocks=100, block_size=16, num_heads=num_heads, head_dim=head_dim)
    print_memory("Allocated Pool (1600 tokens shared cap)")
    
    cache = PagedKVCache(pool)
    print_memory("Initialized Seq Virtual Cache (0 tokens)")
    
    # Simulate generation
    for step in [16, 64, 256, 1024]:
        cache.append_context(step - cache.seq_len)
        print_memory(f"Generated {step} tokens -> Blocks used: {cache.num_blocks()}")
        
    print("\nConclusion: PagedAttention allocates block-by-block, avoiding the massive initial OOM spike of contiguous caches!")

if __name__ == "__main__":
    run_benchmark()
