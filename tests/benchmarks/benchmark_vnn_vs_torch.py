import time
import torch
import numpy as np
import vulkan_nn_lib.core as vnn

def benchmark_adam_3way(K=4096, N=4096, iters=5, tile_size=4*1024*1024):
    total_elements = K * N
    num_tiles = (total_elements + tile_size - 1) // tile_size
    print(f"\n🚀 3-Way Adam Benchmark: weight ({K}x{N}) = {total_elements/1e6:.1f}M elements")
    print(f"   tile_size={tile_size/1e6:.1f}M → {num_tiles} tiles, {iters} iterations")
    print("=" * 70)
    
    # Shared initial state
    w_init = np.random.randn(K, N).astype(np.float32) * 0.02
    grads = [np.random.randn(K, N).astype(np.float32) * 0.01 for _ in range(iters)]

    # --- 1. PyTorch CPU ---
    wt = torch.tensor(w_init.copy(), requires_grad=True)
    opt_torch = torch.optim.Adam([wt], lr=1e-3)
    
    t0 = time.perf_counter()
    for g in grads:
        opt_torch.zero_grad()
        wt.grad = torch.tensor(g)
        opt_torch.step()
    torch_time = time.perf_counter() - t0
    torch_result = wt.detach().numpy().copy()

    # --- 2. VNN GPU-only Adam ---
    vnn.warmup()
    wv = vnn.Tensor(w_init.copy(), requires_grad=True)
    opt_gpu = vnn.Adam([wv], lr=1e-3, tile_size=tile_size)

    t0 = time.perf_counter()
    for g in grads:
        opt_gpu.zero_grad()
        wv.grad = vnn.Tensor(g)
        opt_gpu.step()
    gpu_time = time.perf_counter() - t0
    gpu_result = wv.to_numpy()

    # --- 3. VNN Hybrid Adam (CPU+GPU) ---
    wh = vnn.Tensor(w_init.copy(), requires_grad=True, device='cpu')
    opt_hybrid = vnn.HybridAdam([wh], lr=1e-3, tile_size=tile_size)

    t0 = time.perf_counter()
    for g in grads:
        opt_hybrid.zero_grad()
        wh.grad = vnn.Tensor(g)
        opt_hybrid.step()
    hybrid_time = time.perf_counter() - t0
    hybrid_result = wh.to_numpy()

    # --- Results ---
    print(f"\n{'Mode':<25} | {'Total':>10} | {'Per Step':>10} | {'vs Torch':>10}")
    print("-" * 70)
    
    def row(name, t, ref_t):
        ratio = ref_t / t
        icon = "🟢" if ratio >= 0.95 else "🔴"
        print(f"{name:<25} | {t*1000:8.1f}ms | {t*1000/iters:8.1f}ms | {icon} x{ratio:.2f}")
    
    row("PyTorch CPU (baseline)", torch_time, torch_time)
    row("VNN GPU-only", gpu_time, torch_time)
    row("VNN Hybrid (CPU+GPU)", hybrid_time, torch_time)
    
    # Verify numerical correctness
    print(f"\n📊 Numerical Verification (atol=1e-4):")
    gpu_ok = np.allclose(torch_result, gpu_result, atol=1e-4)
    hybrid_ok = np.allclose(torch_result, hybrid_result, atol=1e-4)
    print(f"   GPU vs Torch:    {'✅ MATCH' if gpu_ok else '❌ MISMATCH (max diff: ' + str(np.max(np.abs(torch_result - gpu_result)).item()) + ')'}")
    print(f"   Hybrid vs Torch: {'✅ MATCH' if hybrid_ok else '❌ MISMATCH (max diff: ' + str(np.max(np.abs(torch_result - hybrid_result)).item()) + ')'}")
    
    # GPU vs Hybrid
    hybrid_ok_vs_gpu = np.allclose(gpu_result, hybrid_result, atol=1e-5)
    print(f"   Hybrid vs GPU:   {'✅ MATCH' if hybrid_ok_vs_gpu else '❌ MISMATCH'}")
    
    # Speedup
    speedup = gpu_time / hybrid_time
    print(f"\n⚡ Hybrid vs GPU-only speedup: x{speedup:.2f}")

if __name__ == "__main__":
    # 4M tile = 16MB → 4 tiles for 4096x4096 (2 pairs = good parallelism)
    benchmark_adam_3way(K=4096, N=4096, iters=5, tile_size=4*1024*1024)
    
    # 4M tile → 16 tiles for 8192x8192 (8 pairs = great parallelism)
    benchmark_adam_3way(K=8192, N=8192, iters=3, tile_size=4*1024*1024)
