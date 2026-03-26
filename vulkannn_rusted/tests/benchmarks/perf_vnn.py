import time
import torch
import numpy as np
import oxtorch as vnn

def benchmark_vnn_layer(M, K, N, device="cpu", iterations=10):
    print(f"\n--- VNN Performance: M={M}, K={K}, N={N} ({device}) ---")
    
    # Weights (N, K)
    w_q = torch.randint(-1, 2, (N, K)).to(torch.float32)
    w_vnn_int8 = vnn.Tensor(w_q.numpy(), dtype="int8")
    w_vnn = w_vnn_int8.to_bitnet("bitnet2").to(device)
    
    # Input (M, K)
    x_q = torch.randint(-128, 127, (M, K)).to(torch.float32)
    x_vnn = vnn.Tensor(x_q.numpy(), dtype="int8").to(device)
    
    # Scale (N,)
    s_vnn = vnn.Tensor(np.random.randn(N).astype(np.float32), dtype="float32").to(device)
    
    # Warmup
    for _ in range(3):
        _ = x_vnn.bit_linear(w_vnn, s_vnn)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        _ = x_vnn.bit_linear(w_vnn, s_vnn)
    end = time.perf_counter()
    
    avg_s = (end - start) / iterations
    ms = avg_s * 1000
    
    # Calculations
    ops = 2 * M * N * K
    gflops = (ops / avg_s) / 1e9
    
    print(f"  Avg Time: {ms:.3f} ms")
    print(f"  GFLOPS:   {gflops:.2f}")
    
    return ms

if __name__ == "__main__":
    # vnn.init() is automatic on import
    
    shapes = [
        (1, 2560, 6912),    # MLP Up/Gate (BitNet-2B)
        (1, 6912, 2560),    # MLP Down (BitNet-2B)
        (1, 2048, 2048),    # Standard Layer GEMV
        (128, 2048, 2048),  # Standard Layer GEMM (Prompt)
        (2048, 2048, 2048), # Large Batch GEMM
    ]
    
    results = {}
    
    for device in ["cpu", "vga"]:
        try:
            results[device] = []
            for M, K, N in shapes:
                ms = benchmark_vnn_layer(M, K, N, device=device)
                results[device].append(ms)
        except Exception as e:
            print(f"Error on {device}: {e}")

    # Summary Table
    print("\n" + "="*50)
    print(f"{'Shape (M, K, N)':<20} | {'CPU (ms)':<10} | {'VGA (ms)':<10} | {'Speedup':<8}")
    print("-" * 50)
    for i, (M, K, N) in enumerate(shapes):
        cpu_ms = results["cpu"][i]
        vga_ms = results["vga"][i] if "vga" in results else 0
        speedup = f"{cpu_ms/vga_ms:.2x}" if vga_ms > 0 else "N/A"
        print(f"{f'({M}, {K}, {N})':<20} | {cpu_ms:<10.3f} | {vga_ms:<10.3f} | {speedup:<8}")
    print("="*50)
