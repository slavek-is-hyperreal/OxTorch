import time
import taichi as ti
import Python_Legacy.vulkan_nn_lib.torch_shim as torch

def benchmark_matmul(size=1024, iterations=10):
    print(f"\n--- Benchmarking MatMul (Vulkan Backend) ---")
    print(f"Matrix Size: {size}x{size}")
    print(f"Iterations: {iterations}")
    print(f"Hardware: Vulkan/Taichi")
    print(f"--------------------------------------------")
    
    # Initialize tensors on Vulkan
    a = torch.randn(size, size, device='vulkan')
    b = torch.randn(size, size, device='vulkan')
    
    # Warmup
    print("Warming up JIT compiler...")
    _ = a @ b
    ti.sync() # Ensure warmup is finished
    
    print("Running benchmark...")
    times = []
    
    for i in range(iterations):
        t0 = time.perf_counter()
        c = a @ b
        ti.sync() # Wait for GPU to finish execution
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    
    # Calculate statistics
    avg_time = sum(times) / iterations
    min_time = min(times)
    
    # FLOPS calculation: 2 * N^3 for NxN matrix multiplication
    flops_per_iter = 2 * (size ** 3)
    tflops_avg = (flops_per_iter / avg_time) / 1e12
    tflops_peak = (flops_per_iter / min_time) / 1e12
    
    print(f"\nResults:")
    print(f"  Average Time: {avg_time * 1000:.2f} ms")
    print(f"  Peak Time:    {min_time * 1000:.2f} ms")
    print(f"  Avg TFLOPS:   {tflops_avg:.3f} TFLOPS")
    print(f"  Peak TFLOPS:  {tflops_peak:.3f} TFLOPS")
    
    # Verification check
    print(f"\nSanity Check: {c.shape} generated successfully.")

if __name__ == "__main__":
    benchmark_matmul(size=1024, iterations=20)
    benchmark_matmul(size=2048, iterations=10)
