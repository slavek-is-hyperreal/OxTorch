from ..base import BenchmarkBase

if __name__ == "__main__":
    # Benchmark 1024x1024 MatMul with BOTH matrices transposed (non-contiguous)
    bench = BenchmarkBase(
        name="MatMul_f32_stride_vulkan",
        op="MatMul",
        shape=(1024, 1024),
        mode="vulkan",
        dtype="f32",
        transpose_a=True,
        transpose_b=True,
        iterations=5
    )
    bench.run()
