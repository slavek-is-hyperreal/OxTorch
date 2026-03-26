from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="LayerNorm_bf16_vulkan",
        op="LayerNorm",
        shape=(2, 64, 4096),
        mode="vulkan",
        dtype="bf16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
