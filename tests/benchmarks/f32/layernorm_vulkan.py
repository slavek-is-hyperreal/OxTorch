from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="LayerNorm_f32_vulkan",
        op="LayerNorm",
        shape=(2, 64, 4096),
        mode="vulkan",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
