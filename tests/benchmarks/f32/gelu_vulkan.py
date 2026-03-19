from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="GELU_f32_vulkan",
        op="GELU",
        shape=(2048, 2048),
        mode="vulkan",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
