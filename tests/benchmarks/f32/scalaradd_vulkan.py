from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ScalarAdd_f32_vulkan",
        op="ScalarAdd",
        shape=(2048, 2048),
        mode="vulkan",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
