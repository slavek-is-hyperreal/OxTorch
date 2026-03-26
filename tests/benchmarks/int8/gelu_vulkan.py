from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="GELU_int8_vulkan",
        op="GELU",
        shape=(2048, 2048),
        mode="vulkan",
        dtype="int8",
        inplace=False,
        is_ssd=False
    )
    bench.run()
