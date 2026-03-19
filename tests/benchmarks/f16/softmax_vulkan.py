from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Softmax_f16_vulkan",
        op="Softmax",
        shape=(2048, 2048),
        mode="vulkan",
        dtype="f16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
