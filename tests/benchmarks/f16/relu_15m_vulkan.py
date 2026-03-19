from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_15M_f16_vulkan",
        op="ReLU",
        shape=(15000000,),
        mode="vulkan",
        dtype="f16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
