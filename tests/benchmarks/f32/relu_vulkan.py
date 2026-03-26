from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_f32_vulkan",
        op="ReLU",
        shape=(1000000,),
        mode="vulkan",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
