from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Div_f32_vulkan",
        op="Div",
        shape=(1000000,),
        mode="vulkan",
        dtype="f32",
        is_ssd=False
    )
    bench.run()
