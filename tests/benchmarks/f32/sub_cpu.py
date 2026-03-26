from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Sub_f32_cpu",
        op="Sub",
        shape=(2048, 2048),
        mode="cpu",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
