from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ScalarAdd_f16_cpu",
        op="ScalarAdd",
        shape=(2048, 2048),
        mode="cpu",
        dtype="f16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
