from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Sum_int8_cpu",
        op="Sum",
        shape=(2048, 2048),
        mode="cpu",
        dtype="int8",
        inplace=False,
        is_ssd=False
    )
    bench.run()
