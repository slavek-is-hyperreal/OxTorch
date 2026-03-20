from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Linear_int8_hybrid",
        op="Linear",
        shape=(2048, 2048),
        mode="hybrid",
        dtype="int8",
        inplace=False,
        is_ssd=False
    )
    bench.run()
