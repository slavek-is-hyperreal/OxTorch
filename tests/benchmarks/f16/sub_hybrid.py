from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Sub_f16_hybrid",
        op="Sub",
        shape=(2048, 2048),
        mode="hybrid",
        dtype="f16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
