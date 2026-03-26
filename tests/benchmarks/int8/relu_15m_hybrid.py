from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_15M_int8_hybrid",
        op="ReLU",
        shape=(15000000,),
        mode="hybrid",
        dtype="int8",
        inplace=False,
        is_ssd=False
    )
    bench.run()
