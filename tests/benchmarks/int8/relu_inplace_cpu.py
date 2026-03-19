from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_int8_cpu_inplace",
        op="ReLU",
        shape=(1000000,),
        mode="cpu",
        dtype="int8",
        inplace=True,
        is_ssd=False
    )
    bench.run()
