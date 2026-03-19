from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_f16_cpu_inplace",
        op="ReLU",
        shape=(1000000,),
        mode="cpu",
        dtype="f16",
        inplace=True,
        is_ssd=False
    )
    bench.run()
