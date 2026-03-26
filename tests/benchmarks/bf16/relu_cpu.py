from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_bf16_cpu",
        op="ReLU",
        shape=(1000000,),
        mode="cpu",
        dtype="bf16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
