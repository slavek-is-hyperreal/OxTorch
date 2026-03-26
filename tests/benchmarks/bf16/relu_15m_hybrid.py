from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_15M_bf16_hybrid",
        op="ReLU",
        shape=(15000000,),
        mode="hybrid",
        dtype="bf16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
