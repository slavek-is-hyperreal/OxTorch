from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_f32_hybrid",
        op="ReLU",
        shape=(1000000,),
        mode="hybrid",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
