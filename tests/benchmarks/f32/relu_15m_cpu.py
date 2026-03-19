from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="ReLU_15M_f32_cpu",
        op="ReLU",
        shape=(15000000,),
        mode="cpu",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
