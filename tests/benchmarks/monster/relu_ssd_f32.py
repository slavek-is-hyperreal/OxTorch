from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Monster_ReLU_F32_SSD",
        op="ReLU",
        shape=(4000000000,),
        mode="cpu",
        dtype="f32",
        inplace=False,
        is_ssd=True
    )
    bench.run()
