from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Cat_f32_cpu",
        op="Cat",
        shape=(1000000,),
        mode="cpu",
        dtype="f32",
        is_ssd=False
    )
    bench.run()
