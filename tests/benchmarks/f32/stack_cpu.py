from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Stack_f32_cpu",
        op="Stack",
        shape=(10000, 100),
        mode="cpu",
        dtype="f32",
        is_ssd=False
    )
    bench.run()
