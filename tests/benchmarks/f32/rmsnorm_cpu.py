from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="RMSNorm_f32_cpu",
        op="RMSNorm",
        shape=(2, 64, 4096),
        mode="cpu",
        dtype="f32",
        inplace=False,
        is_ssd=False
    )
    bench.run()
