from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="RMSNorm_bf16_cpu",
        op="RMSNorm",
        shape=(2, 64, 4096),
        mode="cpu",
        dtype="bf16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
