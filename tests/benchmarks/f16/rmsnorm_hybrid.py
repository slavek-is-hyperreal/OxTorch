from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="RMSNorm_f16_hybrid",
        op="RMSNorm",
        shape=(2, 64, 4096),
        mode="hybrid",
        dtype="f16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
