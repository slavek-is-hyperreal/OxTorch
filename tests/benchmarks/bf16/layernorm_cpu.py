from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="LayerNorm_bf16_cpu",
        op="LayerNorm",
        shape=(2, 64, 4096),
        mode="cpu",
        dtype="bf16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
