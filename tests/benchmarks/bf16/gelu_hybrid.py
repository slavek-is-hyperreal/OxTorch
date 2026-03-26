from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="GELU_bf16_hybrid",
        op="GELU",
        shape=(2048, 2048),
        mode="hybrid",
        dtype="bf16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
