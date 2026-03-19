from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Softmax_bf16_cpu",
        op="Softmax",
        shape=(2048, 2048),
        mode="cpu",
        dtype="bf16",
        inplace=False,
        is_ssd=False
    )
    bench.run()
