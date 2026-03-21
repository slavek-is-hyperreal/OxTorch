from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Chunk_f32_cpu",
        op="Chunk",
        shape=(1000, 1000),
        mode="cpu",
        dtype="f32",
        is_ssd=False
    )
    bench.run()
