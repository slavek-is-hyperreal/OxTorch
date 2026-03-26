from tests.benchmarks.base import BenchmarkBase

if __name__ == "__main__":
    # LLaMA 7B vocab size shape (32000, 4096), indexing 1024 tokens
    b = BenchmarkBase("IndexSelect_f16_cpu", "IndexSelect", [8192, 4096], mode="cpu", dtype="f16", kwargs={"num_indices": 1024, "dim": 0})
    b.run()
