from tests.benchmarks.base import BenchmarkBase

if __name__ == "__main__":
    b = BenchmarkBase("IndexSelect_f32_hybrid", "IndexSelect", [8192, 4096], mode="hybrid", dtype="f32", kwargs={"num_indices": 1024, "dim": 0})
    b.run()
