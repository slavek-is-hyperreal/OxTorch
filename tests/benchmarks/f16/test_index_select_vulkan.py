from tests.benchmarks.base import BenchmarkBase

if __name__ == "__main__":
    b = BenchmarkBase("IndexSelect_f16_vulkan", "IndexSelect", [8192, 4096], mode="vulkan", dtype="f16", kwargs={"num_indices": 1024, "dim": 0})
    b.run()
