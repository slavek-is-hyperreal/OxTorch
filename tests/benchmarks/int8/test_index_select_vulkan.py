from tests.benchmarks.base import BenchmarkBase

if __name__ == "__main__":
    b = BenchmarkBase("IndexSelect_int8_vulkan", "IndexSelect", [8192, 4096], mode="vulkan", dtype="int8", kwargs={"num_indices": 1024, "dim": 0})
    b.run()
