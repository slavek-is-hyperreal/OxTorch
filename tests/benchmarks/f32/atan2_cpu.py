from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="Atan2_f32_cpu",
        op="Atan2",
        shape=(2048 * 4, 1024 * 4), # 32M elements (128MB per tensor)
        mode="cpu",
        dtype="f32",
        inplace=False,
        is_ssd=True # Test SSD-MSTS path for scientific validation
    )
    bench.run()
