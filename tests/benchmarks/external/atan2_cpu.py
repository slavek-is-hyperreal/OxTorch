from ..base import BenchmarkBase

if __name__ == "__main__":
    # atan2 is a binary fallback op
    bench = BenchmarkBase(
        name="Ext_Atan2_f32_cpu",
        op="atan2",
        shape=(2048, 2048),
        mode="cpu",
        dtype="f32"
    )
    bench.run()
