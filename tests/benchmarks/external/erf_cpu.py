from ..base import BenchmarkBase

if __name__ == "__main__":
    # erf is a unary fallback op
    bench = BenchmarkBase(
        name="Ext_Erf_f32_cpu",
        op="erf",
        shape=(2048, 2048),
        mode="cpu",
        dtype="f32"
    )
    bench.run()
