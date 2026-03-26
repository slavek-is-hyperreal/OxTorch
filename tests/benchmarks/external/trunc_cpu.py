from ..base import BenchmarkBase

if __name__ == "__main__":
    # trunc is a unary fallback op
    bench = BenchmarkBase(
        name="Ext_Trunc_f32_cpu",
        op="trunc",
        shape=(2048, 2048),
        mode="cpu",
        dtype="f32"
    )
    bench.run()
