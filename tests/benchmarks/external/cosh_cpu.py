from ..base import BenchmarkBase

if __name__ == "__main__":
    # cosh is a fallback op (not native in VNN yet)
    bench = BenchmarkBase(
        name="Ext_Cosh_f32_cpu",
        op="cosh",
        shape=(2048, 2048),
        mode="cpu",
        dtype="f32"
    )
    bench.run()
