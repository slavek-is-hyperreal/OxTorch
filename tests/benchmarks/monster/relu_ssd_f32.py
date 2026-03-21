from .monster_base import MonsterBenchmarkBase

if __name__ == "__main__":
    bench = MonsterBenchmarkBase(
        name="Monster_ReLU_F32_SSD",
        op="ReLU",
        dtype="f32",
        mode="cpu",
    )
    bench.run()
