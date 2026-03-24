"""
Monster GELU F32 SSD

Tensor size = available_ram × 5.0 (always exceeds RAM).
Uses unary_op_ssd("gelu") directly → Path C (Full CrookScheduler, ≥32MB).
GELU uses the tanh approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
"""
from .monster_base import MonsterBenchmarkBase


class MonsterGELUF32SSD(MonsterBenchmarkBase):
    def _dispatch(self, a_ox, torch_ox):
        return a_ox._vnn.unary_op_ssd("gelu", 0.0, 0.0)

    def _torch_reference(self, a_t):
        import torch.nn.functional as F
        return F.gelu(a_t.float())


if __name__ == "__main__":
    bench = MonsterGELUF32SSD(
        name="Monster_GELU_F32_SSD",
        op="GELU",
        dtype="f32",
        mode="ssd",
    )
    bench.run()
