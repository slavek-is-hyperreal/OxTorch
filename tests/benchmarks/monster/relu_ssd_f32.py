"""
Monster ReLU F32 SSD

Tensor size = available_ram × 1.2 (always exceeds RAM).
Uses unary_op_ssd() directly → guaranteed Path C (Full CrookScheduler).

Note: mode="ssd" here means "this tensor lives on SSD, not a device type".
The MonsterBenchmarkBase._dispatch() now calls unary_op_ssd("relu") instead
of torch_ox.relu() which would route through the PyTorch fallback.
"""
from .monster_base import MonsterBenchmarkBase


class MonsterReLUF32SSD(MonsterBenchmarkBase):
    def _dispatch(self, a_ox, torch_ox):
        # Call native MSTS path directly — bypasses oxtorch drop-in fallback.
        # For monster tensors this is always Path C (Full CrookScheduler, ≥32MB).
        return a_ox._vnn.unary_op_ssd("relu", 0.0, 0.0)


if __name__ == "__main__":
    bench = MonsterReLUF32SSD(
        name="Monster_ReLU_F32_SSD",
        op="ReLU",
        dtype="f32",
        mode="ssd",
    )
    bench.run()
