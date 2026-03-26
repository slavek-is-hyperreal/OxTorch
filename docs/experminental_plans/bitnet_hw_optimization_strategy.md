# BitNet Optimization Report: Target Hardware Implementation

## Hardware Context: i5-3450 (Ivy Bridge) & AMD Bonaire (GCN 1.1)
The target system lacks **AVX2** (integer 256-bit) but supports **AVX1**, **SSSE3**, **F16C**, and **GCN 1.1 Asynchronous Compute**.

## 1. Dynamic Packing (1.6-bit vs 2.0-bit)
- **Problem**: 1GB VRAM on Bonaire is the absolute bottleneck. 2.0-bit packing wastes ~20% of capacity.
- **Solution**: 
  - **GPU**: Support 1.6-bit (Base-3) for model parameters to fit larger models.
  - **CPU**: Support 2.0-bit (TQ2_0) for maximum speed where memory bandwidth is higher (DDR3 Dual Channel).

## 2. The Shifted-Sum Trick (CPU - No-AVX2)
Since `_mm256_maddubs_epi16` requires AVX2, we fall back to the **SSSE3** implementation of the "Shifted-Sum" logic:
- **Math**: $(\sum W_i X_i) = (\sum V_i X_i) - \sum X_i$ where $V_i = W_i + 1$ (mapped to $\{0, 1, 2\}$).
- **Implementation**: 
  - Use 128-bit `_mm_maddubs_epi16`.
  - Process two 128-bit blocks per iteration.
  - This avoids branchy "if weight == -1" logic and achieves near-AVX2 speeds on Ivy Bridge.

## 3. Bit-Plane formulation (GPU - Bonaire)
Bonaire has no Tensor Cores. We utilize its high **Bit-Arithmetic throughput**:
- **Strategy**: Split 2-bit weights into two 32-bit registers ($W_{mask}$, $W_{sign}$).
- **Logic**:
  - `Positive_Bits = (Activation_bits & W_mask) & (~W_sign)`
  - `Negative_Bits = (Activation_bits & W_mask) & W_sign`
- **Reduction**: Use `bitCount()` (SPIR-V) which maps to hardware `v_bcnt_u32_b32`, followed by `subgroupAdd` (64-lane butterfly reduction) for maximum wavefront efficiency.

## 4. Register-Resident Fusion
To "collapse the memory wall" on a GPU with minimal VRAM and a PCIe 3.0 x16 bus:
- **Design**: One single Compute Shader launch for:
  - `BitLinear` (Dot product)
  - `Eager Dequantization` (FP32 conversion)
  - `ReLU^2` (Square activation)
  - `RMSNorm` (Parallel variance calculation)
- **Benefit**: Zero intermediate VRAM writes for activation tensors between these layers.
