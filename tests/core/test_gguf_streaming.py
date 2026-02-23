import numpy as np
import vulkan_nn_lib as vnn
from vulkan_nn_lib.tensor import Tensor
import taichi as ti

def float16_to_bytes(val):
    return np.array([val], dtype=np.float16).view(np.uint8)

def test_q4_0_dequantization():
    ti.init(arch=ti.vulkan)
    print("--- Testing Q4_0 Zero-Copy Dequantization ---")
    
    # Block size is 32 elements. Total size is 18 bytes.
    num_blocks = 2
    raw_data = np.zeros((num_blocks * 18,), dtype=np.uint8)
    
    expected_f32 = np.zeros((num_blocks * 32,), dtype=np.float32)
    
    # Fill Block 0 (scale = 0.5)
    raw_data[0:2] = float16_to_bytes(0.5)
    for j in range(16):
        # We encode x0 = -8 (byte lower 0000), x1 = 7 (byte upper 1111)
        # byte: 1111 0000 -> 0xF0 = 240
        raw_data[2 + j] = 240
        expected_f32[0 * 32 + j] = (-8) * 0.5
        expected_f32[0 * 32 + j + 16] = (7) * 0.5
        
    # Fill Block 1 (scale = 2.0)
    raw_data[18:20] = float16_to_bytes(2.0)
    for j in range(16):
        # We encode x0 = 2 (byte lower 1010), x1 = -3 (byte upper 0101)
        # x0 offset: 2 + 8 = 10 -> 0xA
        # x1 offset: -3 + 8 = 5 -> 0x5
        # byte: 0101 1010 -> 0x5A = 90
        raw_data[18 + 2 + j] = 90
        expected_f32[1 * 32 + j] = 2 * 2.0
        expected_f32[1 * 32 + j + 16] = -3 * 2.0
        
    packed_gpu = Tensor(raw_data, device='vulkan')
    unpacked_gpu = Tensor(None, shape=(num_blocks * 32,), device='vulkan')
    
    import vulkan_nn_lib.kernels as K
    K.k_dequantize_q4_0(packed_gpu.arr, unpacked_gpu.arr, num_blocks)
    ti.sync()
    
    actual_f32 = unpacked_gpu.to_numpy()
    
    diff = np.max(np.abs(actual_f32 - expected_f32))
    print(f"Max Diff (Hardware Shader vs Python): {diff}")
    
    assert diff < 1e-4, "Dequantization mismatch!"
    print("SUCCESS: Sub-Byte Vulkan Shader is 100% correct!")

if __name__ == "__main__":
    test_q4_0_dequantization()
