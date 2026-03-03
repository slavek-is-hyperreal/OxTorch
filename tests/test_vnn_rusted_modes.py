import os
import numpy as np
import pytest
import vulkannn_rusted as vnn

def test_rust_cpu_mode_math():
    print("\n--- Testing VulkanNN Rusted CPU Mode ---")
    data1 = np.ones(100, dtype=np.float32) * 2.0
    data2 = np.ones(100, dtype=np.float32) * 3.0

    t1 = vnn.Tensor(data1, shape=[10], device="cpu")
    t2 = vnn.Tensor(data2, shape=[10], device="cpu")

    # Add
    t3 = t1 + t2
    assert t3.device == "cpu"
    assert np.allclose(t3.to_numpy(), 5.0)
    print("CPU Add OK")

    # MatMul
    tm1 = vnn.Tensor(np.ones((3, 4), dtype=np.float32) * 2.0, device="cpu")
    tm2 = vnn.Tensor(np.ones((4, 5), dtype=np.float32) * 3.0, device="cpu")
    tm3 = tm1 @ tm2
    assert tm3.device == "cpu"
    assert tm3.shape == [3, 5]
    assert np.allclose(tm3.to_numpy(), 24.0)
    print("CPU MatMul OK")

    # Activations
    ta = vnn.Tensor(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32), shape=[4], device="cpu")
    t_relu = ta.relu()
    assert np.allclose(t_relu.to_numpy(), [0.0, 0.0, 1.0, 2.0])
    print("CPU ReLU OK")


def test_rust_hybrid_mode_math():
    print("\n--- Testing VulkanNN Rusted HYBRID Mode ---")
    # Hybrid mode uses 70% GPU, 30% CPU, so we need large enough tensors to see the split 
    # without running into edge cases of 0 elements. Let's use 1000 elements.
    data1 = np.ones(1000, dtype=np.float32) * 5.0
    data2 = np.ones(1000, dtype=np.float32) * 7.0

    t1 = vnn.Tensor(data1, shape=[1000], device="hybrid")
    t2 = vnn.Tensor(data2, shape=[1000], device="hybrid")

    t3 = t1 + t2
    assert t3.device == "hybrid"
    assert np.allclose(t3.to_numpy(), 12.0)
    print("Hybrid Add OK")

    # Matmul Hybrid
    m, k, n = 100, 50, 40
    data_m1 = np.ones((m, k), dtype=np.float32) * 2.0
    data_m2 = np.ones((k, n), dtype=np.float32) * 3.0
    tm1 = vnn.Tensor(data_m1, device="hybrid")
    tm2 = vnn.Tensor(data_m2, device="hybrid")
    
    tm3 = tm1 @ tm2
    assert tm3.device == "hybrid"
    assert tm3.shape == [m, n]
    assert np.allclose(tm3.to_numpy(), 2.0 * 3.0 * k)
    print("Hybrid MatMul OK")

    # Activations Hybrid
    ta = vnn.Tensor(np.ones(1000, dtype=np.float32) * -2.0, shape=[1000], device="hybrid")
    t_relu = ta.relu()
    assert np.allclose(t_relu.to_numpy(), 0.0)
    print("Hybrid Relu OK")

if __name__ == "__main__":
    test_rust_cpu_mode_math()
    test_rust_hybrid_mode_math()
    print("✅ All parity tests passed!")
