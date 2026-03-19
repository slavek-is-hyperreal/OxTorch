import vulkannn_rusted as vnn
import numpy as np

def test_f16_relu():
    print("Testing F16 ReLU on CPU...")
    # Create a random F16 tensor on CPU
    shape = [4, 4]
    x = vnn.Tensor.rand(shape, vnn.DataType.F16, "cpu")
    print(f"Tensor dtype: {x.dtype}")
    
    # Apply ReLU
    y = x.relu()
    
    # Check parity with numpy
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    expected = np.maximum(x_np, 0)
    
    diff = np.abs(y_np - expected).max()
    print(f"Max diff: {diff}")
    
    if diff < 1e-4:
        print("✅ Parity OK")
    else:
        print("❌ Parity FAIL")
        print(f"Input:\n{x_np}")
        print(f"Output:\n{y_np}")
        print(f"Expected:\n{expected}")

if __name__ == "__main__":
    test_f16_relu()
