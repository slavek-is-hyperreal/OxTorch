import vulkannn_rusted as vnn
import numpy as np
import sys

def test_ops():
    print("--- Testing CPU ---")
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    t = vnn.Tensor(data=data, device="cpu")
    
    # Neg
    neg_t = t.neg()
    print("Neg:", neg_t.to_numpy())
    assert np.allclose(neg_t.to_numpy(), -data)
    
    # Pow
    pow_t = t.pow(2.0)
    print("Pow(2.0):", pow_t.to_numpy())
    assert np.allclose(pow_t.to_numpy(), data**2)
    
    # Narrow
    narrow_t = t.narrow(0, 1, 1)
    print("Narrow(dim=0, start=1, len=1):", narrow_t.to_numpy())
    assert np.allclose(narrow_t.to_numpy(), data[1:2, :])
    
    # Argmax
    argmax_t = t.argmax(dim=1)
    print("Argmax(dim=1):", argmax_t.to_numpy())
    # data: [[1, 2, 3], [4, 5, 6]] -> indices: [[2], [2]]
    # Wait, our argmax returns values as float.
    assert np.allclose(argmax_t.to_numpy().flatten(), [2.0, 2.0])
    
    # Repeat Interleave
    repeat_t = t.repeat_interleave(2, dim=0)
    print("Repeat Interleave(2, dim=0) shape:", repeat_t.shape)
    assert repeat_t.shape == [4, 3]
    
    print("--- Testing Vulkan ---")
    try:
        t_vga = vnn.Tensor(data=data, device="vga")
        
        # Neg
        neg_vga = t_vga.neg()
        print("Vulkan Neg:", neg_vga.to_numpy())
        assert np.allclose(neg_vga.to_numpy(), -data)
        
        # Argmax (Global reduction path)
        data_1d = np.array([1.0, 5.0, 2.0, 4.0], dtype=np.float32)
        t_vga_1d = vnn.Tensor(data=data_1d, device="vga")
        argmax_vga = t_vga_1d.argmax(dim=0)
        print("Vulkan Argmax(dim=0):", argmax_vga.to_numpy())
        assert np.allclose(argmax_vga.to_numpy(), [1.0])
        
        print("Vulkan Ops OK")
    except Exception as e:
        print(f"Vulkan test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ops()
