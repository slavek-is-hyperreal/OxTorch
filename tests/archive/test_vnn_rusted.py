import os
import numpy as np
import vulkannn_rusted as vnn

def test_rust_ssd_streaming():
    print("--- Testing VulkanNN Rusted SSD Streaming ---")
    
    # 1. Create a dummy binary file
    file1 = "dummy1.bin"
    file2 = "dummy2.bin"
    
    data1 = np.ones(100, dtype=np.float32) * 5.0
    data2 = np.ones(100, dtype=np.float32) * 2.0
    
    data1.tofile(file1)
    data2.tofile(file2)
    
    # 2. Map via Rust Native Memmap 
    # Zero copy! Python does not allocate the 100 floats in numpy array form.
    t1 = vnn.Tensor.from_ssd(file1, [100])
    t2 = vnn.Tensor.from_ssd(file2, [100])
    
    print(f"T1: {t1}")
    print(f"T2: {t2}")
    
    # 3. Native Addition via WGPU!
    # This directly pointers the mapped memory to the Vulkan buffer.
    t3 = t1 + t2
    
    print(f"T3 (Result): {t3}")
    print("T3 Sample:", t3.to_numpy()[:5])
    
    assert np.all(t3.to_numpy() == 7.0)
    print("✅ RUST NATIVE SSD STREAMING + WGPU MATH PARITY OK!")
    
    # Cleanup
    os.remove(file1)
    os.remove(file2)

if __name__ == "__main__":
    test_rust_ssd_streaming()
