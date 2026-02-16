import taichi as ti
import numpy as np

try:
    ti.init(arch=ti.vulkan, offline_cache=False, device_memory_GB=1.0)
    print("Taichi initialized on Vulkan")
except Exception as e:
    print(f"Failed to init Taichi: {e}")
    exit(1)

@ti.kernel
def test_add(a: ti.types.ndarray(), b: ti.types.ndarray(), c: ti.types.ndarray(), n: int):
    for i in range(n):
        c[i] = a[i] + b[i]

N = 10
try:
    a_ti = ti.ndarray(ti.f32, shape=(N,))
    b_ti = ti.ndarray(ti.f32, shape=(N,))
    c_ti = ti.ndarray(ti.f32, shape=(N,))

    print("Created ndarrays")
    a_np = np.ones(N, dtype=np.float32)
    b_np = np.ones(N, dtype=np.float32) * 2.0
    
    a_ti.from_numpy(a_np)
    b_ti.from_numpy(b_np)
    print("Copied from numpy")
    
    test_add(a_ti, b_ti, c_ti, N)
    print("Kernel launched")
    
    ti.sync()
    result = c_ti.to_numpy()
    print(f"Result: {result}")
    
    expected = a_np + b_np
    if np.allclose(result, expected):
        print("SUCCESS: Vulkan works correctly")
    else:
        print("FAILURE: Vulkan result mismatch")
        print(f"Expected: {expected}")
        print(f"Got: {result}")

except Exception as e:
    print(f"Error during execution: {e}")
