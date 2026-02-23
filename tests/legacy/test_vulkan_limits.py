import taichi as ti
import time

ti.init(arch=ti.vulkan, offline_cache=True, device_memory_GB=1.0)

print("Starting max allocation test...")
arrays = []
try:
    for i in range(10000):
        # Create a tiny 1-float array
        arr = ti.ndarray(dtype=ti.f32, shape=(1,))
        arrays.append(arr)
        if i % 1000 == 0:
            print(f"Allocated {i} arrays...")
    print(f"Successfully allocated {len(arrays)} arrays! Taichi handles suballocation internally.")
except Exception as e:
    print(f"Failed at allocation {len(arrays)} with error: {e}")
