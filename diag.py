import numpy as np
import taichi as ti
ti.init(arch=ti.vulkan)
import vulkan_nn_lib as vnn
from vulkan_nn_lib.tensor import Tensor

print("Testing Vulkan Tensor Init...")
a_np = np.random.randn(10).astype(np.float32)
a_vnn = Tensor(a_np, device='vulkan')

print(f"a_np: {a_np[:3]}")
print(f"a_vnn.np_arr: {a_vnn.np_arr[:3]}")
print(f"a_vnn.to_numpy(): {a_vnn.to_numpy().flatten()[:3]}")

from vulkan_nn_lib.streaming_ops import SOE
res = SOE.elementwise_op(a_vnn, 1.0, 'add')
print(f"res.np_arr: {res.np_arr[:3]}")
print(f"res.to_numpy(): {res.to_numpy().flatten()[:3]}")
