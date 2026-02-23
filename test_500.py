import vulkan_nn_lib.torch_shim as torch
import numpy as np
import taichi as ti

# Fill with 2.0 and 3.0 via numpy for easy math testing: 2.0 * 3.0 * 500 = 3000.0
a = torch.Tensor.from_numpy(np.ones((500, 500), dtype=np.float32) * 2.0).to('vulkan')
b = torch.Tensor.from_numpy(np.ones((500, 500), dtype=np.float32) * 3.0).to('vulkan')

c = a @ b
ti.sync()

np_res = c.to_numpy()
print(f"Shape: {np_res.shape}")
print(f"Sample: {np_res[0, :5]}")

if np.allclose(np_res, 3000.0, atol=1e-4):
    print("500x500 MatMul SUCCESS: Math is 100% correct!")
else:
    print("FAILED: Math mismatch.")
