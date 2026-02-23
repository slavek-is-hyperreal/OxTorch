import torch
import numpy as np
import vulkan_nn_lib.core as vnn

t = vnn.Tensor(np.array([-1.0, 1.0], dtype=np.float32), device='cpu')
res = t.leaky_relu(alpha=0.1)
print(res.to_numpy())
