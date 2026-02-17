import vulkan_nn_lib.torch_shim as torch
import numpy as np

def test_simple():
    print("Simple test start")
    x = torch.tensor([1, 2, 3])
    print(f"Tensor: {x}")
    print("Simple test end")

if __name__ == "__main__":
    test_simple()
