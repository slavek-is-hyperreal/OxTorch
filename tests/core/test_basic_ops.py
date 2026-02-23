import torch
import vulkan_nn_lib.core as vnn
from tests.utils import to_vnn, check_close, check_grads

def test_add_backward():
    print("\n--- Testing Add Backward ---")
    a = torch.tensor([1.5, 2.0], requires_grad=True)
    b = torch.tensor([0.5, 3.0], requires_grad=True)
    c = a + b
    c.sum().backward()
    
    va = to_vnn(a, requires_grad=True)
    vb = to_vnn(b, requires_grad=True)
    vc = va + vb
    vc.backward(vnn.Tensor([1.0, 1.0], shape=(2,)))
    
    check_close(vc, c, "Sum")
    check_grads(va, a, "Grad A")
    check_grads(vb, b, "Grad B")

def test_mul_backward():
    print("\n--- Testing Mul Backward ---")
    a = torch.tensor([1.5, 2.0], requires_grad=True)
    b = torch.tensor([0.5, 3.0], requires_grad=True)
    c = a * b
    c.sum().backward()
    
    va = to_vnn(a, requires_grad=True)
    vb = to_vnn(b, requires_grad=True)
    vc = va * vb
    vc.backward(vnn.Tensor([1.0, 1.0], shape=(2,)))
    
    check_close(vc, c, "Product")
    check_grads(va, a, "Grad A")
    check_grads(vb, b, "Grad B")

def test_matmul_backward():
    print("\n--- Testing MatMul Backward ---")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([[0.5, 0.1], [0.2, 0.3]], requires_grad=True)
    c = torch.matmul(a, b)
    c.sum().backward()
    
    va = to_vnn(a, requires_grad=True)
    vb = to_vnn(b, requires_grad=True)
    vc = va @ vb
    vc.backward(vnn.Tensor([[1.0, 1.0], [1.0, 1.0]], shape=(2, 2)))
    
    check_close(vc, c, "MatMul")
    check_grads(va, a, "Grad A")
    check_grads(vb, b, "Grad B")

if __name__ == "__main__":
    test_add_backward()
    test_mul_backward()
    test_matmul_backward()
