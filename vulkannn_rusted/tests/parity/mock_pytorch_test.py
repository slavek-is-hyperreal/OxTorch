import torch
import unittest

class TestOxTorchParity(unittest.TestCase):
    """
    A simulated PyTorch test case. 
    When run via the bridge, 'torch' will be 'oxtorch'.
    """
    def test_basic_arithmetic(self):
        a = torch.ones([2, 2])
        b = torch.ones([2, 2])
        c = a + b
        self.assertEqual(c.sum().item(), 8.0)

    def test_fallback_op(self):
        # det() currently falls back to real PyTorch via our proxy
        a = torch.as_tensor([[1.0, 2.0], [3.0, 4.0]])
        d = a.det()
        # d.item() should work thanks to our recent add
        self.assertAlmostEqual(d.item(), -2.0)

    def test_factory_randn(self):
        a = torch.randn(10, 10)
        self.assertEqual(list(a.shape), [10, 10])

    def test_indexing_fallback(self):
        a = torch.ones([5, 5])
        # __getitem__ is not explicitly implemented in OxTorchTensor yet, 
        # so it should currently fail OR we should add it.
        # Let's add it to OxTorchTensor before running.
        slice_a = a[0:2, 0:2]
        self.assertEqual(slice_a.sum().item(), 4.0)
