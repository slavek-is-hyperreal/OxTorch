import sys
import os

# Add current directory to path to pick up the new 'oxtorch' package
sys.path.append(os.getcwd())

import oxtorch
import numpy as np

print(f"Testing OxTorch v{oxtorch.__version__} Fallback Dispatcher...")

# 1. Test Native Construction
print("\n[1] Testing Native Construction...")
a = oxtorch.zeros(2, 2)
print(f"Native Zeros: {a}")
assert list(a.shape) == [2, 2]

# 2. Test Fallback Factory (randn is not native in VNN yet)
print("\n[2] Testing Fallback Factory (randn)...")
b = oxtorch.randn(2, 2)
print(f"Fallback Randn: {b}")
assert list(b.shape) == [2, 2]

# 3. Test Native Element-wise
print("\n[3] Testing Native Op (add)...")
c = a + b
print(f"Result (a+b): {c}")

# 4. Test Fallback Method (exp() on tensor)
print("\n[4] Testing Fallback Method (exp)...")
# Note: VNN has exp? Let's check. If it has, use something else like sin() or log10()
d = b.exp()
print(f"Fallback Exp: {d}")

# 5. Test Fallback for advanced op (e.g. svd or det)
print("\n[5] Testing Advanced Fallback (det)...")
m = oxtorch.Tensor([[1.0, 2.0], [3.0, 4.0]])
determinant = m.det()
print(f"Determinant: {determinant}")
# det returns a scalar (0-dim tensor or float)
assert np.isclose(determinant, -2.0)

print("\n✅ All fallback tests PASSED!")
