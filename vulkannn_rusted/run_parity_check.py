from tests.parity.bridge import run_pytorch_test_class
from tests.parity.mock_pytorch_test import TestOxTorchParity
import sys
import os

# Ensure we pick up the local oxtorch
sys.path.append(os.getcwd())

if __name__ == "__main__":
    print("Starting OxTorch Extreme Parity Check...")
    result = run_pytorch_test_class(TestOxTorchParity)
    
    if result.wasSuccessful():
        print("\n" + "="*50)
        print("✨ ALL PARITY TESTS PASSED via OxTorch Bridge!")
        print("Functional Parity: 100%")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ SOME PARITY TESTS FAILED.")
        print("Please check the mismatch logs above.")
        print("="*50)
        sys.exit(1)
