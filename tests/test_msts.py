import pytest
import os
import numpy as np
from typing import Optional

def test_msts_io_uring():
    """
    Tests the novel Phase 3 Iron Age Optimization:
    Out-of-Core io_uring Engine with the MERA-400 CROOK OS Scheduler (MSTS).
    """
    try:
        import vulkannn_rusted as vnn
    except ImportError:
        pytest.skip("vulkannn_rusted module not found")

    # 1. Create a dummy test file
    # We will use 4MB which is 4 * 1024 * 1024 bytes = 1,048,576 f32 elements
    shape = [1024, 1024]
    test_file = "test_msts_tensor.vnn"
    
    # Create an initial matrix the regular way to save to disk via numpy memmap or direct write
    # We serialize some known values
    original_data = np.arange(1024 * 1024, dtype=np.float32).reshape(shape)
    original_data.tofile(test_file)
    
    try:
        # Load exactly the newly created file using the brand new io_uring Direct Engine!
        tensor = vnn.Tensor.from_ssd(test_file, shape, vnn.DataType.F32)
        
        # Verify it loaded as an SSD tensor using MSTS
        assert "ssd" in tensor.device.lower()
        
        # Pull it back to numpy using the MSTS chunk streaming
        streamed_array = tensor.to_numpy()
        
        assert streamed_array.shape == tuple(shape)
        np.testing.assert_array_equal(streamed_array, original_data)
        
        print("Success! io_uring and MSTS successfully streamed the 4MB tensor chunk by chunk!")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_msts_io_uring()
