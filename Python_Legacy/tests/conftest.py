import pytest
import os
import shutil

# Tolerance settings for float32 precision
ATOL = 1e-4
RTOL = 1e-4

@pytest.fixture(autouse=True)
def init_vnn():
    """Ensure Taichi is initialized before any tests run and clean up cache."""
    import taichi as ti
    from Python_Legacy.vulkan_nn_lib import config
    
    # Optional: Clear global SSD cache between test sessions to avoid disk filling
    test_cache = "./temp_test_cache"
    os.environ["VNN_CACHE_DIR"] = test_cache
    
    ti.init(arch=ti.vulkan, offline_cache=True, device_memory_GB=1.0)
    
    yield
    
    # Teardown
    if os.path.exists(test_cache):
        shutil.rmtree(test_cache, ignore_errors=True)
