import pytest
import taichi as ti
from vulkan_nn_lib.memory_pool import get_pool, VulkanTensorPool

@pytest.fixture
def mem_pool():
    return get_pool()

def test_pool_singleton():
    pool1 = get_pool()
    pool2 = get_pool()
    assert pool1 is pool2, "Memory pool should be a singleton instance"
    assert isinstance(pool1, VulkanTensorPool)

def test_basic_allocation(mem_pool):
    shape = (100, 100)
    arr = mem_pool.allocate(shape=shape, dtype=ti.f32)
    
    assert isinstance(arr, ti.Ndarray), "Pool must return a Ti.Ndarray view"
    assert arr.shape == shape, "Allocated view must match requested shape"

def test_zero_filling(mem_pool):
    # Verify we can interact with the allocated memory via Taichi primitives
    arr = mem_pool.allocate(shape=(50,), dtype=ti.f32)
    arr.fill(3.14)
    
    np_view = arr.to_numpy()
    assert (np_view == 3.14).all(), "Data should be correctly written to the pooled array"

def test_type_handling(mem_pool):
    arr_f64 = mem_pool.allocate(shape=(10,), dtype=ti.f64)
    arr_i32 = mem_pool.allocate(shape=(10,), dtype=ti.i32)
    
    assert arr_f64.dtype == ti.f64
    assert arr_i32.dtype == ti.i32
