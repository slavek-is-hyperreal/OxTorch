import pytest
import os

# Tolerance settings for float32 precision
ATOL = 1e-4
RTOL = 1e-4

@pytest.fixture(autouse=True)
def init_env():
    """Basic environment setup for Rust tests."""
    yield
