import os
from dotenv import load_dotenv

# Load .env file from the library's directory
_lib_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_lib_dir, ".env")
load_dotenv(_env_path)

def get_ssd_path():
    """
    Returns the SSD cache path.
    Prioritizes:
    1. Environment variable VNN_CACHE_DIR
    2. Default universal path ./vnn_cache
    """
    return os.getenv("VNN_CACHE_DIR", "./vnn_cache")
