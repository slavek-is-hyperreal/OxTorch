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

def get_kaggle_enabled():
    """Returns True if VNN_KAGGLE_MODE is set to '1'."""
    return os.getenv("VNN_KAGGLE_MODE", "0") == "1"

def get_kaggle_user():
    """Returns the Kaggle username from environment or defaults to 'vnn_user'."""
    return os.getenv("KAGGLE_USERNAME", "vnn_user")

def get_kaggle_threshold():
    """Returns the size threshold in bytes to trigger Kaggle offload."""
    return int(os.getenv("VNN_KAGGLE_THRESHOLD", "1000000000")) # Default 1GB
