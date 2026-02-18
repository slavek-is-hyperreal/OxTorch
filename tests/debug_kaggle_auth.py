import os
import json
import subprocess
import sys

def test_auth():
    print("--- Kaggle Auth Debug Script ---")
    json_path = "vulkan_nn_lib/kaggle.json"
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found!")
        return

    with open(json_path, 'r') as f:
        creds = json.load(f)
    
    username = creds.get('username')
    key = creds.get('key')
    
    print(f"Testing credentials for user: {username}")
    
    # Set environment variables to override any existing ~/.kaggle/kaggle.json for this test
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    
    # Try calling the CLI version from the current venv
    venv_bin = os.path.dirname(sys.executable)
    kaggle_cmd = os.path.join(venv_bin, 'kaggle')
    if not os.path.exists(kaggle_cmd):
        kaggle_cmd = "kaggle"

    print(f"Running: {kaggle_cmd} kernels list --mine")
    try:
        res = subprocess.run([kaggle_cmd, "kernels", "list", "--mine"], capture_output=True, text=True)
        if res.returncode == 0:
            print("SUCCESS: Kaggle API responded correctly.")
            print("Output snippet:", res.stdout[:100])
        else:
            print("FAILURE: Kaggle API returned error.")
            print("Stdout:", res.stdout)
            print("Stderr:", res.stderr)
    except Exception as e:
        print(f"EXCEPTION during API call: {e}")

if __name__ == "__main__":
    test_auth()
