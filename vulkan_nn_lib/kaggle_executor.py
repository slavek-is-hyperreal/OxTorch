import os
import json
import time
import subprocess
import tempfile
import numpy as np
import shutil

def _safe_save_npy(path, arr):
    """Saves a numpy array to disk chunk-by-chunk to prevent OOM on huge memmaps."""
    if arr.nbytes < 1024 * 1024 * 512:
        np.save(path, arr) # Standard save for < 512MB
        return
        
    print(f"  [Kaggle] Safely streaming {arr.nbytes/1e9:.2f}GB to {path}...")
    import numpy.lib.format as fmt
    with open(path, 'wb') as f:
        fmt.write_array_header_2_0(f, fmt.header_data_from_array_1_0(arr))
        
        # Stream in 128MB chunks
        chunk_bytes = 128 * 1024 * 1024
        item_size = arr.itemsize
        chunk_len = chunk_bytes // item_size
        
        flat_arr = arr.reshape(-1) # reshape doesn't copy on memmap
        for i in range(0, flat_arr.size, chunk_len):
            end = min(i + chunk_len, flat_arr.size)
            chunk = flat_arr[i:end].copy() # Small copy into RAM
            f.write(chunk.tobytes())

class KaggleExecutor:
    """
    Orchestrates remote computation on Kaggle.
    Reuses VNN's tiling logic to partition workloads and offload them to Kaggle GPU kernels.
    """
    
    KAGGLE_RAM_LIMIT_GB = 13.0 # Standard Kaggle GPU kernel limit
    
    def __init__(self):
        # Adjusting search for kaggle.json based on discovery
        self.kaggle_json_path = os.path.join(os.path.dirname(__file__), "kaggle.json")
        self.username = None
        self._ensure_kaggle_auth()

    def _get_kaggle_cmd(self):
        """Finds the kaggle CLI path, prioritizing the virtual environment."""
        import sys
        venv_bin = os.path.dirname(sys.executable)
        venv_kaggle = os.path.join(venv_bin, 'kaggle')
        if os.path.exists(venv_kaggle):
            return venv_kaggle
        return "kaggle"

    def _ensure_kaggle_auth(self):
        """Ensures kaggle.json is in the expected location for the Kaggle CLI."""
        target_dir = os.path.expanduser("~/.kaggle")
        target_path = os.path.join(target_dir, "kaggle.json")
        
        # Look for it in vulkan_nn_lib or root
        src = self.kaggle_json_path
        if not os.path.exists(src):
            src = "kaggle.json" # try root
            
        if os.path.exists(src):
            # Always copy if source exists to ensure latest credentials are used
            os.makedirs(target_dir, exist_ok=True)
            import shutil
            shutil.copy(src, target_path)
            os.chmod(target_path, 0o600)
            print(f"  [Kaggle] Credentials synchronized to {target_path}")
            
            # Extract username and key
            try:
                with open(src, 'r') as f:
                    creds = json.load(f)
                    self.username = creds.get('username')
                    if creds.get('key'):
                        os.environ['KAGGLE_USERNAME'] = self.username
                        os.environ['KAGGLE_KEY'] = creds.get('key')
                        print(f"  [Kaggle] Environment variables KAGGLE_USERNAME and KAGGLE_KEY set.")
            except: pass
        elif os.path.exists(target_path):
            # Extract from existing target
            try:
                with open(target_path, 'r') as f:
                    creds = json.load(f)
                    self.username = creds.get('username')
                    if creds.get('key'):
                        os.environ['KAGGLE_USERNAME'] = self.username
                        os.environ['KAGGLE_KEY'] = creds.get('key')
            except: pass
            
        if self.username is None:
            # Fallback to env var
            from .config import get_kaggle_user
            self.username = get_kaggle_user()
            
        if self.username == 'vnn_user':
             print("Warning: Using default Kaggle username 'vnn_user'. Ensure KAGGLE_USERNAME or kaggle.json is set.")

    def _ensure_dataset(self, dataset_slug, data_dir):
        """Creates or updates a Kaggle dataset with the data in data_dir and waits for readiness."""
        full_slug = f"{self.username}/{dataset_slug}"
        
        # Check if dataset exists
        check_proc = subprocess.run([self._get_kaggle_cmd(), "datasets", "status", full_slug], capture_output=True, text=True)
        
        try:
            if "not found" in check_proc.stdout.lower() or check_proc.returncode != 0:
                # Dataset does not exist, create it
                metadata = {"title": dataset_slug, "id": full_slug, "licenses": [{"name": "CC0-1.0"}]}
                with open(os.path.join(data_dir, 'dataset-metadata.json'), 'w') as f:
                    json.dump(metadata, f)
                subprocess.run([self._get_kaggle_cmd(), "datasets", "create", "-p", data_dir, "-u"], check=True)
            else:
                # Dataset exists, try to update it
                subprocess.run([self._get_kaggle_cmd(), "datasets", "version", "-p", data_dir, "-m", "New data"], check=True)
            
            # Wait for dataset to be "ready"
            print(f"  [Kaggle] Waiting for dataset {dataset_slug} to be ready...")
            for _ in range(30): # max 5 mins
                res = subprocess.run([self._get_kaggle_cmd(), "datasets", "status", full_slug], capture_output=True, text=True)
                if "ready" in res.stdout.lower():
                    print(f"  [Kaggle] Dataset {dataset_slug} is ready.")
                    return
                time.sleep(10)
            print(f"  [Kaggle] Warning: Dataset {dataset_slug} still not ready after timeout.")
        except Exception as e:
            print(f"Warning: Dataset creation/update failed: {e}")

    def _generate_execution_script(self, op_type, a_path, b_path=None, extra=None):
        """Generates a standalone Python script to be executed on Kaggle."""
        script = f"""
import numpy as np
import torch
import os

def run_op():
    import os
    import sys
    print("--- Kaggle Environment Debug ---")
    print("Python version:", sys.version)
    print("Working directory:", os.getcwd())
    
    def find_file(name):
        print(f"Searching for {{name}}...")
        for root, dirs, files in os.walk('/kaggle'):
            if name in files:
                p = os.path.join(root, name)
                print(f"FOUND: {{p}}")
                return p
        return None

    # Thoroughly list what's in /kaggle/input
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
             print(f"Path: {{root}}, Dirs: {{dirs}}, Files: {{files}}")
    else:
        print("/kaggle/input does not exist!")
    
    a_f = find_file('input_a.npy')
    b_f = find_file('input_b.npy')
    
    if not a_f:
        print("FAIL: could not find input_a.npy")
        # Final emergency list
        print("Full /kaggle structure search complete.")
        raise FileNotFoundError("Could not find input_a.npy in /kaggle/")
        
    a = np.load(a_f)
    if b_f:
        b = np.load(b_f)
    else:
        # Check if extra is a constant or if we expect input_b.npy
        b_path_str = '{b_path}'
        if b_path_str != 'None' and not b_f:
             print(f"FAIL: expected input_b.npy but not found at {{b_path_str}}")
             raise FileNotFoundError(f"Expected input_b.npy but not found")
        b = {extra}
        
    a_t = torch.from_numpy(a).cuda()
    b_t = torch.from_numpy(b).cuda() if isinstance(b, np.ndarray) else b
    
    op = '{op_type}'
    if op == 'add': r_t = a_t + b_t
    elif op == 'sub': r_t = a_t - b_t
    elif op == 'mul': r_t = a_t * b_t
    elif op == 'div': r_t = a_t / b_t
    elif op == 'matmul': r_t = torch.matmul(a_t, b_t)
    elif op == 'sum': r_t = torch.sum(a_t)
    elif op == 'add_sum': r_t = torch.sum(a_t + b_t)
    elif op == 'mul_sum': r_t = torch.sum(a_t * b_t)
    else: r_t = a_t
    
    res = r_t.cpu().numpy()
    np.save('output.npy', res)
    print("Workload complete.")

if __name__ == "__main__":
    run_op()
"""
        return script

    def submit_operation(self, op_type, a_tensor, b_tensor=None, extra=None):
        from .tensor import Tensor
        n = a_tensor.total_size
        item_size = a_tensor.item_size
        
        # Partitioning logic depends on the operation
        is_reduction = 'sum' in op_type or 'mean' in op_type
        is_matmul = op_type == 'matmul'
        
        if is_matmul:
            # For matmul, we send the whole B matrix and partition A by rows
            # This is safe as long as B fits in 13GB (user's 12GB case)
            M, K_dim = a_tensor.shape
            K2, N = b_tensor.shape
            
            # Estimate how many rows of A we can process per super-tile
            # Each row of A is K_dim elements. Each row of result is N elements.
            # Total RAM used: B (K_dim*N) + A_part (rows*K_dim) + Res_part (rows*N)
            row_size_bytes = (K_dim + N) * item_size
            b_size_bytes = (K_dim * N) * item_size
            available_ram_bytes = (self.KAGGLE_RAM_LIMIT_GB * 0.7 * 1024**3) - b_size_bytes
            
            super_tile_rows = int(available_ram_bytes // row_size_bytes)
            super_tile_rows = max(1, min(M, super_tile_rows))
            
            res = Tensor(None, shape=(M, N), device='ssd', dtype=a_tensor.dtype)
            for row_start in range(0, M, super_tile_rows):
                row_end = min(row_start + super_tile_rows, M)
                
                # We need to slice A by rows safely without loading the whole matrix 
                if hasattr(a_tensor, 'arr') and hasattr(a_tensor.arr, 'shape'):
                    a_part = a_tensor.arr[row_start:row_end, :].copy()
                else:
                    a_part = a_tensor.to_numpy()[row_start:row_end, :]
                
                slug = f"vnn-matmul-{int(time.time()*100)}"
                ds_slug = f"vnn-data-{int(time.time()*100)}"
                
                with tempfile.TemporaryDirectory(dir='.') as tmpdir:
                    _safe_save_npy(os.path.join(tmpdir, 'input_a.npy'), a_part)
                    
                    b_view = b_tensor.arr if hasattr(b_tensor, 'arr') else b_tensor.to_numpy()
                    _safe_save_npy(os.path.join(tmpdir, 'input_b.npy'), b_view)
                    
                    self._ensure_dataset(ds_slug, tmpdir)
                    script = self._generate_execution_script(op_type, 'input_a.npy', 'input_b.npy', extra)
                    res_part = self._push_and_wait_with_dataset(slug, script, ds_slug)
                    res.arr.reshape(M, N)[row_start:row_end, :] = res_part
            return res

        # Super-tile size (approx 10GB for elementwise)
        super_tile_len = int((self.KAGGLE_RAM_LIMIT_GB * 0.75 * 1024**3) // item_size)
        
        if is_reduction:
            final_val = 0.0
            for start in range(0, n, super_tile_len):
                end = min(start + super_tile_len, n)
                res_part = self._run_remote_chunk(op_type, a_tensor, start, end, b_tensor, extra)
                final_val += float(res_part)
            return Tensor([final_val], shape=(), device='cpu')
        
        res_dtype = a_tensor.dtype if a_tensor.dtype != 'int4' else np.float32
        res = Tensor(None, shape=a_tensor.shape, device='ssd', dtype=res_dtype)
        for start in range(0, n, super_tile_len):
            end = min(start + super_tile_len, n)
            res_part = self._run_remote_chunk(op_type, a_tensor, start, end, b_tensor, extra)
            res.arr[start:end] = res_part.flatten()
        return res

    def _run_remote_chunk(self, op_type, a_tensor, start, end, b_tensor, extra):
        slug = f"vnn-op-{int(time.time()*100)}"
        ds_slug = f"vnn-data-{int(time.time()*100)}"
        
        with tempfile.TemporaryDirectory(dir='.') as tmpdir:
            # Use safe views instead of to_numpy().flatten() to prevent 20GB+ OOM
            if hasattr(a_tensor, 'arr') and hasattr(a_tensor.arr, 'reshape'):
                a_data = a_tensor.arr.reshape(-1)[start:end]
            else:
                a_data = a_tensor.to_numpy().reshape(-1)[start:end]
                
            _safe_save_npy(os.path.join(tmpdir, 'input_a.npy'), a_data)
            b_path = None
            if b_tensor is not None:
                if hasattr(b_tensor, 'arr'):
                    # Handle if B should be sliced or is a constant matrix
                    if b_tensor.total_size == a_tensor.total_size:
                        if hasattr(b_tensor.arr, 'reshape'):
                            b_data = b_tensor.arr.reshape(-1)[start:end]
                        else:
                            b_data = b_tensor.to_numpy().reshape(-1)[start:end]
                        _safe_save_npy(os.path.join(tmpdir, 'input_b.npy'), b_data)
                    else:
                        b_view = b_tensor.arr if hasattr(b_tensor, 'arr') else b_tensor.to_numpy()
                        _safe_save_npy(os.path.join(tmpdir, 'input_b.npy'), b_view)
                    b_path = 'input_b.npy'

            self._ensure_dataset(ds_slug, tmpdir)
            script = self._generate_execution_script(op_type, 'input_a.npy', b_path, extra)
            
            # Submitting kernel
            return self._push_and_wait_with_dataset(slug, script, ds_slug)

    def _push_and_wait_with_dataset(self, slug, script_content, ds_slug):
        username = self.username
        with tempfile.TemporaryDirectory(dir='.') as tmpdir:
            script_file = os.path.join(tmpdir, 'script.py')
            
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            metadata = {
                "id": f"{username}/{slug}",
                "title": slug,
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_internet": "false",
                "dataset_sources": [f"{username}/{ds_slug}"],
            }
            with open(os.path.join(tmpdir, 'kernel-metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            subprocess.run([self._get_kaggle_cmd(), "kernels", "push", "-p", tmpdir], check=True)
            
            # 3. Poll for completion
            print(f"  [Kaggle] Kernel pushed. Waiting for results ({slug})...")
            while True:
                res = subprocess.run([self._get_kaggle_cmd(), "kernels", "status", f"{username}/{slug}"], capture_output=True, text=True)
                status = res.stdout
                if "complete" in status.lower(): break
                if "error" in status.lower():
                    raise RuntimeError(f"Kaggle kernel failed: {status}")
                time.sleep(15)

            # 4. Download results
            subprocess.run([self._get_kaggle_cmd(), "kernels", "output", f"{username}/{slug}", "-p", tmpdir], check=True)
            return np.load(os.path.join(tmpdir, 'output.npy'))
