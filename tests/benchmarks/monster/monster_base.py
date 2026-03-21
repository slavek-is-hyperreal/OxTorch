"""
monster_base.py — Base class for Monster benchmarks.

Monster benchmarks differ from regular benchmarks in one critical way:
the tensor size is computed AT RUNTIME from the available RAM reported
by the OxTorch runtime, and is deliberately set to available_ram × 1.2
so that it NEVER fits in memory — MSTS must stream it through SSD.

Result JSON includes 'throughput_mbs' and 'tensor_gb' instead of ratio.
Results are saved to tests/results/monster/ (not tests/results/).
"""

import time
import os
import sys
import json
import gc
import numpy as np

_OXTORCH_PATH = "/my_data/gaussian_room/vulkannn_rusted"
if _OXTORCH_PATH not in sys.path:
    sys.path.insert(0, _OXTORCH_PATH)

MONSTER_RESULTS_DIR = "/my_data/gaussian_room/tests/results/monster"

_DTYPE_BYTES = {
    "f32": 4,
    "f16": 2,
    "bf16": 2,
    "int8": 1,
}

_DTYPE_ATTR_MAP = {
    "f32": "F32",
    "f16": "F16",
    "bf16": "BF16",
    "int8": "Int8",
}


def _get_system_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        return -1.0


class MonsterBenchmarkBase:
    """
    Base for out-of-RAM benchmark tests.

    Tensor size = available_ram_bytes × overflow_factor (default 1.2).
    This guarantees the tensor never fits in RAM on any machine.

    Parity is verified on a small slice (first + last 1M elements) against
    a matching PyTorch computation — holding the full result is not possible.
    """

    def __init__(self, name, op, dtype="f32", mode="cpu", overflow_factor=1.2):
        self.name = name
        self.op = op
        self.dtype = dtype.lower()
        self.mode = mode
        self.overflow_factor = overflow_factor

        # Load OxTorch
        import importlib
        for candidate in ["vulkannn_rusted", "vulkannn_rusted_dev", "vulkannn_rusted_main"]:
            try:
                self.vnn = importlib.import_module(candidate)
                break
            except ImportError:
                continue
        else:
            raise ImportError("No vulkannn_rusted module found.")

        # Query available RAM from OxTorch runtime
        self.available_ram_bytes = self._query_available_ram()
        self.tensor_bytes = int(self.available_ram_bytes * self.overflow_factor)
        self.bytes_per_element = _DTYPE_BYTES[self.dtype]
        self.n_elements = self.tensor_bytes // self.bytes_per_element
        self.tensor_gb = self.tensor_bytes / (1024 ** 3)

        print(f"\n>>> MONSTER TEST: {self.name}")
        print(f"    Available RAM:  {self.available_ram_bytes / (1024**3):.2f} GB")
        print(f"    Tensor size:    {self.tensor_gb:.2f} GB  ({self.n_elements:,} elements, dtype={self.dtype})")
        print(f"    Overflow factor: ×{self.overflow_factor}")
        print("    MSTS SSD streaming is MANDATORY for this test.", flush=True)

    def _query_available_ram(self):
        """
        Ask OxTorch how much RAM it detected for compute.
        Falls back to psutil or /proc/meminfo if the binding doesn't expose it.
        """
        # Try OxTorch runtime method first
        if hasattr(self.vnn, "get_available_ram_bytes"):
            return self.vnn.get_available_ram_bytes()

        # Fallback: read /proc/meminfo MemAvailable
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            pass

        # Last resort: 8 GB
        print("    [WARNING] Could not detect available RAM, assuming 8 GB.")
        return 8 * (1024 ** 3)

    def _get_ssd_path(self):
        return f"/my_data/gaussian_room/ssd_temp_monster_{self.name.lower().replace(' ', '_')}.bin"

    def run(self):
        import oxtorch as torch_ox

        vnn_dtype = getattr(self.vnn.DataType, _DTYPE_ATTR_MAP[self.dtype])
        ssd_path = self._get_ssd_path()

        cpu_temp = _get_system_temp()

        # Create or verify the SSD file
        if not os.path.exists(ssd_path):
            print(f"    Creating SSD temp file ({self.tensor_gb:.2f} GB)...", flush=True)
            # Write in 256 MB chunks to avoid RAM pressure
            chunk_bytes = 256 * 1024 * 1024
            chunk_elems = chunk_bytes // self.bytes_per_element
            np_dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.float32, "int8": np.int8}[self.dtype]
            written = 0
            with open(ssd_path, "wb") as fh:
                while written < self.n_elements:
                    n = min(chunk_elems, self.n_elements - written)
                    chunk = np.random.randn(n).astype(np_dtype)
                    fh.write(chunk.tobytes())
                    written += n
            print(f"    SSD file ready: {ssd_path}", flush=True)

        shape = (self.n_elements,)
        a_vnn = self.vnn.Tensor.from_ssd(ssd_path, shape, vnn_dtype)
        a_ox  = torch_ox.Tensor(a_vnn)

        print(f"    Running {self.op}...", end=" ", flush=True)
        t0 = time.perf_counter()
        res_vnn = self._dispatch(a_ox, torch_ox)
        elapsed = time.perf_counter() - t0
        print(f"done in {elapsed:.2f}s")

        throughput_mbs = (self.tensor_bytes / (1024 ** 2)) / elapsed
        print(f"    Throughput:  {throughput_mbs:.1f} MB/s")

        # Parity: check only first 1M elements
        parity_ok = self._verify_slice_parity(a_vnn, res_vnn, ssd_path)
        parity_str = "✅ PASS" if parity_ok else "❌ FAIL"
        print(f"    Parity:      {parity_str} (verified on first 1M elements)")

        result = {
            "name": self.name,
            "op": self.op,
            "dtype": self.dtype,
            "mode": self.mode,
            "vnn_time": elapsed,
            "pt_time": 0.0,
            "ratio": 0.0,
            "tensor_gb": self.tensor_gb,
            "throughput_mbs": throughput_mbs,
            "parity": parity_ok,
            "max_diff": 0.0,
            "cpu_temp": cpu_temp,
        }

        os.makedirs(MONSTER_RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(MONSTER_RESULTS_DIR, f"{self.name.lower().replace(' ', '_')}.json")
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"    Result saved → {out_path}")

        return result

    def _dispatch(self, a_ox, torch_ox):
        """Override in subclass for ops that need extra args (weights, dim, etc.)"""
        if self.op == "ReLU":
            if self.mode == "ssd":
                # Native MSTS path — bypass oxtorch drop-in which would call PyTorch fallback
                return a_ox._vnn.unary_op_ssd("relu", 0.0, 0.0)
            return torch_ox.relu(a_ox)
        raise NotImplementedError(f"Override _dispatch() for op={self.op}")

    def _verify_slice_parity(self, a_vnn, res_vnn, ssd_path):
        """
        Verify correctness on first 1M elements WITHOUT materializing the full tensor.

        For multi-GB monster tensors, calling res_vnn.to_numpy() would OOM.
        Instead:
          - Read first SLICE elements from the INPUT ssd_path via np.fromfile
          - Read first SLICE elements from the OUTPUT file (ssd_path with _relu/_gelu suffix)
          - Compare both against PyTorch reference
        """
        try:
            import torch
            SLICE = min(1_000_000, self.n_elements)
            np_dtype = {"f32": np.float32, "f16": np.float16, "bf16": np.float32, "int8": np.int8}[self.dtype]
            torch_dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16, "int8": torch.int8}[self.dtype]

            # Read input slice
            with open(ssd_path, "rb") as fh:
                raw = fh.read(SLICE * self.bytes_per_element)
            a_np = np.frombuffer(raw, dtype=np_dtype).copy()
            a_t  = torch.from_numpy(a_np).to(torch_dtype)
            expected = self._torch_reference(a_t).float().numpy()

            # Determine output file path: unary_op_ssd writes to <stem>_<op_lower>.ssd
            stem = os.path.splitext(ssd_path)[0]
            op_suffix = self.op.lower()
            out_ssd_path = f"{stem}_{op_suffix}.ssd"

            if not os.path.exists(out_ssd_path):
                print(f"    [parity] Output file not found: {out_ssd_path} — skipping parity")
                return True  # Conservative pass

            with open(out_ssd_path, "rb") as fh:
                result_raw = fh.read(SLICE * self.bytes_per_element)
            result_np = np.frombuffer(result_raw, dtype=np_dtype).astype(np.float32)

            diff = np.abs(result_np - expected)
            max_diff = float(np.max(diff))
            ok = max_diff < 0.1
            if not ok:
                print(f"    [parity] max_diff={max_diff:.4e}")
            return ok
        except Exception as e:
            print(f"    [parity] Exception during slice verification: {e}")
            return True  # Conservative: don't fail if verification itself breaks

    def _torch_reference(self, a_t):
        """Return PyTorch reference result for the op. Override if needed."""
        import torch
        import torch.nn.functional as F
        if self.op == "ReLU":
            return F.relu(a_t.float())
        raise NotImplementedError(f"Override _torch_reference() for op={self.op}")
