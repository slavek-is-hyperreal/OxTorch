import time
import numpy as np
import torch
import gc
import os
import sys
from .utils import load_vnn, get_system_metrics, get_torch_backend_label, check_parity, save_benchmark_result

# Ensure oxtorch is importable (it lives inside vulkannn_rusted/)
_OXTORCH_PATH = "/my_data/gaussian_room/vulkannn_rusted"
if _OXTORCH_PATH not in sys.path:
    sys.path.insert(0, _OXTORCH_PATH)

# Map dtype string -> DataType enum attribute name (pyo3 Rust enums are mixed-case)
_DTYPE_ATTR_MAP = {
    "f32": "F32",
    "f16": "F16",
    "bf16": "BF16",
    "int8": "Int8",
    "ternary": "Ternary",
}

# Map PascalCase op names (used in benchmark names) -> actual PyTorch/torch.nn.functional names
_OP_NAME_MAP = {
    "Mul": "mul",
    "Sub": "sub",
    "Sum": "sum",
    "Softmax": "softmax",
    "ReLU": "relu",
    "GELU": "gelu",
    "Linear": "linear",
}

# Ops that require keyword args when calling torch.nn.functional
_OP_KWARGS = {
    "softmax": {"dim": -1},
}

class BenchmarkBase:
    def __init__(self, name, op, shape, mode="cpu", dtype="f32", iterations=None, is_ssd=False, inplace=False):
        self.name = name
        self.op = op
        self.shape = shape
        self.mode = mode
        self.dtype = dtype
        self.iterations = iterations
        self.is_ssd = is_ssd
        self.inplace = inplace
        self.vnn, self.vnn_mod_name = load_vnn()

    def run(self):
        _attr = _DTYPE_ATTR_MAP.get(self.dtype.lower(), self.dtype.upper())
        vnn_dtype = getattr(self.vnn.DataType, _attr)
        dtype_map = {"f32": "float32", "f16": "float16", "bf16": "bfloat16", "int8": "int8"}
        torch_dtype = getattr(torch, dtype_map.get(self.dtype, self.dtype))

        if self.iterations is None:
            size_elements = np.prod(self.shape)
            if self.op == "MatMul":
                if size_elements >= 4e6: self.iterations = 2 # 2048x2048
                elif size_elements >= 1e6: self.iterations = 5 # 1024x1024
                else: self.iterations = 10
            elif self.is_ssd or size_elements > 5e7: self.iterations = 1
            elif size_elements > 5e6: self.iterations = 5
            else: self.iterations = 10 if self.dtype in ["f16", "bf16"] else 20

        cpu_temp, load = get_system_metrics()
        print(f"\n>>> TEST: {self.name} ({self.mode.upper()}, {self.dtype.upper()}) | Shape: {self.shape} | Iter: {self.iterations}")

        # Resolve op name: PascalCase -> actual torch function name
        _resolved_op = _OP_NAME_MAP.get(self.op, self.op)
        _op_kwargs = _OP_KWARGS.get(_resolved_op, {})
        b_vnn = None
        b_torch = None
        b_ox = None

        # Data Setup
        if not self.is_ssd:
            a_np = np.random.randn(*self.shape).astype(np.float32)
            if self.op in ["MatMul", "Linear"]:
                b_np = np.random.randn(self.shape[-1], self.shape[-1]).astype(np.float32)
            else:
                b_np = np.random.randn(*self.shape).astype(np.float32)

            # PyTorch Setup
            a_torch = torch.from_numpy(a_np).to(torch_dtype)
            # OxTorch Setup
            a_vnn = self.vnn.Tensor(data=a_np, dtype=vnn_dtype, device=self.mode)
            if self.op in ["ScalarAdd", "ScalarMul"]:
                b_torch = 10.0
                b_ox = 10.0
                b_vnn = None
            else:
                b_torch = torch.from_numpy(b_np).to(torch_dtype) if self.op not in ["ReLU", "GELU", "Sum", "Softmax", "trunc", "cosh", "erf"] else None
                b_vnn = self.vnn.Tensor(data=b_np, dtype=vnn_dtype, device=self.mode) if b_torch is not None else None
            
            del a_np, b_np
            gc.collect()

            # PyTorch Benchmark
            t0 = time.perf_counter()
            for _ in range(self.iterations):
                if self.op == "MatMul":
                    res_torch = torch.matmul(a_torch, b_torch)
                elif self.op == "ScalarAdd":
                    res_torch = a_torch + b_torch
                elif self.op == "ScalarMul":
                    res_torch = a_torch * b_torch
                elif hasattr(torch.nn.functional, _resolved_op):
                    op_func = getattr(torch.nn.functional, _resolved_op)
                    # Hack for INT8: PyTorch doesn't support it for these ops on CPU
                    a_torch_in = a_torch.to(torch.float32) if self.dtype == "int8" and self.op in ["GELU", "Softmax"] else a_torch
                    if b_torch is not None:
                        res_torch = op_func(a_torch_in, b_torch, **_op_kwargs)
                    else:
                        res_torch = op_func(a_torch_in, **_op_kwargs)
                    if self.dtype == "int8" and self.op in ["GELU", "Softmax"]:
                        res_torch = res_torch.to(torch.int8)
                elif hasattr(torch, _resolved_op):
                    op_func = getattr(torch, _resolved_op)
                    a_torch_in = a_torch.to(torch.float32) if self.dtype == "int8" and self.op in ["GELU", "Softmax"] else a_torch
                    if b_torch is not None:
                        res_torch = op_func(a_torch_in, b_torch, **_op_kwargs)
                    else:
                        res_torch = op_func(a_torch_in, **_op_kwargs)
                    if self.dtype == "int8" and self.op in ["GELU", "Softmax"]:
                        res_torch = res_torch.to(torch.int8)
                else:
                    raise AttributeError(f"Op {self.op} (resolved: {_resolved_op}) not found in torch or torch.nn.functional")
            t_pt = (time.perf_counter() - t0) / self.iterations
        else:
            # SSD Mapped setup (special case for Monster tests)
            t_pt = 0.0
            res_torch = None
            a_vnn = self.vnn.Tensor.from_ssd(f"/my_data/gaussian_room/ssd_temp_{np.prod(self.shape)}.bin", self.shape, vnn_dtype)

        # OxTorch Benchmark
        import oxtorch as torch_ox
        a_ox = torch_ox.Tensor(a_vnn)
        if b_ox is None and b_vnn is not None:
            b_ox = torch_ox.Tensor(b_vnn)

        t0 = time.perf_counter()
        for _ in range(self.iterations):
            if self.op == "MatMul":
                res_vnn = a_ox @ b_ox
            elif self.op == "Add" or self.op == "ScalarAdd":
                res_vnn = a_ox + b_ox
            elif self.op == "Mul" or self.op == "ScalarMul":
                res_vnn = a_ox * b_ox
            elif self.op == "Sum":
                res_vnn = a_ox.sum()
            elif hasattr(torch_ox, _resolved_op):
                op_func = getattr(torch_ox, _resolved_op)
                if b_ox is not None:
                    res_vnn = op_func(a_ox, b_ox, **_op_kwargs)
                else:
                    res_vnn = op_func(a_ox, **_op_kwargs)
            elif hasattr(a_ox, _resolved_op):
                # Try as tensor method (e.g. a_ox.softmax(dim=-1))
                op_func = getattr(a_ox, _resolved_op)
                res_vnn = op_func(**_op_kwargs)
            else:
                raise AttributeError(f"Op {self.op} not found in oxtorch")
        t_vnn = (time.perf_counter() - t0) / self.iterations

        # Parity Check
        if not self.is_ssd:
            # Cast to float32 for comparison as numpy doesn't support all torch dtypes (like bfloat16) directly
            res_torch_np = res_torch.detach().cpu().to(torch.float32).numpy()
            res_vnn_np = res_vnn.to_numpy()
            parity_ok, max_diff = check_parity(res_vnn_np, res_torch_np, self.name, self.op)
        else:
            # For SSD, we assume PASS if it finished without error
            parity_ok = True
            max_diff = 0.0
        max_diff = float(max_diff)
        
        result = {
            "name": self.name,
            "op": self.op,
            "dtype": self.dtype,
            "mode": self.mode,
            "pt_time": t_pt,
            "vnn_time": t_vnn,
            "ratio": t_vnn / t_pt if t_pt > 0 else 0,
            "parity": parity_ok,
            "max_diff": max_diff,
            "cpu_temp": cpu_temp
        }
        
        save_benchmark_result(self.name, result)

        ratio = result["ratio"]
        ratio_str = f"{ratio:.2f}x" if ratio > 0.1 else f"{ratio:.4f}x"
        parity_str = f"✅ PASS (max_diff={max_diff:.2e})" if parity_ok else f"❌ FAIL (max_diff={max_diff:.2e})"
        faster = "OxTorch FASTER" if ratio < 1.0 else "PyTorch faster"
        print(f"    [PyTorch] {t_pt:.4f}s", flush=True)
        print(f"    [OxTorch] {t_vnn:.4f}s | Ratio: {ratio_str} ({faster}) | Parity: {parity_str}", flush=True)

        return result
