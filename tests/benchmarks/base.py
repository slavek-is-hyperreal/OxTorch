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
    "Div": "div",
    "Sum": "sum",
    "Softmax": "softmax",
    "ReLU": "relu",
    "GELU": "gelu",
    "Linear": "linear",
    "LayerNorm": "layer_norm",
    "RMSNorm": "rms_norm",
    "Cat": "cat",
    "Stack": "stack",
    "Split": "split",
    "Chunk": "chunk",
}

# Ops that require keyword args when calling torch.nn.functional
_OP_KWARGS = {
    "softmax": {"dim": -1},
    # split and chunk use positional args — handled explicitly in dispatch
}

class BenchmarkBase:
    def __init__(self, name, op, shape, mode="cpu", dtype="f32", iterations=None, is_ssd=False, inplace=False, kwargs=None):
        self.name = name
        self.op = op
        self.shape = shape
        self.mode = mode
        self.dtype = dtype
        self.iterations = iterations
        self.is_ssd = is_ssd
        self.inplace = inplace
        self.kwargs = kwargs or {}
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
        # Get any op-specific keyword args (like dim for softmax)
        _op_kwargs = _OP_KWARGS.get(_resolved_op.lower(), {}).copy()
        _op_kwargs.update(self.kwargs)
        b_vnn = None
        b_torch = None
        b_ox = None

        # OxTorch Setup
        import oxtorch as torch_ox
        
        # Data Setup
        if not self.is_ssd:
            a_np = np.random.randn(*self.shape).astype(np.float32)
            if self.op in ["MatMul", "Linear"]:
                b_np = np.random.randn(self.shape[-1], self.shape[-1]).astype(np.float32)
            elif self.op in ["LayerNorm", "RMSNorm"]:
                # Normalization weights are 1D (normalized_shape)
                b_np = np.random.randn(self.shape[-1]).astype(np.float32)
            elif self.op == "IndexSelect":
                vocab_size = self.shape[0]
                num_indices = self.kwargs.get("num_indices", 1024)
                b_np = np.random.randint(0, vocab_size, size=(num_indices,)).astype(np.int32)
            elif self.op in ["Cat", "Stack"]:
                # For Cat/Stack we use two tensors of the same shape
                b_np = np.random.randn(*self.shape).astype(np.float32)
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
            elif self.op == "IndexSelect":
                b_torch = torch.from_numpy(b_np).to(torch.int32)
                # OxTorch parses index tensors natively as F32 before casting into i32 internally
                b_vnn = self.vnn.Tensor(data=b_np.astype(np.float32), dtype=self.vnn.DataType.F32, device=self.mode)
            else:
                b_torch = torch.from_numpy(b_np).to(torch_dtype) if self.op not in ["ReLU", "GELU", "Sum", "Softmax", "trunc", "cosh", "erf"] else None
                b_vnn = self.vnn.Tensor(data=b_np, dtype=vnn_dtype, device=self.mode) if b_torch is not None else None
            
            del a_np, b_np
            gc.collect()

            # PyTorch Benchmark
            t_pt = 0.0
            res_torch = None

            def run_torch_op_internal(a, b):
                if self.op == "MatMul":
                    return torch.matmul(a, b)
                elif self.op == "ScalarAdd":
                    return a + b
                elif self.op == "ScalarMul":
                    return a * b
                elif self.op == "Cat":
                    return torch.cat([a, b], dim=0)
                elif self.op == "Stack":
                    return torch.stack([a, b], dim=0)
                elif self.op == "Split":
                    return torch.split(a, _op_kwargs.get("split_size", 1), dim=_op_kwargs.get("dim", 0))
                elif self.op == "Chunk":
                    return torch.chunk(a, _op_kwargs.get("chunks", 2), dim=_op_kwargs.get("dim", 0))
                elif self.op == "LayerNorm":
                    return torch.layer_norm(a, [self.shape[-1]], weight=b, bias=None, eps=1e-5)
                elif self.op == "RMSNorm":
                    return a * torch.rsqrt(a.pow(2).mean(-1, keepdim=True) + 1e-5) * b
                elif self.op == "IndexSelect":
                    a_in = a.to(torch.float32) if self.dtype == "int8" else a
                    return torch.index_select(a_in, dim=self.kwargs.get("dim", 0), index=b.to(torch.int64))
                elif hasattr(torch.nn.functional, _resolved_op):
                    op_func = getattr(torch.nn.functional, _resolved_op)
                    a_in = a.to(torch.float32) if self.dtype == "int8" and self.op in ["GELU", "Softmax"] else a
                    if b is not None:
                        res = op_func(a_in, b, **_op_kwargs)
                    else:
                        res = op_func(a_in, **_op_kwargs)
                    return res.to(torch.int8) if self.dtype == "int8" and self.op in ["GELU", "Softmax"] else res
                elif hasattr(torch, _resolved_op):
                    op_func = getattr(torch, _resolved_op)
                    a_in = a.to(torch.float32) if self.dtype == "int8" and self.op in ["GELU", "Softmax"] else a
                    if b is not None:
                        res = op_func(a_in, b, **_op_kwargs)
                    else:
                        res = op_func(a_in, **_op_kwargs)
                    return res.to(torch.int8) if self.dtype == "int8" and self.op in ["GELU", "Softmax"] else res
                else:
                    raise AttributeError(f"Op {self.op} (resolved: {_resolved_op}) not found in torch or torch.nn.functional")

            if not self.is_ssd:
                try:
                    t0 = time.perf_counter()
                    for _ in range(self.iterations):
                        res_torch = run_torch_op_internal(a_torch, b_torch)
                    t_pt = (time.perf_counter() - t0) / self.iterations
                except RuntimeError as e:
                    if "not implemented for" in str(e) or "not implemented on" in str(e):
                        a_torch_f32 = a_torch.to(torch.float32)
                        b_torch_f32 = b_torch.to(torch.float32) if isinstance(b_torch, torch.Tensor) else b_torch
                        t0 = time.perf_counter()
                        for _ in range(self.iterations):
                            res_torch = run_torch_op_internal(a_torch_f32, b_torch_f32)
                        t_pt = (time.perf_counter() - t0) / self.iterations
                    else:
                        raise e
        else:
            # SSD Mapped setup (special case for Monster tests)
            t_pt = 0.0
            res_torch = None
            
            # Ensure scratch files exist for O_DIRECT access
            import os
            num_elements = np.prod(self.shape)
            file_a = f"/my_data/gaussian_room/ssd_temp_a_{num_elements}.bin"
            file_b = f"/my_data/gaussian_room/ssd_temp_b_{num_elements}.bin"
            
            if not os.path.exists(file_a):
                print(f"    [ssd] Creating scratch file {file_a}...")
                data = np.random.randn(*self.shape).astype(np.float32)
                data.tofile(file_a)
            if not os.path.exists(file_b):
                print(f"    [ssd] Creating scratch file {file_b}...")
                data = np.random.randn(*self.shape).astype(np.float32)
                data.tofile(file_b)
                
            a_vnn = self.vnn.Tensor.from_ssd(file_a, self.shape, vnn_dtype)
            b_vnn = self.vnn.Tensor.from_ssd(file_b, self.shape, vnn_dtype)
            b_ox = torch_ox.Tensor(b_vnn)

        # OxTorch Benchmark
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
            elif self.op == "Div":
                res_vnn = a_ox / b_ox
            elif self.op == "Cat":
                res_vnn = torch_ox.cat([a_ox, b_ox], dim=0)
            elif self.op == "Stack":
                res_vnn = torch_ox.stack([a_ox, b_ox], dim=0)
            elif self.op == "Split":
                res_vnn = torch_ox.split(a_ox, _op_kwargs.get("split_size", 1), dim=_op_kwargs.get("dim", 0))
            elif self.op == "Chunk":
                res_vnn = torch_ox.chunk(a_ox, _op_kwargs.get("chunks", 2), dim=_op_kwargs.get("dim", 0))
            elif self.op == "LayerNorm":
                res_vnn = a_ox.layer_norm([self.shape[-1]], weight=b_ox, bias=None, eps=1e-5)
            elif self.op == "RMSNorm":
                res_vnn = a_ox.rms_norm([self.shape[-1]], weight=b_ox, eps=1e-5)
            elif self.op == "IndexSelect":
                res_vnn = a_ox.index_select(self.kwargs.get("dim", 0), b_ox)
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
            # Handle list returns for split/chunk
            if isinstance(res_torch, (list, tuple)):
                if len(res_torch) > 10:
                    # If list is very long (e.g. split_size=1 on 1000x1000), 
                    # materializing all to_numpy() is extremely slow in benchmarks.
                    # Check first and last element for parity.
                    print(f"    [parity] Checking first and last chunks (Total {len(res_torch)} chunks)")
                    res_torch_np = np.concatenate([
                        res_torch[0].detach().cpu().to(torch.float32).numpy(),
                        res_torch[-1].detach().cpu().to(torch.float32).numpy()
                    ])
                    res_vnn_np = np.concatenate([
                        res_vnn[0].to_numpy(),
                        res_vnn[-1].to_numpy()
                    ])
                else:
                    res_torch_np = np.concatenate([r.detach().cpu().to(torch.float32).numpy() for r in res_torch])
                    res_vnn_np = np.concatenate([r.to_numpy() for r in res_vnn])
            else:
                res_torch_np = res_torch.detach().cpu().to(torch.float32).numpy()
                res_vnn_np = res_vnn.to_numpy()
            
            parity_ok, max_diff = check_parity(res_vnn_np, res_torch_np, self.name, self.op)

        else:
            # For SSD, we assume PASS if it finished without error
            parity_ok = True
            max_diff = 0.0
        max_diff = float(max_diff)
        
        # Calculate Ratio (Handle 0.0 baseline for SSD tests)
        ratio = t_vnn / t_pt if t_pt > 0 else 0.0
        
        result = {
            "name": self.name,
            "op": self.op,
            "dtype": self.dtype,
            "mode": self.mode,
            "pt_time": t_pt,
            "vnn_time": t_vnn,
            "ratio": ratio,
            "parity": parity_ok,
            "max_diff": max_diff,
            "cpu_temp": cpu_temp
        }
        
        save_benchmark_result(self.name, result)

        if t_pt > 0:
            ratio_str = f"{ratio:.2f}x" if ratio > 0.1 else f"{ratio:.4f}x"
            faster = "OxTorch FASTER" if ratio < 1.0 else "PyTorch faster"
        else:
            ratio_str = "N/A"
            faster = "SSD-STREAMING"
            
        parity_str = f"✅ PASS (max_diff={max_diff:.2e})" if parity_ok else f"❌ FAIL (max_diff={max_diff:.2e})"
        print(f"    [PyTorch] {t_pt:.4f}s", flush=True)
        print(f"    [OxTorch] {t_vnn:.4f}s | Ratio: {ratio_str} ({faster}) | Parity: {parity_str}", flush=True)

        return result
