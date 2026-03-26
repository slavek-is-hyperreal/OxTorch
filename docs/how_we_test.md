# How We Test — OxTorch v3.7.0

This document describes the complete testing and benchmarking infrastructure for OxTorch.

---

## Overview

OxTorch uses an **Atomized Benchmark Suite** — 144+ individual, self-contained benchmark scripts, each covering exactly one operation / dtype / backend combination. The suite replaced all monolithic benchmark files in v3.7.0.

```
tests/
├── benchmarks/               ← Atomized suite (144+ individual tests)
│   ├── base.py               ← BenchmarkBase class (shared harness)
│   ├── utils.py              ← load_vnn(), check_parity(), save_benchmark_result()
│   ├── f16/                  ← F16 dtype benchmarks
│   │   ├── matmul_cpu.py
│   │   ├── matmul_vulkan.py
│   │   ├── matmul_hybrid.py
│   │   ├── relu_cpu.py
│   │   └── ...
│   ├── bf16/                 ← BF16 dtype benchmarks
│   ├── f32/                  ← F32 dtype benchmarks
│   ├── int8/                 ← INT8 dtype benchmarks
│   ├── ssd/                  ← MSTS 3-path SSD dispatch benchmarks (v3.7.1+)
│   │   ├── msts_path_a_relu_f32.py  ← Path A: Direct (<3 MB, zero threads)
│   │   ├── msts_path_b_relu_f32.py  ← Path B: Single-thread (<32 MB, L2-resident)
│   │   └── msts_path_c_relu_f32.py  ← Path C: Full CrookScheduler (>=32 MB, rayon)
│   └── monster/              ← SSD out-of-core benchmarks (16GB+)
│       └── relu_ssd_f32.py
├── results/                  ← Per-benchmark JSON output (gitignored)
│   └── matmul_f16_vulkan.json
└── run_all_benchmarks.py     ← Orchestrator: runs all, collects summary
```

---

## Running the Benchmarks

### Run all benchmarks
```bash
source venv/bin/activate
PYTHONPATH=/path/to/vulkannn_rusted python tests/run_all_benchmarks.py
```

### Run a single benchmark
```bash
PYTHONPATH=/path/to/vulkannn_rusted python tests/benchmarks/f16/matmul_vulkan.py
```

### Run with timeout override
Edit `BENCHMARK_TIMEOUT_S` in `tests/run_all_benchmarks.py` (currently 600s).

---

## The `BenchmarkBase` Class (`tests/benchmarks/base.py`)

Every atomized test extends `BenchmarkBase`. Example from `matmul_f16_vulkan.py`:

```python
from benchmarks.base import BenchmarkBase

class MatMulF16Vulkan(BenchmarkBase):
    def __init__(self):
        super().__init__(
            name="MatMul_f16_vulkan",
            op="MatMul",
            shape=(2048, 2048),
            mode="vulkan",
            dtype="f16",
        )

if __name__ == "__main__":
    MatMulF16Vulkan().run()
```

### What `BenchmarkBase.run()` does

1. **Setup**: Allocates numpy arrays, converts to the target dtype for both PyTorch and OxTorch
2. **PyTorch Benchmark**: Runs `N` iterations, measures wall time with `perf_counter`
3. **OxTorch Benchmark**: Same — runs via `import oxtorch as torch`
4. **Parity Check**: Converts both results to float32 numpy and runs `np.testing.assert_allclose`
5. **Saves result** to `tests/results/<name>.json`
6. **Prints live output** in the format:
```
>>> TEST: MatMul_f16_vulkan (VULKAN, F16) | Shape: (2048, 2048) | Iter: 2
    [PyTorch] 4.6100s
    [OxTorch] 0.1999s | Ratio: 0.04x (OxTorch FASTER) | Parity: ✅ PASS (max_diff=2.50e-01)
[benchmark] Result saved to tests/results/matmul_f16_vulkan.json
```

---

## Iteration Count Heuristic

`BenchmarkBase` auto-selects iteration count based on shape/op:

| Condition | Iterations |
|:---|:---|
| MatMul, ≥ 4M elements | 2 |
| MatMul, ≥ 1M elements | 5 |
| SSD or > 50M elements | 1 |
| F16/BF16 other | 10 |
| F32/INT8 other | 20 |

---

## Parity Tolerances (`tests/benchmarks/utils.py`)

Each dtype has different numerical tolerances for `np.testing.assert_allclose`:

| DType | `atol` | Reason |
|:---|:---|:---|
| F32 | `1e-4` | IEEE 754 exact |
| F16 | `0.25` | 10-bit mantissa; MatMul accumulation error |
| BF16 | `1.0` | 7-bit mantissa; large accumulated error on 2k MatMul |
| INT8 | `2.0` | Integer quantization rounding |
| INT8 Sum | `5000.0` | PyTorch sums as float32 (lossy); OxTorch sums as i64 (exact) |
| INT8 MatMul | `300.0` | PyTorch computes in float32 then clamps; paths differ |

Special cases:
- **INT8 GELU / INT8 Softmax**: PyTorch has **no native kernel** for these on CPU. OxTorch computes them natively. Parity is checked against a float32 reference instead.
- **SSD (Monster) tests**: Parity is assumed ✅ PASS if the op completes without error (full materialization would OOM).
- **F16 MatMul Hybrid**: PyTorch takes ~150s per iteration on non-F16C CPUs. Benchmark timeout is 600s.

---

## The Orchestrator (`tests/run_all_benchmarks.py`)

`run_all_benchmarks.py` discovers all benchmark scripts by walking `tests/benchmarks/`, spawns each as a subprocess with a 600s timeout, collects stdout/stderr, and prints a summary table:

```
===================================================================================================================
                                      OxTorch ATOMIZED BENCHMARK SUMMARY
===================================================================================================================
Test Case                           | PT Time    | VNN Time   | Ratio    | Baseline   | CPU°C  | Status
-------------------------------------------------------------------------------------------------------------------
MatMul_f16_vulkan                   |  148.1644s |    0.1829s |  0.0012x |          - |   46°  | ✅ PASS
MatMul_bf16_hybrid                  |   83.2839s |    0.1802s |  0.0022x |          - |   45°  | ✅ PASS
Monster_ReLU_F32_SSD                |    0.0000s |   10.1291s |  0.0000x |          - |   42°  | ✅ PASS
```

Summary stats at the end: total tests, PASS/FAIL counts, fastest/slowest ratio.

---

## Monster Tests (`tests/benchmarks/monster/`)

Monster tests operate on SSD-resident tensors larger than available RAM (typically 16GB+ = 4B F32 elements). They use `Tensor.from_ssd()` backed by Linux `io_uring` + `O_DIRECT`.

**Requirements:**
- A pre-existing binary file at the expected path (created by `Tensor.new_ssd()`)
- ZFS pool with `recordsize=1M` for optimal alignment
- No PyTorch reference — parity assumed ✅ if the op completes

---

## Why Atomized?

The previous `unified_benchmark.py` ran all tests in a single process. One OOM or assertion failure would abort the entire session. With the Atomized suite:

- **Isolation**: each test runs in its own subprocess — a crash in one doesn't affect others
- **Parallelism**: the orchestrator can run tests in parallel (not currently default, but possible)
- **Reproducibility**: each test saves its own JSON result to `tests/results/`
- **Selective re-runs**: `python tests/benchmarks/f16/matmul_vulkan.py` re-runs exactly one test

---

## Obsolete Test Files (Can be Deleted)

The following files predate the Atomized suite and are **no longer needed**:

| File | Why obsolete |
|:---|:---|
| `tests/unified_benchmark.py` | Monolithic predecessor; saves to `tests/last_results.json` (gone) |
| `tests/big_benchmark.py` | Larger-shape variant of unified; same issues |
| `tests/overnight_bench.py` | Stress tester; hardcodes `/vectorlegis_ssd_pool/` path |
| `tests/analyze_history.py` | Reads `last_results.json` written by unified_benchmark.py |
| `tests/generate_chart.py` | Reads same; has hardcoded `v3.4.0` in chart title |
| `tests/benchmark_gemma_2b.py` | Early alpha (2024); uses `add_into()` which no longer exists |
| `tests/benchmark_gemma_3_4b.py` | Same era; same issues |
| `tests/conftest.py` | Empty pytest fixture; no pytest tests use it |

None of these files produced data that feeds into the current result pipeline.
