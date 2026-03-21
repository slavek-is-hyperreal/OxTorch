"""
MSTS Path A (Direct) — ReLU F32 SSD
Tensor size: < MSTS_DIRECT_MAX (≈3 MB for i5-3450 with L3=6MB).
Zero thread overhead: mmap read_exact + single AVX loop.

Benchmark fairness note:
  Both OxTorch AND PyTorch measure the full disk pipeline:
    read from SSD → compute ReLU → write result to SSD
  OxTorch:  io_uring + O_DIRECT (zero page cache)
  PyTorch:  np.fromfile → torch.relu → np.tofile (via kernel VFS)
  This gives a fair comparison of end-to-end SSD throughput.
"""
import os, sys, time
import numpy as np

_OXTORCH_PATH = "/my_data/gaussian_room/vulkannn_rusted"
if _OXTORCH_PATH not in sys.path:
    sys.path.insert(0, _OXTORCH_PATH)
sys.path.insert(0, "/my_data/gaussian_room")

from tests.benchmarks.utils import load_vnn, save_benchmark_result

# Path A threshold: < 3MB → ~750K f32 elements
N_ELEMENTS = 512_000  # ~2MB F32, well below MSTS_DIRECT_MAX
ITERS = 3
NAME = "MSTS_PathA_ReLU_F32_SSD"
SSD_IN  = "/tmp/msts_bench_a.ssd"
SSD_OUT = "/tmp/msts_bench_a_relu.ssd"
SSD_PT_OUT = "/tmp/msts_bench_a_pt_relu.bin"


def _drop_page_cache_file(path):
    """Best-effort page cache eviction for a single file."""
    try:
        with open(path, "rb") as f:
            import fcntl, ctypes
            # POSIX_FADV_DONTNEED = 4  — hint kernel to drop page cache for this file
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.posix_fadvise(f.fileno(), 0, 0, 4)
    except Exception:
        pass


def run():
    import torch
    vnn, _ = load_vnn()

    # --- Write input file (shared by both OxTorch and PyTorch baseline) ---
    data = np.ascontiguousarray(np.random.randn(N_ELEMENTS).astype(np.float32))
    data.tofile(SSD_IN)

    # ── OxTorch: io_uring + O_DIRECT (bypasses page cache) ──────────────────
    t_vnn = 0.0
    result_tensor = None
    for _ in range(ITERS):
        _drop_page_cache_file(SSD_IN)
        t_ssd = vnn.Tensor.from_ssd(SSD_IN, shape=[N_ELEMENTS], dtype=vnn.DataType.F32)
        t0 = time.perf_counter()
        result_tensor = t_ssd.unary_op_ssd("relu", 0.0, 0.0)
        t_vnn += time.perf_counter() - t0
    t_vnn /= ITERS

    # ── PyTorch baseline: kernel VFS (fromfile + relu + tofile) ─────────────
    t_pt = 0.0
    for _ in range(ITERS):
        _drop_page_cache_file(SSD_IN)
        t0 = time.perf_counter()
        arr = np.fromfile(SSD_IN, dtype=np.float32)
        arr_out = torch.relu(torch.from_numpy(arr)).numpy()
        arr_out.tofile(SSD_PT_OUT)
        t_pt += time.perf_counter() - t0
    t_pt /= ITERS

    # ── Parity ───────────────────────────────────────────────────────────────
    result_data = np.array(result_tensor.load_to_f32_vec_msts())
    expected = np.maximum(data, 0.0)
    max_diff = float(np.max(np.abs(result_data - expected)))
    parity = max_diff < 1e-4

    ratio = t_vnn / t_pt if t_pt > 0 else 0.0
    mb_per_s_vnn = (N_ELEMENTS * 4 / 1e6) / t_vnn
    mb_per_s_pt  = (N_ELEMENTS * 4 / 1e6) / t_pt

    print(f"\n>>> TEST: {NAME} | SSD Path A (Direct) | N={N_ELEMENTS:,} | Iter: {ITERS}")
    print(f"    Comparing: disk read→relu→disk write (same workload for both)")
    print(f"    [PyTorch/VFS]  {t_pt:.4f}s  ({mb_per_s_pt:.1f} MB/s)")
    print(f"    [OxTorch/uring]{t_vnn:.4f}s  ({mb_per_s_vnn:.1f} MB/s) | Ratio: {ratio:.2f}x | Parity: {'✅ PASS' if parity else '❌ FAIL'} (max_diff={max_diff:.2e})")

    save_benchmark_result(NAME, {
        "name": NAME, "op": "ReLU", "dtype": "f32", "mode": "ssd",
        "pt_time": t_pt, "vnn_time": t_vnn, "ratio": ratio,
        "parity": parity, "max_diff": max_diff, "cpu_temp": 0.0
    })

    for p in [SSD_IN, SSD_OUT, SSD_PT_OUT]:
        try: os.unlink(p)
        except: pass


if __name__ == "__main__":
    run()
