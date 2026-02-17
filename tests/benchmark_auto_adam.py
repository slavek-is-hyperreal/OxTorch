#!/usr/bin/env python3
"""5-Way Adam Benchmark with Smart Resource Management.
Optimized for 2GB VRAM and 24GB RAM hardware.
"""
import subprocess, sys, json, os, numpy as np, gc

TILE = 4 * 1024 * 1024
BENCH_DIR = "/vectorlegis_ssd_pool/vnn_cache/bench"

# Worker script run in isolated subprocess
WORKER = '''
import time, json, sys, gc, numpy as np
import vulkan_nn_lib.core as vnn
from vulkan_nn_lib.tensor import Tensor
import torch
import os, resource

def get_peak_ram():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

vnn.warmup()
Tensor.setup_ssd_storage("/vectorlegis_ssd_pool/vnn_cache/bench_tensors")

mode = sys.argv[1]
n_elems = int(sys.argv[2])
iters = int(sys.argv[3])
tile_size = int(sys.argv[4])
tmp_dir = "/vectorlegis_ssd_pool/vnn_cache/bench"

device = 'auto'

# Load data as views to avoid OOM
w_np = np.load(f"{tmp_dir}/_bench_w.npy", mmap_mode='r')
grads = [np.load(f"{tmp_dir}/_bench_g{j}.npy", mmap_mode='r') for j in range(iters)]

if mode == "torch":
    if n_elems * 4 * 4 > 22 * 1024*1024*1024:
        print(json.dumps({"elapsed": 0, "skipped": True}))
        sys.exit(0)
    wt = torch.tensor(w_np.copy(), requires_grad=True)
    opt = torch.optim.Adam([wt], lr=1e-3)
    for g in grads: opt.zero_grad(); wt.grad = torch.tensor(g); opt.step()
    wt = torch.tensor(w_np.copy(), requires_grad=True)
    opt = torch.optim.Adam([wt], lr=1e-3)
    t0 = time.perf_counter()
    for g in grads: opt.zero_grad(); wt.grad = torch.tensor(g); opt.step()
    elapsed = time.perf_counter() - t0
    result = wt.detach().numpy().flatten()

elif mode == "gpu":
    # Pure GPU Adam requires 4x model size in VRAM.
    if n_elems * 4 > 256 * 1024 * 1024: 
         print(json.dumps({"elapsed": 0, "skipped": True}))
         sys.exit(0)
    ww = Tensor(w_np, device='vulkan', requires_grad=True)
    ow = vnn.Adam([ww], lr=1e-3, tile_size=tile_size)
    ow.zero_grad(); ww.grad = Tensor(grads[0], device='vulkan'); ow.step()
    del ww, ow; gc.collect()
    w = Tensor(w_np, device='vulkan', requires_grad=True)
    opt = vnn.Adam([w], lr=1e-3, tile_size=tile_size)
    t0 = time.perf_counter()
    for g in grads: opt.zero_grad(); w.grad = Tensor(g, device='vulkan'); opt.step()
    elapsed = time.perf_counter() - t0
    result = w.to_numpy().flatten()

elif mode == "hybrid":
    ww = Tensor(w_np, device=device, requires_grad=True)
    ow = vnn.HybridAdam([ww], lr=1e-3, tile_size=tile_size)
    ow.zero_grad(); ww.grad = Tensor(grads[0], device=device); ow.step()
    del ww, ow; gc.collect()
    w = Tensor(w_np, device=device, requires_grad=True)
    opt = vnn.HybridAdam([w], lr=1e-3, tile_size=tile_size)
    t0 = time.perf_counter()
    for g in grads: opt.zero_grad(); w.grad = Tensor(g, device=device); opt.step()
    elapsed = time.perf_counter() - t0
    result = w.to_numpy().flatten()

elif mode == "auto":
    ww = Tensor(w_np, device=device, requires_grad=True)
    ow = vnn.AutoAdam([ww], lr=1e-3)
    ow.zero_grad(); ww.grad = Tensor(grads[0], device=device); ow.step()
    del ww, ow; gc.collect()
    w = Tensor(w_np, device=device, requires_grad=True)
    opt = vnn.AutoAdam([w], lr=1e-3)
    t0 = time.perf_counter()
    for g in grads: opt.zero_grad(); w.grad = Tensor(g, device=device); opt.step()
    elapsed = time.perf_counter() - t0
    result = w.to_numpy().flatten()

np.save(f"{tmp_dir}/_bench_result.npy", result)
print(json.dumps({"elapsed": elapsed, "peak_ram": get_peak_ram()}))
'''

def run_mode(mode, n_elems, iters, tile_size):
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    result = subprocess.run(
        [sys.executable, '-c', WORKER, mode, str(n_elems), str(iters), str(tile_size)],
        capture_output=True, text=True, cwd='/my_data/gaussian_room', env=env
    )
    if result.returncode != 0:
        print(f"  ❌ {mode} FAILED:")
        lines = result.stderr.strip().split('\n')
        for line in lines[-10:]:
            print(f"     {line}")
        return None, None
        
    all_lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
    for line in all_lines:
        if not line.startswith('{'):
             print(f"  {line}")
             
    data = None
    for line in reversed(all_lines):
        try:
            val = json.loads(line)
            if isinstance(val, dict) and 'elapsed' in val:
                data = val
                break
        except:
            continue
            
    if data is None:
        return None, None
    if data.get("skipped"):
        return None, "skipped"
    r = np.load(f"{BENCH_DIR}/_bench_result.npy", mmap_mode='r')
    return data, r

def bench(n_elems, iters=3):
    mb = n_elems * 4 / (1024*1024)
    print(f"\n{'='*72}")
    print(f"🚀 {n_elems/1e6:.1f}M elements ({mb:.0f}MB model), {iters} iters")
    print(f"{'='*72}")

    os.makedirs(BENCH_DIR, exist_ok=True)
    os.makedirs("/vectorlegis_ssd_pool/vnn_cache/bench_tensors", exist_ok=True)
    
    # Save to SSD (chunked to avoid OOM in preparation)
    print("  Preparing data on SSD...")
    w_filename = f"{BENCH_DIR}/_bench_w.npy"
    chunk_size = 64 * 1024 * 1024  # 256MB per chunk
    if not os.path.exists(w_filename):
        w_mmap = np.lib.format.open_memmap(w_filename, mode='w+', dtype='float32', shape=(n_elems,))
        for i in range(0, n_elems, chunk_size):
            size = min(chunk_size, n_elems - i)
            w_mmap[i:i+size] = np.random.randn(size).astype(np.float32) * 0.02
            if i % (chunk_size * 4) == 0:
                print(f"    W: {i/n_elems*100:3.1f}% done")
        w_mmap.flush()
        del w_mmap; gc.collect()
    
    for j in range(iters):
        g_filename = f"{BENCH_DIR}/_bench_g{j}.npy"
        if not os.path.exists(g_filename):
            g_mmap = np.lib.format.open_memmap(g_filename, mode='w+', dtype='float32', shape=(n_elems,))
            for i in range(0, n_elems, chunk_size):
                size = min(chunk_size, n_elems - i)
                g_mmap[i:i+size] = np.random.randn(size).astype(np.float32) * 0.01
                if i % (chunk_size * 4) == 0:
                    print(f"    G{j}: {i/n_elems*100:3.1f}% done")
            g_mmap.flush()
            del g_mmap; gc.collect()

    modes = [
        ("Torch CPU",    "torch"),
        ("VNN GPU-only", "gpu"),
        ("VNN Hybrid",   "hybrid"),
        ("AutoAdam VBR", "auto"),
    ]

    results = {}
    for label, mode in modes:
        print(f"\n  Running {label}...")
        data, r = run_mode(mode, n_elems, iters, TILE)
        if data == "skipped":
            print(f"  ⚠️ {label} SKIPPED")
            continue
        if data is not None:
            results[label] = (data, r)

    if "Torch CPU" in results:
        data_torch, ref = results["Torch CPU"]
        t_torch = data_torch['elapsed']
        print(f"\n  {'Mode':<20} {'Peak RAM':>9} {'Per Step':>11} {'vs Torch':>8} Match")
        print(f"  {'-'*68}")
    else:
        print("\n  ⚠️ No Torch baseline (Too large for RAM)")
        print(f"\n  {'Mode':<20} {'Peak RAM':>9} {'Per Step':>11} Match")
        print(f"  {'-'*55}")
        t_torch = None

    for label, mode in modes:
        if label not in results: continue
        data, r = results[label]
        t = data['elapsed']
        ram = data['peak_ram']
        if t_torch:
            ratio = t_torch / t
            ok = np.allclose(r.flatten()[:1000], ref.flatten()[:1000], atol=1e-3)
            mark = '✅' if ok else '❌'
            print(f"  {label:<20} {ram:7.0f}MB {t*1000/iters:10.0f}ms  x{ratio:5.2f}    {mark}")
        else:
            print(f"  {label:<20} {ram:7.0f}MB {t*1000/iters:10.0f}ms  N/A")

    import shutil
    try:
        shutil.rmtree("/vectorlegis_ssd_pool/vnn_cache/bench_tensors")
    except:
        pass

if __name__ == '__main__':
    # 1. 1GB Test (Fits in RAM/VRAM overlap)
    bench(256 * 1024*1024)
    # 2. 8GB Test (Fits in RAM, uses SSD for Adam states)
    bench(2 * 1024 * 1024*1024)
    # 3. 32GB Monster test (Uses SSD for everything)
    print("\n🛑 STARTING MONSTER TEST (32GB)...")
    bench(8 * 1024 * 1024*1024, iters=1)
