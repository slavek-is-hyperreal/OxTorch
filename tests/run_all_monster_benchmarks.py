"""
run_all_monster_benchmarks.py
─────────────────────────────
Runs only the tests in tests/benchmarks/monster/.

Monster tests create tensors LARGER than available RAM (available_ram × 1.2),
forcing MSTS to stream data through the SSD ring buffer.

Key difference from regular benchmarks:
  - No PyTorch comparison (tensors too large for PyTorch to hold)
  - Reports MB/s throughput instead of ratio
  - Much longer timeouts (no 600s cap — streaming 13+ GB takes time)
  - Separate results dir: tests/results/monster/

Usage:
    python -m tests.run_all_monster_benchmarks
    python -m tests.run_all_monster_benchmarks --filter ReLU RMSNorm
    python -m tests.run_all_monster_benchmarks --save-baseline
"""

import os
import subprocess
import json
import sys
import time
import argparse


MONSTER_DIR   = "/my_data/gaussian_room/tests/benchmarks/monster"
RESULTS_DIR   = "/my_data/gaussian_room/tests/results/monster"
BASELINE_DIR  = "/my_data/gaussian_room/tests/results/monster/baseline"
PROJECT_ROOT  = "/my_data/gaussian_room"
OXTORCH_PATH  = "/my_data/gaussian_room/vulkannn_rusted"


def discover_monster_benchmarks(base_dir):
    benchmarks = []
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if not d.startswith("__")]
        for file in files:
            if file.endswith(".py") and not file.startswith("__") and file not in ("base.py", "utils.py"):
                rel_path = os.path.relpath(os.path.join(root, file), PROJECT_ROOT)
                mod_name = rel_path.replace("/", ".").replace(".py", "")
                benchmarks.append(mod_name)
    return sorted(benchmarks)


def run_monster(mod_name):
    print(f"\n{'─'*70}")
    print(f"  MONSTER: {mod_name}")
    print(f"{'─'*70}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = OXTORCH_PATH

    try:
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-m", mod_name],
            env=env,
            check=True,
            timeout=7200.0,  # 2h max — these are huge tensors
        )
        elapsed = time.perf_counter() - t0
        print(f"  ✅ Finished in {elapsed:.1f}s")
        return True
    except subprocess.TimeoutExpired:
        print("  ⏰ TIMEOUT (>2h)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"  ❌ FAILED (exit {e.returncode})")
        return False


def aggregate_monster_results(results_dir, baseline_dir):
    results = []
    baselines = {}

    if os.path.exists(baseline_dir):
        for f in os.listdir(baseline_dir):
            if f.endswith(".json"):
                with open(os.path.join(baseline_dir, f)) as fh:
                    d = json.load(fh)
                    baselines[d["name"]] = d

    if not os.path.exists(results_dir):
        return results

    for f in os.listdir(results_dir):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(results_dir, f)) as fh:
            res = json.load(fh)
        res["baseline"] = baselines.get(res["name"])
        results.append(res)

    return results


def print_monster_summary(data, total_found):
    if not data:
        print("No monster results found.")
        return

    print("\n" + "=" * 100)
    print(f"{'OxTorch MONSTER BENCHMARK SUMMARY':^100}")
    print(f"{'(all tensors > available RAM — MSTS SSD streaming required)':^100}")
    print("=" * 100)

    headers = ["Test Case", "VNN Time", "Tensor GB", "MB/s", "Baseline MB/s", "CPU°C", "Status"]
    print(
        f"{headers[0]:<35} | {headers[1]:>10} | {headers[2]:>9} | "
        f"{headers[3]:>9} | {headers[4]:>13} | {headers[5]:>6} | {headers[6]}"
    )
    print("-" * 100)

    passed = 0
    for res in sorted(data, key=lambda r: r.get("throughput_mbs", 0), reverse=True):
        parity_ok = res.get("parity", True)
        if parity_ok:
            passed += 1

        vnn_t   = res.get("vnn_time", 0.0)
        gb      = res.get("tensor_gb", 0.0)
        mbs     = res.get("throughput_mbs", 0.0)
        temp    = res.get("cpu_temp", 0.0)

        baseline = res.get("baseline")
        baseline_mbs_str = f"{baseline['throughput_mbs']:.1f}" if baseline and "throughput_mbs" in baseline else "—"

        # Regression flag: >20% slower than baseline
        slow_flag = ""
        if baseline and "throughput_mbs" in baseline and mbs < baseline["throughput_mbs"] * 0.8:
            slow_flag = " ⚠️ SLOW"

        status = ("✅ PASS" if parity_ok else "❌ FAIL") + slow_flag

        print(
            f"{res['name']:<35} | {vnn_t:>9.2f}s | {gb:>8.2f}G | "
            f"{mbs:>8.1f}M | {baseline_mbs_str:>13} | {temp:>5.0f}° | {status}"
        )

    print("=" * 100)
    print(f"Benchmarks Found:    {total_found}")
    print(f"Benchmarks Executed: {len(data)}")
    print(f"Parity Passed:       {passed}/{len(data)}")
    print("=" * 100)
    print()
    print("NOTE: Ratio column is omitted — PyTorch cannot hold tensors this large.")
    print("      Compare MB/s against your SSD sequential read speed instead.")
    print("      SATA SSD target: ≥ 400 MB/s  |  NVMe target: ≥ 2500 MB/s")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OxTorch Monster benchmarks (out-of-RAM MSTS streaming)")
    parser.add_argument("--filter",       type=str, nargs="+", help="Only run tests whose name contains any of these strings")
    parser.add_argument("--save-baseline", action="store_true", help="Save current results as baseline for regression tracking")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Clear previous non-baseline monster results
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(RESULTS_DIR, f))

    mods = discover_monster_benchmarks(MONSTER_DIR)
    if args.filter:
        mods = [m for m in mods if any(kw.lower() in m.lower() for kw in args.filter)]

    print(f"\n🐉 Monster benchmarks found: {len(mods)}")
    print("   Each tensor is sized to available_ram × 1.2 — MSTS SSD streaming is mandatory.\n")

    for m in mods:
        run_monster(m)

    all_results = aggregate_monster_results(RESULTS_DIR, BASELINE_DIR)
    print_monster_summary(all_results, len(mods))

    if args.save_baseline:
        os.makedirs(BASELINE_DIR, exist_ok=True)
        for res in all_results:
            path = os.path.join(BASELINE_DIR, f"{res['name'].lower()}.json")
            with open(path, "w") as fh:
                json.dump(res, fh, indent=2)
        print(f"✅ Saved {len(all_results)} monster results as baseline.")
