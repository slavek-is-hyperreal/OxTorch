import os
import subprocess
import json
import sys
import time
import time

def discover_benchmarks(base_dir):
    benchmarks = []
    for root, dirs, files in os.walk(base_dir):
        # Never descend into the monster/ directory — use run_all_monster_benchmarks.py for that
        dirs[:] = [d for d in dirs if d != "monster"]
        for file in files:
            if file.endswith(".py") and not file.startswith("__") and file != "generate_atoms.py" and file != "base.py" and file != "utils.py":
                rel_path = os.path.relpath(os.path.join(root, file), "/my_data/gaussian_room")
                mod_name = rel_path.replace("/", ".").replace(".py", "")
                benchmarks.append(mod_name)
    return sorted(benchmarks)

def run_benchmark(mod_name, baseline_time=None):
    print(f"Running {mod_name}...", end=" ", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = "/my_data/gaussian_room/vulkannn_rusted"
    
    # 2x slowdown threshold (+ some grace for noise)
    timeout = baseline_time * 3.0 if baseline_time else 600.0
    
    try:
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-m", mod_name], env=env, check=True, timeout=timeout)
        print("DONE")
        return True
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (exceeded {timeout:.1f}s)")
        return False
    except subprocess.CalledProcessError:
        print(f"FAILED (error)")
        return False

def aggregate_results(results_dir, baseline_dir):
    all_data = []
    # Load baselines
    baselines = {}
    if os.path.exists(baseline_dir):
        for f in os.listdir(baseline_dir):
            if f.endswith(".json"):
                with open(os.path.join(baseline_dir, f), 'r') as b:
                    data = json.load(b)
                    baselines[data["name"]] = data["vnn_time"]

    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            with open(os.path.join(results_dir, file), 'r') as f:
                res = json.load(f)
                res["baseline"] = baselines.get(res["name"])
                all_data.append(res)
    return all_data

def print_summary(data, total_found):
    if not data:
        print("No results found.")
        return

    print("\n" + "="*115)
    print(f"{'OxTorch ATOMIZED BENCHMARK SUMMARY':^115}")
    print("="*115)
    headers = ["Test Case", "PT Time", "VNN Time", "Ratio", "Baseline", "CPU°C", "Status"]
    rows = []
    
    faster_count = 0
    slower_count = 0
    passed_count = 0
    
    for res in data:
        ratio = res["ratio"]
        ratio_str = f"{ratio:.2f}x" if ratio > 0.1 else f"{ratio:.4f}x"
        
        is_pass = res["parity"]
        status = "✅ PASS" if is_pass else "❌ FAIL"
        if is_pass: passed_count += 1
        
        if ratio < 0.95: # 5% margin for noise
            faster_count += 1
        elif ratio > 1.05:
            slower_count += 1
            
        baseline = res.get("baseline")
        b_str = f"{baseline:.4f}s" if baseline else "-"
        
        if baseline and res["vnn_time"] > baseline * 2.0:
            status = "⚠️ SLOW"
        
        rows.append([
            res["name"],
            f"{res['pt_time']:.4f}s",
            f"{res['vnn_time']:.4f}s",
            ratio_str,
            b_str,
            f"{res['cpu_temp']:.0f}°",
            status
        ])
    
    # Sort by performance ratio (FASTER first)
    rows.sort(key=lambda x: float(x[3].replace("x", "")))
    
    print(f"{headers[0]:<35} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<8} | {headers[4]:<10} | {headers[5]:<6} | {headers[6]}")
    print("-" * 115)
    for row in rows:
        print(f"{row[0]:<35} | {row[1]:>10} | {row[2]:>10} | {row[3]:>8} | {row[4]:>10} | {row[5]:>5} | {row[6]}")
    print("="*115)
    
    print(f"Benchmarks Found:    {total_found}")
    print(f"Benchmarks Executed: {len(data)}")
    print(f"Parity Passed:       {passed_count}/{len(data)}")
    print(f"OxTorch FASTER:      {faster_count}")
    print(f"PyTorch FASTER:      {slower_count}")
    print("="*115)

if __name__ == "__main__":
    base_dir = "/my_data/gaussian_room/tests/benchmarks"
    results_dir = "/my_data/gaussian_room/tests/results"
    baseline_dir = "/my_data/gaussian_room/tests/results/baseline"

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--filter", type=str, nargs="+")
    args = parser.parse_args()

    # Clear old results (but keep baselines)
    for f in os.listdir(results_dir):
        if f.endswith(".json"):
            os.remove(os.path.join(results_dir, f))

    mods = discover_benchmarks(base_dir)
    if args.filter:
        mods = [m for m in mods if any(f.lower() in m.lower() for f in args.filter)]
    
    print(f"Found {len(mods)} benchmarks.")

    # Load existing baselines for timeouts
    curr_baselines = {}
    if os.path.exists(baseline_dir):
        for f in os.listdir(baseline_dir):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(baseline_dir, f), 'r') as b:
                        d = json.load(b)
                        curr_baselines[d["name"]] = d["vnn_time"]
                except: pass

    for m in mods:
        # Match module name to test name (usually the same but inside module path)
        # We'll just pass None if uncertain or try to match.
        run_benchmark(m)

    all_results = aggregate_results(results_dir, baseline_dir)
    print_summary(all_results, len(mods))
    
    if args.save_baseline:
        os.makedirs(baseline_dir, exist_ok=True)
        for res in all_results:
            b_path = os.path.join(baseline_dir, f"{res['name'].lower()}.json")
            with open(b_path, 'w') as f:
                json.dump(res, f)
        print(f"Saved {len(all_results)} results as baseline.")
