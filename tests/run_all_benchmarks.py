import os
import subprocess
import json
import sys
import time
import time

def discover_benchmarks(base_dir):
    benchmarks = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("__") and file != "generate_atoms.py" and file != "base.py" and file != "utils.py":
                # Convert path to module name
                rel_path = os.path.relpath(os.path.join(root, file), "/my_data/gaussian_room")
                mod_name = rel_path.replace("/", ".").replace(".py", "")
                benchmarks.append(mod_name)
    return sorted(benchmarks)

def run_benchmark(mod_name):
    print(f"Running {mod_name}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/my_data/gaussian_room"
    try:
        subprocess.run([sys.executable, "-m", mod_name], env=env, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"FAILED: {mod_name}")
        return False

def aggregate_results(results_dir):
    all_data = []
    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            with open(os.path.join(results_dir, file), 'r') as f:
                all_data.append(json.load(f))
    return all_data

def print_summary(data):
    if not data:
        print("No results found.")
        return

    print("\n" + "="*105)
    print(f"{'OxTorch ATOMIZED BENCHMARK SUMMARY':^105}")
    print("="*105)
    headers = ["Test Case", "PT Time", "VNN Time", "Ratio", "CPU°C", "Parity"]
    rows = []
    for res in data:
        ratio = res["ratio"]
        ratio_str = f"{ratio:.2f}x" if ratio > 0.1 else f"{ratio:.4f}x"
        parity = "✅ PASS" if res["parity"] else "❌ FAIL"
        rows.append([
            res["name"],
            f"{res['pt_time']:.4f}s",
            f"{res['vnn_time']:.4f}s",
            ratio_str,
            f"{res['cpu_temp']:.0f}°",
            parity
        ])
    
    # Simple table printing if tabulate is missing
    print(f"{headers[0]:<35} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<8} | {headers[4]:<6} | {headers[5]}")
    print("-" * 105)
    for row in rows:
        print(f"{row[0]:<35} | {row[1]:>10} | {row[2]:>10} | {row[3]:>8} | {row[4]:>5} | {row[5]}")
    print("="*105)

if __name__ == "__main__":
    base_dir = "/my_data/gaussian_room/tests/benchmarks"
    results_dir = "/my_data/gaussian_room/tests/results"
    
    # Optional: Clear old results
    # for f in os.listdir(results_dir): os.remove(os.path.join(results_dir, f))

    mods = discover_benchmarks(base_dir)
    print(f"Found {len(mods)} benchmarks.")

    for m in mods:
        run_benchmark(m)

    all_results = aggregate_results(results_dir)
    print_summary(all_results)
