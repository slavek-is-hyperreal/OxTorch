import json
from collections import defaultdict

from datetime import datetime

def analyze_history(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    if not data or not isinstance(data, list):
        print("Data is empty or not in expected format.")
        return
        
    # Sort data by timestamp just in case
    # Format: "Mon Mar  9 12:50:35 2026"
    def parse_time(ts):
        try:
            return datetime.strptime(ts, "%a %b %d %H:%M:%S %Y")
        except:
            return datetime.min

    data.sort(key=lambda x: parse_time(x.get('timestamp', '')))
    
    # We want to compare the first time a test appeared to the most recent time
    # 'results' contains a list of test results per run
    # Format of result in 'results': usually contains 'name', 'vnn_time', etc.
    
    test_history = defaultdict(list)
    
    for run in data:
        run_ts = run.get('timestamp')
        results = run.get('results', [])
        for res in results:
            name = res.get('name')
            vnn_time = res.get('vnn_med')
            failed = res.get('failed', False)
            # Skip failed tests or 0.0 measurements (likely placeholders)
            if name and vnn_time is not None and not failed and vnn_time > 0:
                test_history[name].append({'ts': run_ts, 'time': vnn_time})
                
    improvements = []
    regressions = []
    neutral = []
    
    for name, history in test_history.items():
        if len(history) < 2:
            continue
            
        first_time = history[0]['time']
        last_time = history[-1]['time']
        
        # Calculate percentage change
        if first_time > 0:
            change_pct = ((last_time - first_time) / first_time) * 100
        else:
            change_pct = 0
            
        diff_ms = (last_time - first_time) * 1000
        
        record = {
            'name': name,
            'initial': first_time,
            'final': last_time,
            'change_pct': change_pct,
            'diff_ms': diff_ms
        }
        
        # Threshold for regression/improvement: 5% change
        if change_pct <= -5:
            improvements.append(record)
        elif change_pct >= 5:
            regressions.append(record)
        else:
            neutral.append(record)
            
    # Sort by absolute magnitude of percentage change
    improvements.sort(key=lambda x: x['change_pct']) # Most negative first
    regressions.sort(key=lambda x: x['change_pct'], reverse=True) # Most positive first
    neutral.sort(key=lambda x: x['change_pct'])
    
    def print_section(title, items, is_improvement):
        print(f"## {title} ({len(items)} tests)\n")
        if not items:
            print("*None*\n")
            return
            
        print(f"| Test Name | Initial (s) | Final (s) | Change |")
        print(f"| :--- | ---: | ---: | ---: |")
        for item in items:
            sign = "" if is_improvement or item['change_pct'] <= 0 else "+"
            change_str = f"{sign}{item['change_pct']:.2f}%"
            print(f"| `{item['name']}` | {item['initial']:.5f} | {item['final']:.5f} | **{change_str}** |")
        print("\n")

    print("# Performance History Analysis (Initial vs Final run)\n")
    print_section("🚀 SIGNIFICANT IMPROVEMENTS (>5%)", improvements, True)
    print_section("⚠️ SIGNIFICANT REGRESSIONS (>5%)", regressions, False)
    print_section("➡️ STABLE / MINOR CHANGES (<5%)", neutral, True)

if __name__ == "__main__":
    analyze_history('tests/last_results.json')
