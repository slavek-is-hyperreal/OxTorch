import json
import matplotlib.pyplot as plt
import numpy as np

def generate_chart():
    try:
        with open('tests/last_results.json', 'r') as f:
            data = json.load(f)
        
        # Find the run with runs == 10 (the benchmark we just ran)
        target_run = None
        for run in reversed(data):
            if run.get('runs') == 10:
                target_run = run
                break
                
        if not target_run:
            print("Could not find a benchmark with 10 runs in last_results.json")
            return
            
        results = target_run['results']
        
        # Filter tests that we want to compare (excluding Monster ReLU which is SSD-only)
        comparison_results = [r for r in results if not r['name'].startswith('Monster')]
        
        names = [r['name'] for r in comparison_results]
        vnn_meds = [r['vnn_med'] for r in comparison_results]
        pt_meds = [r['pt_med'] for r in comparison_results]
        
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        rects1 = ax.bar(x - width/2, pt_meds, width, label='PyTorch', color='#ee4c2c')
        rects2 = ax.bar(x + width/2, vnn_meds, width, label='VulkanNN', color='#b7410e')
        
        ax.set_ylabel('Execution Time (seconds, median of 10 runs)')
        ax.set_title(f'VulkanNN-Rusted v3.4.0 vs PyTorch 2.4 (Intel i5-3450 / MD Radeon R7 200)')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        
        # Use log scale because F16/BF16 PyTorch is 100s, while VNN is 0.2s
        ax.set_yscale('log')
        
        fig.tight_layout()
        
        plt.savefig('vulkannn_benchmark_chart.png', dpi=300)
        print("Chart saved as vulkannn_benchmark_chart.png")

    except Exception as e:
        print(f"Error generating chart: {e}")

if __name__ == '__main__':
    generate_chart()
