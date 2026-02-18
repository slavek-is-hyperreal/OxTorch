import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from vulkan_nn_lib.memory import MemoryManager

def diagnose():
    print("--- VNN Memory Diagnostics ---")
    
    # 1. Check System RAM
    ram_info = MemoryManager.get_mem_info()
    safe_budget = MemoryManager.get_safe_budget() / 1024**3
    print(f"System RAM Total: {ram_info.get('MemTotal', 0) / 1024**3:.2f} GB")
    print(f"System RAM Available: {ram_info.get('MemAvailable', 0) / 1024**3:.2f} GB")
    print(f"VNN Safe RAM Budget: {safe_budget:.2f} GB")
    
    # 2. Check VRAM
    vram_info = MemoryManager.get_vram_info()
    vram_budget = MemoryManager.get_vram_budget() / 1024**2
    print(f"Detected GPU VRAM: {vram_info.get('Total', 0) / 1024**2:.0f} MB")
    print(f"Available GPU VRAM: {vram_info.get('Available', 0) / 1024**2:.0f} MB")
    print(f"VNN VRAM Budget: {vram_budget:.0f} MB")
    
    # 3. Check /sys directly for confirmation
    print("\n--- Linux Node Check (/sys/class/drm) ---")
    for card in ['card0', 'card1', 'card2']:
        base = f"/sys/class/drm/{card}/device"
        if os.path.exists(base):
            total_path = os.path.join(base, "mem_info_vram_total")
            vis_path = os.path.join(base, "mem_info_vis_vram_total")
            if os.path.exists(total_path):
                with open(total_path, 'r') as f:
                    t = int(f.read().strip()) / 1024**2
                print(f"{card} Total VRAM: {t:.0f} MB")
            if os.path.exists(vis_path):
                with open(vis_path, 'r') as f:
                    v = int(f.read().strip()) / 1024**2
                print(f"{card} Visible VRAM (BAR): {v:.0f} MB")

if __name__ == "__main__":
    diagnose()
