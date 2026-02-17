import os

class MemoryManager:
    """Centralized RAM awareness for VNN."""
    
    SYSTEM_RESERVE_BYTES = 1 * 1024 * 1024 * 1024 # Reduced to 1GB for higher utilization
    MAX_TOTAL_USAGE_PCT = 0.90 # 90% of total RAM
    HARD_FLOOR_BYTES = 1024 * 1024 * 1024 # 1.0GB Hard Floor for absolute safety

    @staticmethod
    def get_mem_info():
        info = {}
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    parts = line.split(':')
                    if len(parts) == 2:
                        name = parts[0].strip()
                        value = int(parts[1].split()[0]) * 1024
                        info[name] = value
        except:
            # Fallback if /proc/meminfo is unavailable
            info['MemTotal'] = 8 * 1024 * 1024 * 1024
            info['MemAvailable'] = 4 * 1024 * 1024 * 1024
        return info

    @staticmethod
    def get_vram_info():
        # Heuristic for Linux: check /sys/class/drm/
        try:
            # Check card1 (often the discrete GPU in hybrid setups) or card0
            for card in ['card1', 'card0']:
                total_path = f'/sys/class/drm/{card}/device/mem_info_vram_total'
                if os.path.exists(total_path):
                    with open(total_path, 'r') as f:
                        total = int(f.read().strip())
                    used_path = f'/sys/class/drm/{card}/device/mem_info_vram_used'
                    used = 0
                    if os.path.exists(used_path):
                        with open(used_path, 'r') as f:
                            used = int(f.read().strip())
                    return {'Total': total, 'Available': total - used}
        except:
            pass
        # Fallback for RTX 4090 enthusiasts on other platforms or if detection fails
        return {'Total': 2 * 1024 * 1024 * 1024, 'Available': 1 * 1024 * 1024 * 1024}

    @classmethod
    def get_vram_budget(cls):
        """Dynamic VRAM budget estimate using nvidia-smi (OOM-Safe)."""
        import subprocess
        try:
            # Query nvidia-smi for total and free memory in MB
            res = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.free', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL
            )
            total, free = map(int, res.decode().split(','))
            
            # If VRAM is very low (< 2GB), be extremely paranoid to avoid driver timeouts
            if total < 2048:
                return int(free * 1024 * 1024 * 0.4) 
            return int(free * 1024 * 1024 * 0.7)
        except:
            # Fallback to sysfs if nvidia-smi fails
            info = cls.get_vram_info()
            total = info.get('Total', 1024 * 1024 * 1024)
            available = info.get('Available', total // 2)
            return int(available * 0.6)
    @classmethod
    def get_safe_budget(cls):
        info = cls.get_mem_info()
        total = info.get('MemTotal', 8 * 1024 * 1024 * 1024)
        available = info.get('MemAvailable', total // 2)
        
        # Strategy:
        # 1. Respect maximum physical safety (85% of total)
        # 2. But don't starve the system (Available - 2GB)
        usable_from_total = total * cls.MAX_TOTAL_USAGE_PCT
        usable_from_avail = available - cls.SYSTEM_RESERVE_BYTES
        
        # Budget is the stricter of the two, but at least some minimal amount
        budget = max(256 * 1024 * 1024, min(usable_from_total, usable_from_avail))
        return int(budget)

    @staticmethod
    def should_offload_to_ssd(size_bytes):
        mgr = MemoryManager
        budget = mgr.get_safe_budget()
        # Only offload if tensor exceeds 85% of safe budget (less aggressive)
        return size_bytes > (budget * 0.85)

    @classmethod
    def should_tile(cls, size_bytes):
        """Returns True if the operation should be tiled even on CPU."""
        info = cls.get_mem_info()
        total = info.get('MemTotal', 8 * 1024 * 1024 * 1024)
        # Threshold for tiling: 10% of total RAM
        return size_bytes > (total * 0.1)

    @classmethod
    def wait_for_ram(cls, required_bytes=0):
        """Active backoff if RAM is critical."""
        import time
        while True:
            info = cls.get_mem_info()
            avail = info.get('MemAvailable', 0)
            if avail > cls.HARD_FLOOR_BYTES:
                break
            # print(f"  [DRAS] RAM CRITICAL ({avail/1e9:.1f}GB avail). Throttling...")
            time.sleep(0.1)
