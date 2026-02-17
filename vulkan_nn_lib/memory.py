import os

class SafetyViolationError(Exception):
    """Triggered when system RAM usage nears critical limits."""
    pass

class MemoryManager:
    """Centralized RAM awareness for VNN."""
    
    # User-specified hardware constraints: 20GB usable, 80% cap. 5GB ZFS fixed.
    SYSTEM_RESERVE_BYTES = 5 * 1024 * 1024 * 1024 # 5GB Reserve (ZFS + Safety)
    MAX_TOTAL_USAGE_PCT = 0.80 # 80% of usable RAM
    HARD_FLOOR_BYTES = 3 * 1024 * 1024 * 1024 # Raised to 3GB for ZFS and DRAS v4 stability
    CRITICAL_SYSTEM_USAGE_BYTES = 21 * 1024 * 1024 * 1024 # 21GB: Near user's 22GB limit
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
        # 1. We assume ~20GB are available for apps (after ZFS).
        # 2. We take 80% of what's reported as available, BUT cap at 16GB.
        usable_from_avail = available * cls.MAX_TOTAL_USAGE_PCT
        
        # Budget cap at 16GB per user spec
        max_vnn_budget = 16 * 1024 * 1024 * 1024
        budget = min(usable_from_avail, max_vnn_budget)
        
        return int(max(256 * 1024 * 1024, budget))

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
    def get_usage_risk(cls):
        """Returns risk level: 0 (safe) to 1.0 (critical)."""
        info = cls.get_mem_info()
        # total used = Total - Available
        mem_total = info.get('MemTotal', 20*1024**3)
        mem_available = info.get('MemAvailable', 0)
        used = mem_total - mem_available
        
        if used >= cls.CRITICAL_SYSTEM_USAGE_BYTES: return 1.0
        # Start showing risk from 80% of critical
        risk_base = cls.CRITICAL_SYSTEM_USAGE_BYTES * 0.8
        if used < risk_base: return 0.0
        
        risk = (used - risk_base) / (cls.CRITICAL_SYSTEM_USAGE_BYTES - risk_base)
        return max(0.0, min(1.0, risk))

    @classmethod
    def wait_for_ram(cls, required_bytes=0):
        """Active backoff if RAM is critical. Raises SafetyViolationError if usage is unsafe."""
        import time
        while True:
            info = cls.get_mem_info()
            avail = info.get('MemAvailable', 0)
            risk = cls.get_usage_risk()
            
            # 21.5GB+ approximately for 24GB total logic
            if risk >= 0.98: 
                raise SafetyViolationError(f"Adaptive Safety Trigger: Usage ({ (info.get('MemTotal',0)-avail)/1e9:.1f}GB) exceeds safety bounds.")
                
            if avail > cls.HARD_FLOOR_BYTES and risk < 0.9:
                break
            time.sleep(0.05)
