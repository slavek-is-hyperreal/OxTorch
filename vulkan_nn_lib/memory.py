import os
import time
import subprocess

class SafetyViolationError(Exception):
    """Triggered when system RAM usage nears critical limits."""
    pass

class MemoryManager:
    """Centralized RAM awareness for VNN."""
    
    # User-specified hardware constraints: Default 5GB for ZFS/Safety.
    # Can be overridden via environment variable VNN_RESERVE_GB
    _system_reserve_bytes = int(os.getenv('VNN_RESERVE_GB', 5)) * 1024**3
    
    MAX_TOTAL_USAGE_PCT = 0.80 # 80% of usable RAM
    HARD_FLOOR_BYTES = 3 * 1024 * 1024 * 1024 # 3GB "Freeze Protection" floor
    CRITICAL_SYSTEM_USAGE_BYTES = 21 * 1024 * 1024 * 1024 # Overwritten if total RAM is different
    
    _mem_info_cache = None
    _mem_info_time = 0.0
    _mem_total_cached = None
    CACHE_TTL = 0.1 # 100ms
    
    @classmethod
    def get_mem_info(cls):
        now = time.time()
        if cls._mem_info_cache and (now - cls._mem_info_time < cls.CACHE_TTL):
            return cls._mem_info_cache
            
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
            info['MemTotal'] = 8 * 1024 * 1024 * 1024
            info['MemAvailable'] = 4 * 1024 * 1024 * 1024
            
        cls._mem_info_cache = info
        cls._mem_info_time = now
        if cls._mem_total_cached is None:
            total = info.get('MemTotal', 8 * 1024**3)
            cls._mem_total_cached = total
            # Adjust critical thresholds based on actual RAM
            cls.CRITICAL_SYSTEM_USAGE_BYTES = total - (1 * 1024**3) # 1GB Margin of Error
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
        total = info.get('MemTotal', 8 * 1024**3)
        available = info.get('MemAvailable', total // 2)
        
        # Strategy:
        # 1. Respect the static reservation floor (e.g. ZFS, OS overhead).
        # 2. We use max(0, available - reservation) as the starting point.
        usable_available = max(0, available - cls._system_reserve_bytes)
        
        # 3. Take 80% of the truly usable available RAM.
        budget = usable_available * cls.MAX_TOTAL_USAGE_PCT
        
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
        if cls._mem_total_cached is None:
            cls.get_mem_info()
        # Threshold for tiling: 10% of total RAM
        return size_bytes > (cls._mem_total_cached * 0.1)

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
