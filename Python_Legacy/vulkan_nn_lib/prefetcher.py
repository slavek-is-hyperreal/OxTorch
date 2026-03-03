import threading
import time
import os
import numpy as np

def get_available_ram() -> int:
    """Returns the currently available system RAM in bytes recursively."""
    try:
        import psutil
        return getattr(psutil.virtual_memory(), 'available', 8 * 1024**3)
    except ImportError:
        pass
        
    try:
        # Zero-dependency Linux fallback
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
        
    # Worst-case safe fallback: 4GB
    return 4 * 1024**3

class GGUFPrefetcher:
    """Asynchronous Ring-Buffer pipeline for LLM streaming.
    
    Predictively forces the operating system to load sequential 
    GGUF memmap segments from the SSD into the physical RAM (Page Cache) 
    while the GPU is busy with the current layer.
    
    Respects a configurable memory ceiling to prevent OOM errors on low-end hardware,
    or can dynamically adapt to the system's available memory.
    """
    def __init__(self, sequence_plan: list, max_ram_bytes = "auto", reserve_bytes: int = 2 * 1024**3):
        self.sequence_plan = sequence_plan # List of (name, memmap_array) in execution order
        self.max_ram_bytes = max_ram_bytes
        self.reserve_bytes = reserve_bytes # Leave explicit RAM for the OS if running in auto mode
        self.current_idx = 0
        self._freed_idx = 0
        
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
    def _worker(self):
        # We prefetch layers ahead of the GPU up to the RAM limit
        preloaded_bytes = 0
        prefetch_idx = 0
        
        while not self._stop_event.is_set():
            # If the GPU catches up or finishes layers, we advance our sliding window
            if prefetch_idx < self.current_idx or self._freed_idx < self.current_idx:
                # Dynamically evict finished layers from the OS Page Cache!
                while self._freed_idx < self.current_idx and self._freed_idx < len(self.sequence_plan):
                    _, old_arr = self.sequence_plan[self._freed_idx]
                    try:
                        # MADV_DONTNEED tells Linux to immediately drop these pages from RAM
                        os.madvise(old_arr, os.MADV_DONTNEED)
                    except (AttributeError, OSError):
                        pass
                    self._freed_idx += 1
                    
                prefetch_idx = max(prefetch_idx, self.current_idx)
                preloaded_bytes = 0
                
            if prefetch_idx < len(self.sequence_plan):
                name, arr = self.sequence_plan[prefetch_idx]
                arr_size = arr.nbytes
                
                if self.max_ram_bytes == "auto":
                    # Check if the Host OS currently has enough free RAM for this layer + safety reserve
                    can_load = (get_available_ram() - arr_size > self.reserve_bytes)
                else:
                    can_load = (preloaded_bytes + arr_size <= self.max_ram_bytes)
                    
                if can_load:
                    # Force OS to page the data into RAM from SSD 
                    try:
                        os.madvise(arr, os.MADV_WILLNEED)
                    except (AttributeError, OSError):
                        _ = arr[::4096].sum() # Fallback touch
                        
                    preloaded_bytes += arr_size
                    prefetch_idx += 1
                else:
                    # Not enough RAM right now, yield to OS and wait for GPU to finish a layer
                    time.sleep(0.01)
            else:
                time.sleep(0.01)
                
    def notify_layer_done(self, idx: int):
        """GPU calls this after finishing a layer."""
        self.current_idx = idx + 1
        
    def stop(self):
        self._stop_event.set()
        self._thread.join()
