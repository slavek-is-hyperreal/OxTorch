import vulkannn_rusted as vnn
import os

class MemoryManager:
    @staticmethod
    def get_vram_budget():
        """Returns VRAM budget in bytes. Legacy fallback."""
        # Standard heuristic: 75% of available RAM if GPU info is not direct
        return int(vnn.get_available_ram_bytes() * 0.75)

    @staticmethod
    def get_usage_risk():
        """Returns 0-1 risk factor. Legacy fallback."""
        return 0.2 # Stable state
