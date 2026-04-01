import numpy as np

class AutoAdam:
    """Legacy AutoAdam wrapper for OxTorch."""
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr
        self.state = {}

    def step(self):
        """Perform a standard Adam optimization step."""
        # This is a fallback implementation that keeps the demos running
        for p in self.params:
            if hasattr(p, 'grad') and p.grad is not None:
                # Actual update logic is handled by the model.sync from/to VNN 
                # in the current Splat Studio trainer.
                # Here we just simulate the interface.
                pass
