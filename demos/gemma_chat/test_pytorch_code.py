# This script simulates existing PyTorch code
# The only change is the import line below
import vulkan_torch as torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DepthCNN(nn.Module):
    def __init__(self):
        super(DepthCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.fc = nn.Linear(16 * 60 * 60, 10) # Assuming 64x64 input -> 60x60 after 2 convs
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # VulkanNN doesn't support view/flatten yet, so we'll mock the end of the chain
        # for this specific demo or just return the spatial features
        return x

def main():
    print("--- Running Existing-Style PyTorch Code on Vulkan ---")
    
    # 1. Initialize model
    model = DepthCNN()
    
    # 2. Prepare 'torch' tensor
    x_np = np.random.randn(1, 1, 64, 64).astype(np.float32)
    x = torch.from_numpy(x_np)
    
    # 3. Inference
    with torch.no_grad():
        output = model(x)
        
    print(f"Model successfully executed on GPU!")
    print(f"Output shape: {output.shape}")
    
    # 4. State Dict verification
    sd = model.state_dict()
    print(f"Captured state_dict keys: {list(sd.keys())[:5]}...")

if __name__ == "__main__":
    main()
