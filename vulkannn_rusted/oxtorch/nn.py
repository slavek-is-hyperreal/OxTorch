import oxtorch as torch

class Module:
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = args
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
    def forward(self, x):
        # Proxies to native vnn.Tensor.conv2d if available
        # or falls back to PT via oxtorch.Tensor getattr
        return x.conv2d(self.in_channels, self.out_channels, self.kernel_size)

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Upsample(Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return x.upsample(self.scale_factor)
