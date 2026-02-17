from ..tensor import Tensor

class Module:
    """Base class for all VulkanNN layers, mimicking torch.nn.Module."""
    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values(): m.train(mode)
        return self

    def eval(self): return self.train(False)

    def parameters(self):
        for p in self._parameters.values(): yield p
        for m in self._modules.values(): yield from m.parameters()

    def named_parameters(self, prefix=''):
        for name, p in self._parameters.items(): yield prefix + name, p
        for m_name, m in self._modules.items(): yield from m.named_parameters(prefix + m_name + '.')

    def __setattr__(self, name, value):
        if isinstance(value, Module): self._modules[name] = value
        elif isinstance(value, Tensor): self._parameters[name] = value
        super().__setattr__(name, value)

    def state_dict(self):
        sd = {}
        for name, param in self._parameters.items(): sd[name] = param.to_numpy()
        for name, module in self._modules.items():
            for k, v in module.state_dict().items(): sd[f"{name}.{k}"] = v
        return sd

    def load_state_dict(self, state_dict):
        for name, param in self._parameters.items():
            if name in state_dict:
                import numpy as np
                val = state_dict[name].astype(np.float32)
                if param.device == 'vulkan':
                    param.arr.from_numpy(val.flatten())
                else:
                    # CPU or SSD (memmap)
                    param.arr.reshape(param.shape)[...] = val
        for m_name, module in self._modules.items():
            child_sd = {k[len(m_name)+1:]: v for k, v in state_dict.items() if k.startswith(f"{m_name}.")}
            module.load_state_dict(child_sd)

class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._layers = []
        if modules:
            for m in modules: self.append(m)

    def append(self, module):
        idx = len(self._layers)
        self._layers.append(module)
        self._modules[str(idx)] = module

    def __getitem__(self, idx): return self._layers[idx]
    def __len__(self): return len(self._layers)
    def __iter__(self): return iter(self._layers)

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(layers): self._modules[str(i)] = layer

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        return x
