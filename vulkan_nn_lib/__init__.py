from .tensor import Tensor
from . import torch_shim as torch_compat
from .torch_shim import (
    randn, zeros, ones, tensor, from_numpy, arange, manual_seed,
    float32, float16, int32, int16, int8, long, int, double,
    nn, cuda
)
from .modules.base import Module, ModuleList, Sequential
from .modules.layers import Linear, ReLU, SiLU, RMSNorm, Softmax, Embedding, LeakyReLU, Conv2d, MaxPool2d, Upsample
from .modules.tiled import TiledLinear, MatFormerLinear, TiledEmbedding
from .modules.models import Gemma3Block, Gemma3Model, Gemma3ForMultimodalLM, get_cos_sin
from .functional import F
from .optimizers import SGD, Adam, HybridAdam, AutoAdam
from .kernels import warmup

# Alias optim for torch compatibility
import sys
from . import optimizers as optim
sys.modules[__name__ + '.optim'] = optim

__all__ = [
    'Tensor', 'Module', 'ModuleList', 'Sequential',
    'Linear', 'ReLU', 'SiLU', 'RMSNorm', 'Softmax', 'Embedding',
    'TiledLinear', 'MatFormerLinear', 'TiledEmbedding',
    'Gemma3Block', 'Gemma3Model', 'Gemma3ForMultimodalLM', 'get_cos_sin',
    'F', 'GELUTanh', 'SGD', 'Adam', 'HybridAdam', 'AutoAdam', 'warmup', 'nn',
    'randn', 'zeros', 'ones', 'tensor', 'from_numpy', 'arange', 'manual_seed',
    'float32', 'float16', 'int32', 'int16', 'int8', 'long', 'int', 'double', 'cuda', 'optim'
]
