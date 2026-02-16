# Compatibility bridge for older imports
from .tensor import Tensor
from .modules.base import Module, ModuleList, Sequential
from .modules.layers import Linear, ReLU, SiLU, RMSNorm, Softmax, Embedding, LeakyReLU, Conv2d, MaxPool2d, Upsample
from .modules.tiled import TiledLinear, MatFormerLinear, TiledEmbedding
from .modules.models import Gemma3Block, Gemma3Model, Gemma3ForMultimodalLM, get_cos_sin
from .functional import F, GELUTanh
from .optimizers import SGD, Adam
from .kernels import warmup, ti, np

# Re-expose namespaces
class nn:
    Module = Module
    ModuleList = ModuleList
    Sequential = Sequential
    Linear = Linear
    TiledLinear = TiledLinear
    MatFormerLinear = MatFormerLinear
    ReLU = ReLU
    SiLU = SiLU
    RMSNorm = RMSNorm
    Softmax = Softmax
    Embedding = Embedding
    LeakyReLU = LeakyReLU
    Conv2d = Conv2d
    MaxPool2d = MaxPool2d
    Upsample = Upsample
    RMSNorm = RMSNorm
    Softmax = Softmax
    Embedding = Embedding
    TiledEmbedding = TiledEmbedding

# Re-expose all names for wildcard imports if needed
__all__ = [
    'Tensor', 'Module', 'ModuleList', 'Sequential',
    'Linear', 'ReLU', 'SiLU', 'RMSNorm', 'Softmax', 'Embedding',
    'TiledLinear', 'MatFormerLinear', 'TiledEmbedding',
    'Gemma3Block', 'Gemma3Model', 'Gemma3ForMultimodalLM', 'get_cos_sin',
    'F', 'GELUTanh', 'SGD', 'Adam', 'warmup', 'nn'
]
