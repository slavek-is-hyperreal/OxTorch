from .tensor import Tensor
from .modules.base import Module, ModuleList, Sequential
from .modules.layers import Linear, ReLU, SiLU, RMSNorm, Softmax, Embedding
from .modules.tiled import TiledLinear, MatFormerLinear, TiledEmbedding
from .modules.models import Gemma3Block, Gemma3Model, Gemma3ForMultimodalLM, get_cos_sin
from .functional import F, GELUTanh
from .optimizers import SGD, Adam
from .kernels import warmup

# Re-expose nn and F namespaces for backward compatibility
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
    TiledEmbedding = TiledEmbedding

__all__ = [
    'Tensor', 'Module', 'ModuleList', 'Sequential',
    'Linear', 'ReLU', 'SiLU', 'RMSNorm', 'Softmax', 'Embedding',
    'TiledLinear', 'MatFormerLinear', 'TiledEmbedding',
    'Gemma3Block', 'Gemma3Model', 'Gemma3ForMultimodalLM', 'get_cos_sin',
    'F', 'GELUTanh', 'SGD', 'Adam', 'warmup', 'nn'
]
