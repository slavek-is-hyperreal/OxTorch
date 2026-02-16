import numpy as np
import taichi as ti
from ..tensor import Tensor
from .base import Module, ModuleList
from .layers import Linear, RMSNorm, Embedding, Softmax
from .tiled import MatFormerLinear, TiledEmbedding
from ..functional import GELUTanh
from .. import kernels as K

class RoPE(Module):
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, pos_offset: int = 0) -> Tensor:
        B, L, H, D = x.shape
        K.k_rope(x.arr, cos.arr, sin.arr, B, L, H, D, pos_offset)
        return x

class GaussianTopK(Module):
    def __init__(self, sparsity=0.0):
        super().__init__()
        self.sparsity = sparsity
        if sparsity >= 0.95: self.std_multiplier = 1.64485362695
        elif sparsity > 0.0:
            import scipy.stats
            self.std_multiplier = float(scipy.stats.norm.ppf(sparsity))
        else: self.std_multiplier = -1e10
            
    def forward(self, x: Tensor) -> Tensor:
        if self.sparsity > 0.0:
            shape = x.shape
            N = shape[-1]
            K.k_gaussian_topk(x.arr, x.total_size, N, self.std_multiplier)
        return x

class Gemma3nLaurelBlock(Module):
    def __init__(self, hidden_size, laurel_rank, eps=1e-6):
        super().__init__()
        self.linear_left = Linear(hidden_size, laurel_rank, bias=False)
        self.linear_right = Linear(laurel_rank, hidden_size, bias=False)
        self.post_laurel_norm = RMSNorm(hidden_size, eps=eps)
        
    def forward(self, hidden_states: Tensor) -> Tensor:
        res = self.linear_left(hidden_states)
        res = self.linear_right(res)
        res = self.post_laurel_norm(res)
        return res

class Gemma3nAltUp(Module):
    def __init__(self, hidden_size, num_inputs, active_idx=0, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs
        self.active_idx = active_idx
        self.modality_router = Linear(hidden_size, num_inputs, bias=False)
        self.router_norm = RMSNorm(hidden_size, eps=eps)
        self.router_input_scale = hidden_size**-1.0
        self.prediction_coefs = Linear(num_inputs, num_inputs**2, bias=False)
        self.correction_coefs = Linear(num_inputs, num_inputs, bias=False)
        self.correct_output_scale = Tensor(np.zeros(hidden_size, dtype=np.float32))

    def compute_router_modalities(self, x: Tensor) -> Tensor:
        router_inputs = x.clone()
        self.router_norm(router_inputs)
        K.k_scale(router_inputs.arr, self.router_input_scale, router_inputs.total_size)
        routed = self.modality_router(router_inputs)
        K.k_tanh(routed.arr, routed.total_size)
        return routed

    def predict(self, hidden_states: Tensor) -> Tensor:
        B, L, D = hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3]
        active_slice = Tensor(None, shape=(B, L, D))
        K.k_copy_offset(hidden_states.arr, self.active_idx * B * L * D, active_slice.arr, 0, active_slice.total_size)
        modalities = self.compute_router_modalities(active_slice)
        all_coefs = self.prediction_coefs(modalities)
        out = Tensor(None, shape=hidden_states.shape)
        K.k_altup_predict(hidden_states.arr, all_coefs.arr, out.arr, self.num_inputs, B, L, D)
        return out

    def correct(self, predictions: Tensor, activated: Tensor) -> Tensor:
        modalities = self.compute_router_modalities(activated)
        all_coefs = self.correction_coefs(modalities)
        # Using a simple numpy approach for the +1.0 for now
        coefs_np = all_coefs.to_numpy() + 1.0
        all_coefs_gpu = Tensor(coefs_np)
        K.k_altup_correct(predictions.arr, activated.arr, all_coefs_gpu.arr, self.num_inputs, activated.shape[0], activated.shape[1], activated.shape[2], self.active_idx)
        return predictions

    def scale_corrected_output(self, corrected: Tensor) -> Tensor:
        K.k_altup_scale_correction(corrected.arr, self.correct_output_scale.arr, corrected.shape[0], corrected.shape[1], corrected.shape[2])
        return corrected

class Gemma3nPLE(Module):
    def __init__(self, hidden_size, ple_dim, eps=1e-6):
        super().__init__()
        self.per_layer_input_gate = Linear(hidden_size, ple_dim, bias=False)
        self.per_layer_projection = Linear(ple_dim, hidden_size, bias=False)
        self.post_per_layer_input_norm = RMSNorm(hidden_size, eps=eps)
        self.activation = GELUTanh()
        
    def forward(self, active_state: Tensor, per_layer_input: Tensor, full_state: Tensor):
        gate = self.per_layer_input_gate(active_state)
        self.activation(gate)
        K.k_mul(gate.arr, per_layer_input.arr, gate.total_size, per_layer_input.total_size)
        proj = self.per_layer_projection(gate)
        self.post_per_layer_input_norm(proj)
        S, B, L, D = full_state.shape
        K.k_add_to_slices(proj.arr, full_state.arr, 1, S, B, L, D)

class Gemma3Block(Module):
    def __init__(self, hidden_size=2048, num_heads=8, num_kv_heads=2, head_dim=256, 
                 intermediate_size=8192, tile_size=1024, layer_type="full_attention", window=512,
                 altup_num_inputs=4, laurel_rank=64, ple_dim=256, sparsity=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.window = window if layer_type == "sliding_attention" else 0
        self.active_idx = 0
        self.altup_correct_scale = True

        self.altup = Gemma3nAltUp(hidden_size, altup_num_inputs, active_idx=self.active_idx)
        self.laurel = Gemma3nLaurelBlock(hidden_size, laurel_rank)
        self.ple = Gemma3nPLE(hidden_size, ple_dim)
        self.sparsity_gate = GaussianTopK(sparsity)

        self.q_proj = MatFormerLinear(hidden_size, num_heads * head_dim, bias=False, tile_size=tile_size)
        self.k_proj = MatFormerLinear(hidden_size, num_kv_heads * head_dim, bias=False, tile_size=tile_size)
        self.v_proj = MatFormerLinear(hidden_size, num_kv_heads * head_dim, bias=False, tile_size=tile_size)
        self.o_proj = MatFormerLinear(num_heads * head_dim, hidden_size, bias=False, tile_size=tile_size)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.pre_feedforward_layernorm = RMSNorm(hidden_size)
        self.post_feedforward_layernorm = RMSNorm(hidden_size)

        self.gate_proj = MatFormerLinear(hidden_size, intermediate_size, bias=False, tile_size=tile_size)
        self.up_proj   = MatFormerLinear(hidden_size, intermediate_size, bias=False, tile_size=tile_size)
        self.down_proj = MatFormerLinear(intermediate_size, hidden_size, bias=False, tile_size=tile_size)
        
        self.rope = RoPE()
        self.activation = GELUTanh()

    def forward(self, hidden_states: Tensor, cos: Tensor, sin: Tensor, 
                 ple_input: Tensor, k_cache: Tensor = None, v_cache: Tensor = None, 
                 pos_offset: int = 0, active_intermediate_size=None) -> Tensor:
        predictions = self.altup.predict(hidden_states)
        S, B, L_new, D_hidden = predictions.shape
        active_prediction = Tensor(None, shape=(B, L_new, D_hidden))
        K.k_extract_slice(predictions.arr, active_prediction.arr, self.active_idx, B, L_new, D_hidden)
        
        attn_in = self.input_layernorm(active_prediction.clone())
        laurel_output = self.laurel(attn_in)
        
        q = self.q_proj(attn_in)
        k = self.k_proj(attn_in)
        v = self.v_proj(attn_in)
        
        q.shape = (B, L_new, self.num_heads, self.head_dim)
        k.shape = (B, L_new, self.num_kv_heads, self.head_dim)
        v.shape = (B, L_new, self.num_kv_heads, self.head_dim)

        self.q_norm(q)
        self.k_norm(k)
        self.rope(q, cos, sin, pos_offset)
        self.rope(k, cos, sin, pos_offset)
        
        # KV expansion and attention logic would go here, simplified for now
        # ... matching existing code structure ...
        
        self.altup.correct(predictions, active_prediction)
        K.k_copy_offset(active_prediction.arr, 0, predictions.arr, self.active_idx * B * L_new * D_hidden, active_prediction.total_size)
        self.ple(active_prediction, ple_input, predictions)
        return predictions

class Gemma3Model(Module):
    def __init__(self, num_layers=30, hidden_size=2048, num_heads=8, num_kv_heads=2, 
                 head_dim=256, intermediate_size=8192, vocab_size=262400, tile_size=1024,
                 layer_types=None, sliding_window=512,
                 altup_num_inputs=4, laurel_rank=64, ple_dim=256, sparsity_pattern=None):
        super().__init__()
        self.embed_tokens = TiledEmbedding(vocab_size, hidden_size)
        self.altup_projections = ModuleList([Linear(hidden_size, hidden_size, bias=False) for _ in range(altup_num_inputs-1)])
        self.altup_unembed_projections = ModuleList([Linear(hidden_size, hidden_size, bias=False) for _ in range(altup_num_inputs-1)])
        self.layers = ModuleList([Gemma3Block(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, tile_size, 
                                             altup_num_inputs=altup_num_inputs, laurel_rank=laurel_rank, ple_dim=ple_dim) 
                                 for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = MatFormerLinear(hidden_size, vocab_size, bias=False, tile_size=tile_size)

    def forward(self, input_ids: Tensor, cos: Tensor, sin: Tensor, ple_inputs: Tensor, caches=None, pos_offset=0) -> Tensor:
        # Simplified for refactoring
        x0 = self.embed_tokens(input_ids)
        K.k_scale(x0.arr, float(np.sqrt(x0.shape[-1])), x0.total_size)
        return x0 # Placeholder for full flow

class Gemma3ForMultimodalLM(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_model = Gemma3Model(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.intermediate_size[0] if isinstance(config.intermediate_size, list) else config.intermediate_size,
            vocab_size=config.vocab_size,
            altup_num_inputs=config.altup_num_inputs,
            laurel_rank=config.laurel_rank,
            ple_dim=config.hidden_size_per_layer_input
        )

    def forward(self, input_ids: Tensor, cos: Tensor, sin: Tensor, ple_inputs: Tensor, caches=None, pos_offset=0) -> Tensor:
        return self.language_model(input_ids, cos, sin, ple_inputs, caches, pos_offset)

def get_cos_sin(seq_len, head_dim, base=1000000.0):
    cos = np.zeros((seq_len, head_dim), dtype=np.float32)
    sin = np.zeros((seq_len, head_dim), dtype=np.float32)
    for pos in range(seq_len):
        for i in range(0, head_dim, 2):
            theta = 1.0 / (base ** (i / head_dim))
            cos[pos, i] = np.cos(pos * theta)
            cos[pos, i+1] = np.cos(pos * theta)
            sin[pos, i] = np.sin(pos * theta)
            sin[pos, i+1] = np.sin(pos * theta)
    return cos.flatten(), sin.flatten()
