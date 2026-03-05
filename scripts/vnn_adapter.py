import torch
import torch.nn as nn
import torch.nn.functional as F
import vulkannn_rusted as vnn
import json
import os
import numpy as np
import types
from typing import Tuple, Union, Optional

class VNNLinear(nn.Module):
    def __init__(self, weight_vnn, bias_vnn=None, quant=False, weight_scaler=None):
        super().__init__()
        self.weight_vnn = weight_vnn
        self.bias_vnn = bias_vnn
        self.quant = quant
        self.weight_scaler = weight_scaler
        self.weight = nn.Parameter(torch.empty(1, 1), requires_grad=False)
        self.weight_vnn_t = self.weight_vnn.transpose()

    def forward(self, x):
        # x: [..., in_feat]
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        # Use constructor instead of from_numpy
        x_vnn = vnn.Tensor(data=x_flat.detach().cpu().numpy())
        res_vnn = x_vnn @ self.weight_vnn_t
        if self.bias_vnn:
            res_vnn = res_vnn + self.bias_vnn
        out_np = res_vnn.to_numpy()
        out_torch = torch.from_numpy(out_np).to(x.device).to(torch.float32)
        return out_torch.view(*orig_shape[:-1], -1)

class VNNEmbedding(nn.Module):
    def __init__(self, weight_vnn, quant=False, weight_scaler=None):
        super().__init__()
        self.weight_vnn = weight_vnn
        self.quant = quant
        self.weight_scaler = weight_scaler
        self.weight = nn.Parameter(torch.empty(1, 1), requires_grad=False)

    def forward(self, x):
        indices = x.cpu().numpy().astype("int32")
        data_np, _ = self.weight_vnn.to_numpy_no_copy()
        out_np = data_np[indices]
        return torch.from_numpy(out_np).to(x.device).to(torch.float32)

class VNNPLELayer(nn.Module):
    def __init__(self, ple_embedding_vnn, layer_idx, hidden_size=2048, ple_dim=256):
        super().__init__()
        self.ple_embedding_vnn = ple_embedding_vnn
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.ple_dim = ple_dim
        self.weight = nn.Parameter(torch.empty(1, 1), requires_grad=False)

    def forward(self, input_ids):
        indices = input_ids.cpu().numpy().astype("int32")
        full_ple_np, _ = self.ple_embedding_vnn.to_numpy_no_copy()
        start = self.layer_idx * self.ple_dim
        end = start + self.ple_dim
        ple_slice_np = full_ple_np[indices, start:end]
        ple_torch = torch.from_numpy(ple_slice_np).to(input_ids.device).to(torch.float32)
        return ple_torch

def patch_decoder_layer(layer, layer_idx, ple_layer):
    def new_forward(self, hidden_states, freqs_cis, kv_write_indices, kv_cache, mask, local_mask=None, **kwargs):
        # 1. Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
            local_mask=local_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 2. Add PLE contribution
        input_ids = kwargs.get('input_ids')
        if input_ids is None and hasattr(self, '_vnn_parent_model'):
             input_ids = getattr(self._vnn_parent_model, '_vnn_current_input_ids', None)

        if input_ids is not None:
            ple_v = ple_layer(input_ids) # [B, L, 256]
            # Handle sequence length mismatch during generation (e.g. KV cache)
            if ple_v.shape[1] > hidden_states.shape[1]:
                # Generation: only need the last token's PLE? 
                # Actually during generation hidden_states is usually [B, 1, D]
                ple_v = ple_v[:, -hidden_states.shape[1]:, :]
            hidden_states[:, :, :ple_v.shape[-1]] += ple_v
            
        hidden_states = residual + hidden_states

        # 3. MLP
        residual = hidden_states
        if hasattr(self, 'pre_feedforward_layernorm') and self.pre_feedforward_layernorm:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if hasattr(self, 'post_feedforward_layernorm') and self.post_feedforward_layernorm:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    layer.forward = types.MethodType(new_forward, layer)

def patch_gemma_model(model):
    orig_forward = model.forward
    
    def new_forward(self, *args, **kwargs):
        # Propagate input_ids to layers
        input_ids = kwargs.get('input_ids')
        if input_ids is not None:
            self._vnn_current_input_ids = input_ids
        return orig_forward(*args, **kwargs)
    
    model.forward = types.MethodType(new_forward, model)

def patch_multimodal_model(mm_model):
    orig_forward = mm_model.forward
    
    def new_forward(self, *args, **kwargs):
        # Gemma3 uses 'input_token_ids' as first arg or kwarg
        input_ids = kwargs.get('input_token_ids')
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
            
        if input_ids is not None and hasattr(self, 'model'):
            self.model._vnn_current_input_ids = input_ids
            
        return orig_forward(*args, **kwargs)
    
    mm_model.forward = types.MethodType(new_forward, mm_model)

def bridge_to_vnn(model, weights_map_path):
    with open(weights_map_path, "r") as f:
        weights_map = json.load(f)
        
    def get_vnn_tensor(name):
        if name not in weights_map:
            return None
        info = weights_map[name]
        return vnn.Tensor.from_ssd(info["path"], info["shape"])

    prefixes = ["model.language_model.", "model."]
    def get_any(suffix):
        for p in prefixes:
            t = get_vnn_tensor(p + suffix)
            if t: return t
        return None

    # Patch multimodal model for input_ids propagation
    patch_multimodal_model(model)

    # 1. Patch Embeddings & PLE
    w_embed = get_any("embed_tokens.weight")
    if w_embed:
        model.text_token_embedder = VNNEmbedding(w_embed)
        print("Patched text_token_embedder")

    w_ple = get_any("embed_tokens_per_layer.weight")
    
    # 2. Patch Decoder Layers
    inner_model = getattr(model, "model", model)
    patch_gemma_model(inner_model)
    
    layers = getattr(inner_model, "layers", None)
        
    if layers:
        for i, layer in enumerate(layers):
            layer._vnn_parent_model = inner_model
            # Patch weights
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    w_vnn = get_any(f"layers.{i}.self_attn.{proj}.weight")
                    if w_vnn: setattr(attn, proj, VNNLinear(w_vnn))
                
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    w_vnn = get_any(f"layers.{i}.mlp.{proj_name}.weight")
                    if w_vnn: setattr(mlp, proj_name, VNNLinear(w_vnn))
            
            # Patch forward with PLE
            if w_ple:
                ple_layer = VNNPLELayer(w_ple, i)
                patch_decoder_layer(layer, i, ple_layer)
            
            if (i+1) % 10 == 0 or i == 0:
                print(f"Patched layer {i}...")

    print("VNN Bridge fully applied with PLE support.")
