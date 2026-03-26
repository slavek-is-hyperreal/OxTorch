import torch
import torch.nn as nn
from safetensors.torch import load_file
import oxtorch as vnn
import numpy as np
import time
import math
from transformers import AutoTokenizer

def unpack_bitnet_weights(w_packed, out_features, in_features):
    # w_packed: (out, in//4)
    # Each byte contains 4 weights: (w0 | w1<<2 | w2<<4 | w3<<6)
    w_np = w_packed.cpu().to(torch.uint8).numpy()
    
    # Pre-allocate the full matrix
    res = np.zeros((out_features, in_features), dtype=np.int8)
    
    # Unpack 4 weights per byte
    # Mapping: (val & 3) - 1 => 0->-1, 1->0, 2->1
    for i in range(4):
        # The broadcasting error confirmed packing is row-wise: (out//4, in)
        res[i::4, :] = ((w_np >> (2 * i)) & 3).astype(np.int8) - 1
class BITNETRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, weight=None):
        super().__init__()
        self.eps = eps
        self.weight = weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Activation Quantization helper
def quant_activation(x, num_bits=8):
    # symmetric per-token quantization
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    x_q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale

class VNNBitLinear:
    def __init__(self, weights_dict, p_prefix, hidden_size, out_features, in_features, device="vga"):
        self.device = device
        
        # 1. Unpack weights (2-bit packed ternary {-1, 0, 1})
        weight_packed = weights_dict[f"{p_prefix}.weight"]
        scale = weights_dict[f"{p_prefix}.weight_scale"]
        
        # Reverting to cyclic interleaving which showed more promise
        res = torch.zeros((out_features, in_features), dtype=torch.int8)
        w_np = weight_packed.cpu().numpy().astype(np.uint8)
        
        for i in range(4):
            # Interleave rows: row i, i+4, i+8...
            # This byte-access pattern is common in packed HF models
            res[i::4, :] = torch.from_numpy(((w_np >> (2 * i)) & 3).astype(np.int8) - 1)
            
        w_np = res.cpu().float().numpy()
        
        # 3. Create VNN Tensor and convert to BitNet2 packed format
        self.w_vnn = vnn.Tensor(w_np, dtype="int8").to_bitnet("bitnet2").to(device)
        
        # VNN BitLinear requires scale as a vector of size out_features
        # AutoBitLinear configuration MULTIPLIES by weight_scale.
        # VNN kernel also MULTIPLIES. So we use the value directly.
        s_base = scale.float().cpu().numpy()
        s_np = np.full((out_features,), s_base.item(), dtype=np.float32)
            
        self.s_vnn = vnn.Tensor(s_np, dtype="float32").to(device)

    def __call__(self, x_torch):
        # x_torch: (bs, seq_len, hidden) or (seq_len, hidden)
        # VNN BitLinear expects (M, K)
        orig_shape = x_torch.shape
        x_2d = x_torch.reshape(-1, orig_shape[-1])
        
        x_q, scale_x = quant_activation(x_2d)
        
        # VNN requires float32 numpy in constructor, then casts to int8
        x_vnn = vnn.Tensor(x_q.cpu().float().numpy(), dtype="int8").to(self.device)
        
        # Execute BitLinear kernel
        res_vnn = x_vnn.bit_linear(self.w_vnn, self.s_vnn)
        
        # Rescale back to float32 using input scale and reshape back
        res_torch = res_vnn.to_torch() / scale_x
        return res_torch.reshape(*orig_shape[:-1], -1)

class BitNetLayer:
    def __init__(self, weights, layer_idx, config, device="vga"):
        p = f"model.layers.{layer_idx}"
        h = config["hidden_size"]
        i_size = config["intermediate_size"]
        
        kv_h = (config["num_key_value_heads"] * h) // config["num_attention_heads"]
        
        # Projections
        self.q_proj = VNNBitLinear(weights, f"{p}.self_attn.q_proj", h, h, h, device=device)
        self.k_proj = VNNBitLinear(weights, f"{p}.self_attn.k_proj", h, kv_h, h, device=device)
        self.v_proj = VNNBitLinear(weights, f"{p}.self_attn.v_proj", h, kv_h, h, device=device)
        self.o_proj = VNNBitLinear(weights, f"{p}.self_attn.o_proj", h, h, h, device=device)
        
        # MLP
        self.gate_proj = VNNBitLinear(weights, f"{p}.mlp.gate_proj", h, i_size, h, device=device)
        self.up_proj = VNNBitLinear(weights, f"{p}.mlp.up_proj", h, i_size, h, device=device)
        self.down_proj = VNNBitLinear(weights, f"{p}.mlp.down_proj", i_size, h, i_size, device=device)
        
        # Norms
        self.input_layernorm = BITNETRMSNorm(h, eps=config["rms_norm_eps"], weight=weights[f"{p}.input_layernorm.weight"])
        self.post_attention_layernorm = BITNETRMSNorm(h, eps=config["rms_norm_eps"], weight=weights[f"{p}.post_attention_layernorm.weight"])
        self.attn_sub_norm = BITNETRMSNorm(h, eps=config["rms_norm_eps"], weight=weights[f"{p}.self_attn.attn_sub_norm.weight"])
        self.ffn_sub_norm = BITNETRMSNorm(i_size, eps=config["rms_norm_eps"], weight=weights[f"{p}.mlp.ffn_sub_norm.weight"])
        
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = h // self.num_heads

    def forward(self, x, cos, sin):
        # x: (1, seq_len, hidden)
        # cos, sin: (1, seq_len, 1, head_dim)
        residual = x
        x_norm = self.input_layernorm(x)
        
        # Projections
        q = self.q_proj(x_norm).view(1, -1, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(1, -1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_norm).view(1, -1, self.num_kv_heads, self.head_dim)
        
        # RoPE (broadcast along heads)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        bs, seq_len, _, _ = q.shape # Get batch size and sequence length from q
        
        # GQA repeat
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            
        # SDPA expects heads first: (bs, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn_out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Transpose back: (bs, seq_len, n_heads, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(bs, seq_len, -1)
        
        # SubLN
        attn_out = self.attn_sub_norm(attn_out)
        x = residual + self.o_proj(attn_out)
        
        # 2. MLP
        residual = x
        x_norm = self.post_attention_layernorm(x)
        gate = self.gate_proj(x_norm)
        up = self.up_proj(x_norm)
        inner = self.ffn_sub_norm(torch.relu(gate)**2 * up)
        x = residual + self.down_proj(inner)
        
        return x

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def get_rope_cos_sin(seq_len, dim, theta=500000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    # Shape for broadcasting: (1, seq_len, 1, head_dim)
    return emb.cos().unsqueeze(0).unsqueeze(2), emb.sin().unsqueeze(0).unsqueeze(2)

class VNNBitNet:
    def __init__(self, model_path, device="vga"):
        print(f"Loading BitNet-2B from {model_path} on {device}...")
        self.config = {
            "hidden_size": 2560,
            "intermediate_size": 6912,
            "num_attention_heads": 20,
            "num_key_value_heads": 5,
            "num_hidden_layers": 30,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "vocab_size": 128256
        }
        weights = load_file(f"{model_path}/model.safetensors")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
        
        self.embed = weights["model.embed_tokens.weight"].to(torch.float32)
        self.layers = [BitNetLayer(weights, i, self.config, device) for i in range(self.config["num_hidden_layers"])]
        self.norm = BITNETRMSNorm(self.config["hidden_size"], eps=self.config["rms_norm_eps"], weight=weights["model.norm.weight"])
        
        if "lm_head.weight" in weights:
            self.lm_head = weights["lm_head.weight"].to(torch.float32)
        else:
            self.lm_head = self.embed
        print("Model loaded successfully.")

    def generate(self, prompt, max_new_tokens=64):
        # Llama-3 Tokenizer usually needs BOS (128000)
        self.tokenizer.padding_side = "left"
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if input_ids[0] != self.tokenizer.bos_token_id:
             input_ids = [self.tokenizer.bos_token_id] + input_ids
             
        generated = list(input_ids)
        
        print(f"\nPrompt: {prompt}")
        print("Assistant: ", end="", flush=True)
        
        head_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
        cos_table, sin_table = get_rope_cos_sin(4096, head_dim, self.config["rope_theta"])
        
        for t in range(max_new_tokens):
            seq_len = len(generated)
            x = self.embed[generated].unsqueeze(0) # (1, seq_len, hidden)
            
            cos = cos_table[:, :seq_len, :, :]
            sin = sin_table[:, :seq_len, :, :]
            
            with torch.no_grad():
                for layer in self.layers:
                    x = layer.forward(x, cos, sin)
                
                # We only need the last token's logits
                x_last = self.norm(x[:, -1, :])
                logits = torch.matmul(x_last, self.lm_head.T)
                
                next_id = torch.argmax(logits, dim=-1).item()
                generated.append(next_id)
            
            token = self.tokenizer.decode([next_id])
            print(token, end="", flush=True)
            if next_id == self.tokenizer.eos_token_id: break
            
        print("\n\n(Inference complete using OxTorch/VNN engine)")
            
        print("\n\n(Inference complete using OxTorch/VNN engine)")

if __name__ == "__main__":
    model = VNNBitNet("/my_data/gaussian_room/models/bitnet-2B-ternary", device="cpu")
    model.generate("What is 1.58-bit quantization? Provide a concise explanation.")
