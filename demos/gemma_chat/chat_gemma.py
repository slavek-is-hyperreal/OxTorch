import os
import sys
# Add project root to path so we can import vulkan_nn_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import numpy as np
from transformers import AutoTokenizer
import vulkan_nn_lib as vnn
import vulkan_nn_lib.torch_shim as vtorch

class GemmaChat:
    def __init__(self, model_type="e2b", tile_size=1024):
        print(f"--- Initializing Gemma 3n {model_type.upper()} on Vulkan ---")
        self.model_type = model_type
        
        # Load tokenizer from local weights directory to avoid gated repo issues
        weights_dir_raw = f"weights_gemma_3n" if model_type == "e4b" else "weights_gemma_3n_e2b"
        print(f"Loading tokenizer from {weights_dir_raw}...")
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir_raw, local_files_only=True)
        print("Tokenizer loaded successfully.")
        
        # Gemma 3n Params (from text_config in config.json)
        num_kv_heads = 2
        head_dim = 256
        sliding_window = 512
        layer_types = None
        if model_type == "e2b":
            hidden_size = 2048
            num_heads = 8
            num_layers = 30
            intermediate_size = 8192
            vocab_size = 262400
            # E2B pattern: [sliding, sliding, sliding, sliding, full] x 6
            layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 6
        else: # e4b
            hidden_size = 2048
            num_heads = 8
            num_layers = 35
            intermediate_size = 16384
            vocab_size = 262400
            # E4B often has different patterns, defaulting to full if unsure
        
        # 1. Build Model
        print("Building model architecture...")
        self.model = vnn.Gemma3Model(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            tile_size=tile_size,
            layer_types=layer_types,
            sliding_window=sliding_window
        )
        print("Model architecture built.")
        
        # 2. Load Weights from Binaries
        weights_dir = f"vulkan_nn_lib/weights/gemma_3n_{model_type}"
        if not os.path.exists(weights_dir):
            weights_dir = "vulkan_nn_lib/weights/gemma_3n"
            
        print(f"Loading weights from {weights_dir}...")
        self.load_weights(weights_dir)
        
        # 3. RoPE Cache (uses head_dim, not hidden_size // num_heads)
        print("Initializing RoPE cache...")
        self.max_seq_len = 1024
        self.cos_np, self.sin_np = vnn.get_cos_sin(self.max_seq_len, head_dim, base=1000000.0)
        self.cos = vnn.Tensor(self.cos_np)
        self.sin = vnn.Tensor(self.sin_np)
        print("RoPE cache ready.")
        
        # 4. KV-Cache Allocation (GPU) - uses full Q heads after GQA expansion
        print(f"Allocating KV-Cache on GPU (max_seq_len={self.max_seq_len})...")
        self.caches = []
        for _ in range(num_layers):
            k_c = vnn.Tensor(None, shape=(1, self.max_seq_len, num_heads, head_dim))
            v_c = vnn.Tensor(None, shape=(1, self.max_seq_len, num_heads, head_dim))
            self.caches.append((k_c, v_c))
        print("KV-Cache allocated.")

    def load_weights(self, weights_dir):
        """Walk through model parameters and load from disk."""
        # Multi-modal models like Gemma 3 often have a "language_model" prefix
        prefix_base = "model_language_model_"
        
        def load_linear_weight(path, out_feat, in_feat):
            """Load a PyTorch Linear weight (out, in) and transpose to (in, out) for our matmul."""
            w = np.fromfile(path, dtype=np.float32).reshape(out_feat, in_feat)
            return w.T.copy()  # (in, out) for x @ W
        
        # 1. Base components (embedding is (vocab, dim) — already correct orientation)
        self.model.embed_tokens.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix_base}embed_tokens_weight.bin"), dtype=np.float32).reshape(self.model.embed_tokens.num_embeddings, self.model.embed_tokens.embedding_dim)
        self.model.norm.weight.from_disk(os.path.join(weights_dir, f"{prefix_base}norm_weight.bin"), self.model.norm.weight.shape)
        
        # 2. Layers — all Linear weights need transpose
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            layer_prefix = f"{prefix_base}layers_{i}_"
            
            layer.q_proj.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}self_attn_q_proj_weight.bin"), layer.q_proj.out_features, layer.q_proj.in_features)
            layer.k_proj.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}self_attn_k_proj_weight.bin"), layer.k_proj.out_features, layer.k_proj.in_features)
            layer.v_proj.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}self_attn_v_proj_weight.bin"), layer.v_proj.out_features, layer.v_proj.in_features)
            layer.o_proj.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}self_attn_o_proj_weight.bin"), layer.o_proj.out_features, layer.o_proj.in_features)
            
            layer.gate_proj.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}mlp_gate_proj_weight.bin"), layer.gate_proj.out_features, layer.gate_proj.in_features)
            layer.up_proj.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}mlp_up_proj_weight.bin"), layer.up_proj.out_features, layer.up_proj.in_features)
            layer.down_proj.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}mlp_down_proj_weight.bin"), layer.down_proj.out_features, layer.down_proj.in_features)
            
            layer.input_layernorm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}input_layernorm_weight.bin"), layer.input_layernorm.weight.shape)
            layer.post_attention_layernorm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}post_attention_layernorm_weight.bin"), layer.post_attention_layernorm.weight.shape)

        # Head: tied to embedding or separate
        lm_head_path = os.path.join(weights_dir, f"{prefix_base}lm_head_weight.bin")
        if os.path.exists(lm_head_path):
            self.model.lm_head.weight_ram = load_linear_weight(lm_head_path, self.model.lm_head.out_features, self.model.lm_head.in_features)
        else:
            print("lm_head not found, tying to embed_tokens (weight sharing).")
            # embed is (vocab, hidden). lm_head needs (hidden, vocab) for x @ W.
            # embed is already stored as (vocab, hidden) = PT's (out=vocab, in=hidden).
            # So embed.T = (hidden, vocab) which IS what we need for x @ W.
            self.model.lm_head.weight_ram = self.model.embed_tokens.weight_ram.T.copy()
        print("All internal language model weights mapped.")

    def softcap_logits(self, logits_np):
        """Gemma final_logit_softcapping: tanh(logits / cap) * cap"""
        cap = 30.0
        return np.tanh(logits_np / cap) * cap

    def chat_loop(self):
        print("\n--- Gemma 3n Vulkan Chat (Type 'exit' to quit) ---")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: break
            
            chat = [{"role": "user", "content": user_input}]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
            x = vtorch.from_numpy(input_ids.astype(np.int32))
            
            print(f"[DEBUG] Prompt tokens: {input_ids.shape[1]}")
            print("Gemma: ", end="", flush=True)
            
            # Reset KV cache for each turn
            for k_c, v_c in self.caches:
                k_c.arr.fill(0.0)
                v_c.arr.fill(0.0)
            
            # Prefill phase
            pos_offset = 0
            seq_len = x.shape[1]
            logits = self.model(x, self.cos, self.sin, caches=self.caches, pos_offset=pos_offset)
            pos_offset += seq_len
            
            # Apply softcapping and get logits for last token
            raw_logits = logits.to_numpy()[0, -1, :]
            capped_logits = self.softcap_logits(raw_logits)
            
            # Debug: show top-5 predicted tokens
            top5 = np.argsort(capped_logits)[-5:][::-1]
            print(f"\n[DEBUG] Top-5 after prefill: {[(int(t), self.tokenizer.decode([int(t)]), float(capped_logits[t])) for t in top5]}")
            
            next_token = int(np.argmax(capped_logits))
            
            for step in range(50):
                if next_token == self.tokenizer.eos_token_id: break
                
                print(self.tokenizer.decode([next_token]), end="", flush=True)
                
                # Single token step
                x = vtorch.from_numpy(np.array([[next_token]], dtype=np.int32))
                logits = self.model(x, self.cos, self.sin, caches=self.caches, pos_offset=pos_offset)
                
                raw_logits = logits.to_numpy()[0, -1, :]
                capped_logits = self.softcap_logits(raw_logits)
                next_token = int(np.argmax(capped_logits))
                pos_offset += 1
                
            print("\n")

if __name__ == "__main__":
    # Start with E2B to save RAM as requested
    chat = GemmaChat(model_type="e2b")
    chat.chat_loop()
