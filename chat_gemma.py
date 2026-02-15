import os
import time
import numpy as np
from transformers import AutoTokenizer
import vulkan_nn_lib as vnn
import vulkan_nn_lib.torch_shim as torch

class GemmaChat:
    def __init__(self, model_type="e2b", tile_size=1024):
        print(f"--- Initializing Gemma 3n {model_type.upper()} on Vulkan ---")
        self.model_type = model_type
        self.model_id = f"google/gemma-3n-{model_type}-it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Gemma 3n Params (Match HF config)
        if model_type == "e2b":
            hidden_size = 2048
            num_heads = 16
            num_layers = 35 # Gemma 3 nano often has 35 layers
            intermediate_size = 8192
            sub_model_size = 1024 # The "2B" slice
        else: # e4b
            hidden_size = 2048
            num_heads = 16
            num_layers = 35
            intermediate_size = 8192
            sub_model_size = 2048 # The "4B" slice
            
        self.sub_model_size = sub_model_size
        
        # 1. Build Model
        self.model = vnn.Gemma3Model(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            tile_size=tile_size
        )
        
        # 2. Load Weights from Binaries
        weights_dir = f"vulkan_nn_lib/weights/gemma_3n_{model_type}"
        if not os.path.exists(weights_dir):
            # Fallback to general folder if not split
            weights_dir = "vulkan_nn_lib/weights/gemma_3n"
            
        print(f"Loading weights from {weights_dir}...")
        self.load_weights(weights_dir)
        
        # 3. RoPE Cache
        self.max_seq_len = 1024
        self.cos_np, self.sin_np = vnn.get_cos_sin(self.max_seq_len, hidden_size // num_heads)
        self.cos = vnn.Tensor(self.cos_np)
        self.sin = vnn.Tensor(self.sin_np)
        
        # 4. KV-Cache Allocation (GPU)
        self.caches = []
        for _ in range(num_layers):
            k_c = vnn.Tensor(None, shape=(1, self.max_seq_len, num_heads, hidden_size // num_heads))
            v_c = vnn.Tensor(None, shape=(1, self.max_seq_len, num_heads, hidden_size // num_heads))
            self.caches.append((k_c, v_c))

    def load_weights(self, weights_dir):
        """Walk through model parameters and load from disk."""
        # 1. Base components
        self.model.embed_tokens.weight.from_disk(os.path.join(weights_dir, "model_embed_tokens_weight.bin"), self.model.embed_tokens.weight.shape)
        self.model.norm.weight.from_disk(os.path.join(weights_dir, "model_norm_weight.bin"), self.model.norm.weight.shape)
        
        # 2. Layers
        for i in range(len(self.model.layers)):
            layer = self.model.layers[i]
            prefix = f"model_layers_{i}_"
            
            # Use weights_dir + bin names from converter
            layer.q_proj.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix}self_attn_q_proj_weight.bin"), dtype=np.float32).reshape(layer.q_proj.in_features, layer.q_proj.out_features)
            layer.k_proj.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix}self_attn_k_proj_weight.bin"), dtype=np.float32).reshape(layer.k_proj.in_features, layer.k_proj.out_features)
            layer.v_proj.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix}self_attn_v_proj_weight.bin"), dtype=np.float32).reshape(layer.v_proj.in_features, layer.v_proj.out_features)
            layer.o_proj.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix}self_attn_o_proj_weight.bin"), dtype=np.float32).reshape(layer.o_proj.in_features, layer.o_proj.out_features)
            
            layer.gate_proj.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix}mlp_gate_proj_weight.bin"), dtype=np.float32).reshape(layer.gate_proj.in_features, layer.gate_proj.out_features)
            layer.up_proj.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix}mlp_up_proj_weight.bin"), dtype=np.float32).reshape(layer.up_proj.in_features, layer.up_proj.out_features)
            layer.down_proj.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix}mlp_down_proj_weight.bin"), dtype=np.float32).reshape(layer.down_proj.in_features, layer.down_proj.out_features)
            
            layer.input_layernorm.weight.from_disk(os.path.join(weights_dir, f"{prefix}input_layernorm_weight.bin"), layer.input_layernorm.weight.shape)
            layer.post_attention_layernorm.weight.from_disk(os.path.join(weights_dir, f"{prefix}post_attention_layernorm_weight.bin"), layer.post_attention_layernorm.weight.shape)

        self.model.lm_head.weight_ram = np.fromfile(os.path.join(weights_dir, "lm_head_weight.bin"), dtype=np.float32).reshape(self.model.lm_head.in_features, self.model.lm_head.out_features)
        print("All weights mapped.")

    def chat_loop(self):
        print("\n--- Gemma 3n Vulkan Chat (Type 'exit' to quit) ---")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]: break
            
            chat = [{"role": "user", "content": user_input}]
            prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]
            x = torch.from_numpy(input_ids.astype(np.int32))
            
            print("Gemma: ", end="", flush=True)
            
            # Prefill phase
            pos_offset = 0
            seq_len = x.shape[1]
            logits = self.model(x, self.cos, self.sin, caches=self.caches, pos_offset=pos_offset, sub_model_size=self.sub_model_size)
            pos_offset += seq_len
            
            # Decode loop
            next_token = np.argmax(logits.to_numpy()[0, -1, :])
            
            for _ in range(50):
                if next_token == self.tokenizer.eos_token_id: break
                
                print(self.tokenizer.decode([next_token]), end="", flush=True)
                
                # Single token step
                x = torch.from_numpy(np.array([[next_token]], dtype=np.int32))
                logits = self.model(x, self.cos, self.sin, caches=self.caches, pos_offset=pos_offset, sub_model_size=self.sub_model_size)
                
                next_token = np.argmax(logits.to_numpy()[0, -1, :])
                pos_offset += 1
                
            print("\n")

if __name__ == "__main__":
    # Start with E2B since user requested small model
    chat = GemmaChat(model_type="e2b")
    chat.chat_loop()
