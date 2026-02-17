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
        
        # Load tokenizer from local weights directory
        base_weights = os.path.join(os.path.dirname(__file__), "weights")
        weights_dir_raw = os.path.join(base_weights, f"weights_gemma_3n" if model_type == "e4b" else "weights_gemma_3n_e2b")
        print(f"Loading tokenizer from {weights_dir_raw}...")
        self.tokenizer = AutoTokenizer.from_pretrained(weights_dir_raw, local_files_only=True)
        print("Tokenizer loaded successfully.")
        
        # Gemma 3n Params (from text_config in config.json)
        num_kv_heads = 2
        head_dim = 256
        sliding_window = 512
        altup_num_inputs = 4
        laurel_rank = 64
        ple_dim = 256
        layer_types = None
        sparsity_pattern = None
        
        if model_type == "e2b":
            hidden_size = 2048
            num_heads = 8
            num_layers = 30
            intermediate_size = 8192
            vocab_size = 262400 # Real Gemma 3n vocab size
            # E2B pattern: [sliding, sliding, sliding, sliding, full] x 6
            layer_types = ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 6
            # Early layers have sparsity
            sparsity_pattern = [0.95] * 10 + [0.0] * 20
        else: # e4b
            hidden_size = 2048
            num_heads = 8
            num_layers = 35
            intermediate_size = 16384
            vocab_size = 262400
            layer_types = ["full_attention"] * num_layers
            sparsity_pattern = [0.95] * 12 + [0.0] * 23
        
        # 1. Build Model
        print("Building model architecture...")
        class Config: pass
        config = Config()
        config.num_hidden_layers = num_layers
        config.hidden_size = hidden_size
        config.num_attention_heads = num_heads
        config.num_key_value_heads = num_kv_heads
        config.head_dim = head_dim
        config.intermediate_size = intermediate_size
        config.vocab_size = vocab_size
        config.layer_types = layer_types
        config.sliding_window = sliding_window
        config.altup_num_inputs = altup_num_inputs
        config.laurel_rank = laurel_rank
        config.hidden_size_per_layer_input = ple_dim
        
        self.model_wrapper = vnn.Gemma3ForMultimodalLM(config)
        self.model = self.model_wrapper.language_model
        print("Model architecture built.")
        
        # 2. Load Weights from Binaries
        weights_dir = os.path.join(os.path.dirname(__file__), "weights", f"gemma_3n_{model_type}")
        if not os.path.exists(weights_dir):
            # Fallback to subdirectory inside weights_dir_raw if binaries are there
            weights_dir = os.path.join(weights_dir_raw, "vnn_weights")
            
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
        
        # 1. Base components
        self.model.embed_tokens.weight_ram = np.fromfile(os.path.join(weights_dir, f"{prefix_base}embed_tokens_weight.bin"), dtype=np.float32).reshape(self.model.embed_tokens.num_embeddings, self.model.embed_tokens.embedding_dim)
        self.model.norm.weight.from_disk(os.path.join(weights_dir, f"{prefix_base}norm_weight.bin"), self.model.norm.weight.shape)
        
        # 3n specialized global components
        for i in range(len(self.model.altup_projections)):
            self.model.altup_projections[i].weight_ram = load_linear_weight(os.path.join(weights_dir, f"{prefix_base}altup_projections_{i}_weight.bin"), self.model.hidden_size, self.model.hidden_size)
            self.model.altup_unembed_projections[i].weight_ram = load_linear_weight(os.path.join(weights_dir, f"{prefix_base}altup_unembed_projections_{i}_weight.bin"), self.model.hidden_size, self.model.hidden_size)

        # PLE Table (Large! Load to RAM, not VRAM)
        ple_path = os.path.join(weights_dir, f"{prefix_base}embed_tokens_per_layer_weight.bin")
        print(f"Mapping PLE Table from {ple_path}...")
        self.ple_vocab_size = 262144
        self.ple_table = np.memmap(ple_path, dtype='float32', mode='r', shape=(self.ple_vocab_size, len(self.model.layers), self.model.ple_dim))
        
        # 2. Layers
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
            layer.pre_feedforward_layernorm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}pre_feedforward_layernorm_weight.bin"), layer.pre_feedforward_layernorm.weight.shape)
            layer.post_feedforward_layernorm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}post_feedforward_layernorm_weight.bin"), layer.post_feedforward_layernorm.weight.shape)

            layer.q_norm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}self_attn_q_norm_weight.bin"), layer.q_norm.weight.shape)
            layer.k_norm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}self_attn_k_norm_weight.bin"), layer.k_norm.weight.shape)

            # 3n Layer components
            layer.altup.modality_router.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}altup_modality_router_weight.bin"), layer.altup.num_inputs, layer.hidden_size)
            layer.altup.prediction_coefs.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}altup_prediction_coefs_weight.bin"), layer.altup.num_inputs**2, layer.altup.num_inputs)
            layer.altup.correction_coefs.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}altup_correction_coefs_weight.bin"), layer.altup.num_inputs, layer.altup.num_inputs)
            layer.altup.router_norm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}altup_router_norm_weight.bin"), layer.altup.router_norm.weight.shape)
            layer.altup.correct_output_scale.from_disk(os.path.join(weights_dir, f"{layer_prefix}altup_correct_output_scale.bin"), layer.altup.correct_output_scale.shape)

            layer.laurel.linear_left.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}laurel_linear_left_weight.bin"), layer.laurel.linear_left.out_features, layer.laurel.linear_left.in_features)
            layer.laurel.linear_right.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}laurel_linear_right_weight.bin"), layer.laurel.linear_right.out_features, layer.laurel.linear_right.in_features)
            layer.laurel.post_laurel_norm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}laurel_post_laurel_norm_weight.bin"), layer.laurel.post_laurel_norm.weight.shape)

            layer.ple.per_layer_input_gate.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}per_layer_input_gate_weight.bin"), layer.ple.per_layer_input_gate.out_features, layer.ple.per_layer_input_gate.in_features)
            layer.ple.per_layer_projection.weight_ram = load_linear_weight(os.path.join(weights_dir, f"{layer_prefix}per_layer_projection_weight.bin"), layer.ple.per_layer_projection.out_features, layer.ple.per_layer_projection.in_features)
            layer.ple.post_per_layer_input_norm.weight.from_disk(os.path.join(weights_dir, f"{layer_prefix}post_per_layer_input_norm_weight.bin"), layer.ple.post_per_layer_input_norm.weight.shape)

        lm_head_path = os.path.join(weights_dir, f"{prefix_base}lm_head_weight.bin")
        if os.path.exists(lm_head_path):
            self.model.lm_head.weight_ram = load_linear_weight(lm_head_path, self.model.lm_head.out_features, self.model.lm_head.in_features)
        else:
            print("lm_head not found, tying to embed_tokens (weight sharing).")
            self.model.lm_head.weight_ram = self.model.embed_tokens.weight_ram.T.copy()
        print("All internal language model weights mapped.")
        
        # 3. Vision/Audio (Conditional loading support as per gemma3n_introduction.md)
        print("Conditional loading: Vision/Audio parameters bypassed.")

    def softcap_logits(self, logits_np):
        """Gemma final_logit_softcapping: tanh(logits / cap) * cap."""
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
            seq_len = input_ids.shape[1]
            
            # PLE Indexing: [B, L, num_layers, ple_dim]
            # Handle vocab mismatch: tokens >= 262144 get zeros for PLE
            ple_np = np.zeros((input_ids.shape[0], seq_len, len(self.model.layers), self.model.ple_dim), dtype=np.float32)
            flat_ids = input_ids.flatten()
            for i, tid in enumerate(flat_ids):
                if tid < self.ple_vocab_size:
                    ple_np.reshape(-1, len(self.model.layers), self.model.ple_dim)[i] = self.ple_table[tid]
            
            ple_torch = vtorch.from_numpy(ple_np)
            
            logits = self.model_wrapper(x, self.cos, self.sin, ple_inputs=ple_torch, caches=self.caches, pos_offset=pos_offset)
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
                ple_token_np = np.zeros((1, 1, len(self.model.layers), self.model.ple_dim), dtype=np.float32)
                if next_token < self.ple_vocab_size:
                    ple_token_np[0, 0] = self.ple_table[next_token]
                ple_token_torch = vtorch.from_numpy(ple_token_np)
                
                logits = self.model_wrapper(x, self.cos, self.sin, ple_inputs=ple_token_torch, caches=self.caches, pos_offset=pos_offset)
                
                raw_logits = logits.to_numpy()[0, -1, :]
                capped_logits = self.softcap_logits(raw_logits)
                next_token = int(np.argmax(capped_logits))
                pos_offset += 1
                
            print("\n")

if __name__ == "__main__":
    # Start with E2B to save RAM as requested
    chat = GemmaChat(model_type="e2b")
    chat.chat_loop()
