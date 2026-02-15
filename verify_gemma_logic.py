import numpy as np
import taichi as ti
import vulkan_nn_lib as vnn
import vulkan_nn_lib.torch_shim as torch

def verify_gemma_logic():
    print("--- Verifying Gemma 3n Architecture on Vulkan ---")
    
    # Tiny Params for quick verification
    hidden_size = 512
    num_heads = 8
    num_layers = 2
    intermediate_size = 2048
    vocab_size = 1000
    sub_model_size = 256
    seq_len = 16
    max_len = 64
    
    # 1. Initialize Model
    model = vnn.Gemma3Model(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        vocab_size=vocab_size
    )
    
    # 2. Populate Synthetic Weights
    # (Simplified: we'll just let them be zero/random in physical VRAM)
    # TiledLinear layers need weight_ram populated
    for layer in model.layers:
        layer.q_proj.weight_ram = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        layer.k_proj.weight_ram = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        layer.v_proj.weight_ram = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        layer.o_proj.weight_ram = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        layer.gate_proj.weight_ram = np.random.randn(hidden_size, intermediate_size).astype(np.float32)
        layer.up_proj.weight_ram = np.random.randn(hidden_size, intermediate_size).astype(np.float32)
        layer.down_proj.weight_ram = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
        
    model.lm_head.weight_ram = np.random.randn(hidden_size, vocab_size).astype(np.float32)

    # 3. Prepare Inputs
    input_ids = vnn.Tensor(np.random.randint(0, vocab_size, (1, seq_len), dtype=np.int32))
    cos_np, sin_np = vnn.get_cos_sin(max_len, hidden_size // num_heads)
    cos = vnn.Tensor(cos_np)
    sin = vnn.Tensor(sin_np)
    
    # 4. KV-Cache
    caches = []
    for _ in range(num_layers):
        k_c = vnn.Tensor(None, shape=(1, max_len, num_heads, hidden_size // num_heads))
        v_c = vnn.Tensor(None, shape=(1, max_len, num_heads, hidden_size // num_heads))
        caches.append((k_c, v_c))
        
    # 5. Execute Forward Pass (Prefill)
    print(f"Running Prefill (Seq Len: {seq_len})...")
    logits = model(input_ids, cos, sin, caches=caches, pos_offset=0, sub_model_size=sub_model_size)
    print(f"Prefill Output Shape: {logits.shape}")
    
    # 6. Execute Forward Pass (Single Token)
    print("Running Single Token Generation step...")
    next_token_id = vnn.Tensor(np.array([[42]], dtype=np.int32))
    logits_step = model(next_token_id, cos, sin, caches=caches, pos_offset=seq_len, sub_model_size=sub_model_size)
    print(f"Token Step Output Shape: {logits_step.shape}")
    
    print("--- Logic Verification Successful! ---")

if __name__ == "__main__":
    verify_gemma_logic()
