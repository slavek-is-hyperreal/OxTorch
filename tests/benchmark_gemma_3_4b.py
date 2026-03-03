import time
import numpy as np
import torch
from vulkannn_rusted import Tensor

def run_bench():
    print("=== GEMMA 3 4B LAYER PERFORMANCE BENCHMARK (VNN vs PyTorch) ===")
    
    # Gemma 3 4B Params
    h = 2560
    seq = 512 
    inter = 10240
    n_heads = 8
    n_kv_heads = 4
    head_dim = 256
    
    # Projections
    # qkv_proj: h -> (n_heads + 2*n_kv_heads) * head_dim
    # 2560 -> (8 + 2*4) * 256 = 16 * 256 = 4096
    qkv_out_dim = (n_heads + 2 * n_kv_heads) * head_dim
    
    # Weights
    w_qkv_np = np.random.randn(h, qkv_out_dim).astype(np.float32)
    w_o_np = np.random.randn(n_heads * head_dim, h).astype(np.float32)
    
    w_up_np = np.random.randn(h, inter).astype(np.float32)
    w_gate_np = np.random.randn(h, inter).astype(np.float32)
    w_down_np = np.random.randn(inter, h).astype(np.float32)
    
    x_np = np.random.randn(seq, h).astype(np.float32)
    
    # --- PYTORCH ---
    x_pt = torch.from_numpy(x_np)
    wqkv_pt = torch.from_numpy(w_qkv_np)
    wo_pt = torch.from_numpy(w_o_np)
    wup_pt = torch.from_numpy(w_up_np)
    wgate_pt = torch.from_numpy(w_gate_np)
    wdown_pt = torch.from_numpy(w_down_np)
    
    print(f"\n[PyTorch CPU] Running Gemma 3 Layer (seq={seq})...")
    t0 = time.time()
    # 1. Norm (Skip actual math, just linear/matmul are bottlenecks)
    # 2. QKV Projection
    qkv = x_pt @ wqkv_pt
    # 3. O Projection
    # (Simplified: input to o_proj is num_heads * head_dim)
    attn_in = torch.randn(seq, n_heads * head_dim)
    o = attn_in @ wo_pt
    # 4. MLP
    up = x_pt @ wup_pt
    gate = x_pt @ wgate_pt
    act = torch.relu(up) * gate
    down = act @ wdown_pt
    
    t_pt = time.time() - t0
    print(f"  PyTorch Execution: {t_pt:.4f}s")
    
    # --- VNN RUSTED ---
    x_vnn = Tensor(x_np, device="cpu")
    wqkv_vnn = Tensor(w_qkv_np, device="cpu")
    wo_vnn = Tensor(w_o_np, device="cpu")
    wup_vnn = Tensor(w_up_np, device="cpu")
    wgate_vnn = Tensor(w_gate_np, device="cpu")
    wdown_vnn = Tensor(w_down_np, device="cpu")
    
    # Pre-allocate staging tensors for ops
    qkv_res = Tensor(shape=(seq, qkv_out_dim), device="cpu")
    o_res = Tensor(shape=(seq, h), device="cpu")
    up_res = Tensor(shape=(seq, inter), device="cpu")
    gate_res = Tensor(shape=(seq, inter), device="cpu")
    mlp_res = Tensor(shape=(seq, inter), device="cpu")
    down_res = Tensor(shape=(seq, h), device="cpu")
    
    print(f"\n[VNN Rusted CPU] Running Gemma 3 Layer (seq={seq})...")
    t0 = time.time()
    # 1. QKV Projection
    # MatMul is built-in @ or .matmul()
    qkv_v = x_vnn @ wqkv_vnn
    
    # 2. O Projection
    # attn_in proxy
    attn_in_v = Tensor(np.random.randn(seq, n_heads * head_dim).astype(np.float32), device="cpu")
    o_v = attn_in_v @ wo_vnn
    
    # 3. MLP
    up_v = x_vnn @ wup_vnn
    gate_v = x_vnn @ wgate_vnn
    
    # Activation & Multiply proxy
    up_v.relu_into(up_res) # Using separate res to avoid borrow issues
    up_res.add_into(gate_v, mlp_res) # Add as proxy for element-mul latency
    
    # Final MLP Down
    down_v = mlp_res @ wdown_vnn
    
    t_vnn = time.time() - t0
    print(f"  VNN Execution: {t_vnn:.4f}s")
    
    ratio = t_vnn / t_pt
    print(f"\n--- FINAL COMPARISON (Gemma 3 4B Layer) ---")
    print(f"  PyTorch: {t_pt:.4f}s")
    print(f"  VNN Rusted: {t_vnn:.4f}s")
    print(f"  Ratio: {ratio:.2f}x")
    
    # Estimation for full model (34 layers)
    total_vnn = t_vnn * 34
    print(f"\n[Projected Execution for Gemma 3 4B (34 layers)]")
    print(f"  Full Block Forward: {total_vnn:.2f}s")
    print(f"  Approx ms/token: {(total_vnn/seq)*1000:.2f}ms")

if __name__ == "__main__":
    run_bench()
