import time
import numpy as np
import torch
from vulkannn_rusted import Tensor

def run_bench():
    print("=== GEMMA 2B LAYER PERFORMANCE BENCHMARK (VNN vs PyTorch) ===")
    
    # Gemma 2B v1 Params
    h = 2048
    seq = 512 # Processing a block of 512 tokens
    inter = 16384
    
    # Weights (Random for bench)
    w_q_np = np.random.randn(h, h).astype(np.float32)
    w_k_np = np.random.randn(h, 256).astype(np.float32)
    w_v_np = np.random.randn(h, 256).astype(np.float32)
    w_o_np = np.random.randn(h, h).astype(np.float32)
    
    w_up_np = np.random.randn(h, inter).astype(np.float32)
    w_gate_np = np.random.randn(h, inter).astype(np.float32)
    w_down_np = np.random.randn(inter, h).astype(np.float32)
    
    x_np = np.random.randn(seq, h).astype(np.float32)
    
    # --- PYTORCH ---
    device = "cpu"
    x_pt = torch.from_numpy(x_np)
    wq_pt = torch.from_numpy(w_q_np)
    wk_pt = torch.from_numpy(w_k_np)
    wv_pt = torch.from_numpy(w_v_np)
    wo_pt = torch.from_numpy(w_o_np)
    wup_pt = torch.from_numpy(w_up_np)
    wgate_pt = torch.from_numpy(w_gate_np)
    wdown_pt = torch.from_numpy(w_down_np)
    
    print(f"\n[PyTorch CPU] Running Layer Forward (seq={seq})...")
    t0 = time.time()
    # Attention
    q = x_pt @ wq_pt
    k = x_pt @ wk_pt
    v = x_pt @ wv_pt
    # (Simplified attention logic: just linear projections + O)
    attn_out = q @ q.T # Dummy attention matmul just for latency
    attn_proj = x_pt @ wo_pt
    
    # MLP
    up = x_pt @ wup_pt
    gate = x_pt @ wgate_pt
    # GeLU/ReLU proxy
    act = torch.relu(up) * gate
    down = act @ wdown_pt
    
    t_pt = time.time() - t0
    print(f"  PyTorch Execution: {t_pt:.4f}s")
    
    # --- VNN RUSTED ---
    x_vnn = Tensor(x_np, device="cpu")
    wq_vnn = Tensor(w_q_np, device="cpu")
    wk_vnn = Tensor(w_k_np, device="cpu")
    wv_vnn = Tensor(w_v_np, device="cpu")
    wo_vnn = Tensor(w_o_np, device="cpu")
    wup_vnn = Tensor(w_up_np, device="cpu")
    wgate_vnn = Tensor(w_gate_np, device="cpu")
    wdown_vnn = Tensor(w_down_np, device="inter") # dummy placeholder shape
    wdown_vnn = Tensor(w_down_np, device="cpu")
    
    print(f"\n[VNN Rusted CPU] Running Layer Forward (seq={seq})...")
    t0 = time.time()
    # Attention Projections
    q_v = x_vnn @ wq_vnn
    k_v = x_vnn @ wk_vnn
    v_v = x_vnn @ wv_vnn
    # Attention Logic proxy
    # attn_out_v = q_v @ q_v.transpose() # We don't have transpose yet, skip or use fixed
    attn_proj_v = q_v @ wo_vnn # Q projection as proxy
    
    # MLP
    up_v = x_vnn @ wup_vnn
    gate_v = x_vnn @ wgate_vnn
    # Activation
    up_v = up_v.relu()
    # gate is multiplication, we don't have element-wise mul yet? 
    # Current VNN only has Add/ReLU. I'll use Add as a proxy for multiplication latency.
    # Use a new tensor for result to avoid borrow issues
    mlp_res = Tensor(shape=up_v.shape, device="cpu")
    up_v.add_into(gate_v, mlp_res) 
    down_v = mlp_res @ wdown_vnn
    
    t_vnn = time.time() - t0
    print(f"  VNN Execution: {t_vnn:.4f}s")
    
    ratio = t_vnn / t_pt
    print(f"\n--- FINAL COMPARISON (Layer Forward) ---")
    print(f"  PyTorch: {t_pt:.4f}s")
    print(f"  VNN Rusted: {t_vnn:.4f}s")
    print(f"  Ratio: {ratio:.2f}x")
    
    # Estimation for full model (18 layers)
    total_vnn = t_vnn * 18
    print(f"\n[Projected Execution for Gemma 2B (18 layers)]")
    print(f"  Full Block Forward: {total_vnn:.2f}s")
    print(f"  Approx ms/token: {(total_vnn/seq)*1000:.2f}ms")

if __name__ == "__main__":
    run_bench()
