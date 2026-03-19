"""
OxTorch x nanoGPT — 1 import demo
Run: PYTHONPATH=/path/to/vulkannn_rusted python this_script.py

This is a self-contained benchmark comparing PyTorch vs OxTorch
on the core operations inside karpathy/nanoGPT.
No model weights needed — just random tensors at GPT-2 scale.
"""

import time
import numpy as np

# ---- THE ONLY CHANGE YOU NEED IN ANY PYTORCH PROJECT ----
import oxtorch as torch
# (comment above / uncomment below to compare)
# import torch
# ---------------------------------------------------------

B, T, C = 4, 1024, 768      # GPT-2 small: batch=4, seq=1024, embed=768
H, HEAD = 12, 64            # 12 attention heads, head_dim=64
FF = C * 4                  # feedforward hidden = 3072

dtype = torch.float16       # F16 — where OxTorch shines most

print(f"\n{'='*60}")
print(f"  OxTorch x nanoGPT — Core Op Benchmark")
print(f"  B={B}  T={T}  C={C}  dtype=F16")
print(f"{'='*60}\n")

def bench(name, fn, warmup=2, iters=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - t0) / iters
    print(f"  {name:<35} {elapsed*1000:8.2f} ms")
    return elapsed

# ----------- Projection MatMul (QKV projection, C -> 3C) -----------
x   = torch.randn(B * T, C).half()
W   = torch.randn(C, 3 * C).half()

print("[ Attention QKV Projection ]")
t_proj = bench("matmul (B*T x C) @ (C x 3C)",
               lambda: torch.matmul(x, W))

# ----------- Attention Scores (QK^T per head, T x T) ---------------
Q = torch.randn(B * H, T, HEAD).half()
K = torch.randn(B * H, HEAD, T).half()

print("\n[ Attention Scores QK^T ]")
t_attn = bench("matmul QK^T (BH, T, HEAD)@(BH, HEAD, T)",
               lambda: torch.matmul(Q, K))

# ----------- Softmax ---------------------------------------------------
logits = torch.randn(B * H, T, T).half()

print("\n[ Softmax over attention logits ]")
t_softmax = bench("softmax dim=-1 (BH, T, T)",
                  lambda: torch.nn.functional.softmax(logits, dim=-1))

# ----------- FeedForward (GELU MLP) ------------------------------------
xf  = torch.randn(B * T, C).half()
W1  = torch.randn(C, FF).half()
W2  = torch.randn(FF, C).half()

print("\n[ FeedForward Block ]")
t_ff1 = bench("matmul FC1  (B*T x C) @ (C x FF)",
              lambda: torch.matmul(xf, W1))
t_gelu = bench("gelu (B*T x FF)",
               lambda: torch.nn.functional.gelu(torch.matmul(xf, W1)))
t_ff2 = bench("matmul FC2  (B*T x FF) @ (FF x C)",
              lambda: torch.matmul(torch.nn.functional.gelu(torch.matmul(xf, W1)), W2))

print(f"\n{'='*60}")
print(f"  Done. Swap 'import oxtorch as torch' ↔ 'import torch'")
print(f"  and compare the numbers yourself.")
print(f"{'='*60}\n")
