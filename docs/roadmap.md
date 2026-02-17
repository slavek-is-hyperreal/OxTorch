# 🗺️ VNN Development Roadmap & Handover

This document serves as the primary technical guide for subsequent AI agents and developers working on the **VNN Legacy Edition**. It outlines the current state, technical debt, and prioritized phases for future development.

## 📍 Current State
- **Phase 5**: Completed. Hybrid PyTorch backend and operator fusion verified.
- **Phase 6**: Completed. **DRAS v4 (Adaptive Restart)** implemented. Cross-mode verification matrix confirms <1.5x slowdown vs Torch for RAM-resident ops and 140MB/s stable SSD streaming for 34GB tensors.
- **Phase 7 (NEXT STEP)**: Comprehensive Layer & Optimizer maturity (Linear, Conv2d, Adam) for 100GB+ models.
- **Critical Engine State**: Tensors are hashable by ID. `_acc_grad` is the universal entry point for gradients.

---

## 🚀 Phase 4: Optimizer Expansion (NEXT STEP)
The core Autograd is ready, but optimizers still rely partially on RAM/NumPy.

- [x] **SSD-Native Adam (`AutoAdam`)**: 
    - Implement `m` and `v` buffer management on SSD via ARAS.
    - Ensure update rule `w -= lr * m / (sqrt(v) + eps)` is performed in tiles without loading all 3 buffers (w, m, v) into RAM simultaneously.
- [x] **SSD-Native SGD**: Simplest first step for large-scale training.
- [x] **Verification**: Train a dummy model with 20GB+ parameters on a machine with 4GB RAM.

## ⚡ Phase 5: Operator Fusion & I/O Reduction
SSD bandwidth is our primary bottleneck.

- [x] **Kernel Fusion**: Fuse element-wise ops with reductions (e.g., `(w * g).sum()`) to save SSD write cycles.
- [x] **Prefetcher v2**: Use look-ahead logic to overlap SSD reads with CPU computations.
- [ ] **Prefetcher v2**: Implement a look-ahead prefetcher in `streaming_ops.py` that starts loading the next tile's data into RAM while the current tile is being processed by the CPU/GPU.

## 🚀 Phase 7: "Monster" Model Integration & Training
- [ ] **Llama-3 70B Benchmark**: successfully run inference for a 70B model using `from_binary` on 8GB-16GB RAM systems.
- [ ] **SSD-Native Training**: Optimize `Linear` and `Adam` for full SSD-to-SSD weight updates without loading full layers into RAM.
- [ ] **Adaptive Calibration**: Auto-tune `MAX_THREADS` and `tile_len` based on disk I/O latency (ZFS vs NVMe).

---

## 🛠️ Technical Pointers for the Next Agent

### 1. The ARAS Engine (`streaming_ops.py`)
This is where all "Monster Scale" logic lives. 
- **Rule**: Never call `to_numpy()` on a tensor inside a loop if the tensor can be on SSD.
- **Pattern**: Use `process_tile` with `ThreadPoolExecutor`.

### 2. Autograd Graph (`tensor.py`)
- The `_prev` set and `_backward_fn` are initialized in `__init__`.
- Gradients are accumulated via `_acc_grad`. If you add a new operation, ensure it correctly calls `_acc_grad` in its `_backward` closure.

### 3. Taichi vs NumPy
- **Vulkan** is for speed on small/medium buffers.
- **NumPy** is for numerical logic on RAM/SSD tiles.
- **ARAS** is the orchestrator for large buffers.

---

## 🆘 Troubleshooting Common Issues
- **Math Mismatch**: Usually caused by Taichi-on-CPU not matching NumPy floating-point precision or missing zero-init.
- **OOM**: Check if `tile_len` in `streaming_ops.py` is being calculated correctly based on `MemAvailable`.
- **AttributeError**: Ensure Autograd metadata (`_prev`, `_backward_fn`) is initialized for ALL tensors, especially those created via factory functions in `torch_shim.py`.
