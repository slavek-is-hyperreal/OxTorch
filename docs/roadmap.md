# 🗺️ VNN Development Roadmap & Handover

This document serves as the primary technical guide for subsequent AI agents and developers working on the **VNN Legacy Edition**. It outlines the current state, technical debt, and prioritized phases for future development.

## 📍 Current State
- **Phase 1-3**: Completed. Core tensor engine, ARAS (SSD) support, and PyTorch API parity are stable.
- **Phase 4 (In Progress)**: Autograd matured and verified for 1GB SSD tensors. Reductions (`sum`/`mean`) are SSD-native.
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

## 📉 Phase 6: Precision & Quantization
To fit even larger models on the same SSD space.

- [ ] **INT8/FP8 SSD Storage**: Allow tensors to be stored in 8-bit formats on disk and de-quantized during the ARAS prefetch phase.
- [ ] **Native FP16 Support**: Optimize Taichi kernels for half-precision to double the compute speed on supported Vulkan hardware.

## 🦾 Phase 7: "Monster" Model Verification
- [ ] **Llama-3 70B Execution**: Successfully load and run inference for a 70B model on a system with 8GB RAM using `from_binary`.
- [ ] **Checkpointing**: Implement SSD-native checkpointing that writes only the "dirty" pages (modified weights) to disk.

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
