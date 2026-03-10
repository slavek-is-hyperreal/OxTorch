# Vulkan Optimization Research: Small Operations & Latency

## 1. Staging vs. Mappable VRAM
- **Staging Buffers**: Incur high overhead for small data (ReLU, Add) due to:
    - `vkCmdCopyBuffer` command recording.
    - PCIe transfer latency for small packets.
    - Synchronous memory barriers between Transfer and Compute stages.
- **Mappable VRAM (`DEVICE_LOCAL | HOST_VISIBLE`)**: 
    - **Integrated GPUs**: Best case; no physical copy needed.
    - **Discrete GPUs + Resizable BAR**: Allows CPU to write directly to VRAM.
    - **Optimization**: Use `MemoryLocation::CpuToGpu` for buffers < 1MB that are frequently updated.

## 2. Synchronization Overhead
- **`vkQueueWaitIdle`**: Fatal to performance. It flushes the entire GPU pipeline and stalls the CPU.
- **Alternative**: Use **Fences** for CPU-GPU sync.
- **True Fix**: Asynchronous submission via **MSTS**. The CPU should submit work and continue, only waiting when the data is actually needed by the user (e.g., `to_numpy()`).

## 3. Command Submission (Batching)
- **Problem**: Submitting one command buffer per `Tensor` operation (e.g., `a + b`) is expensive (approx. 10-50us overhead per `vkQueueSubmit`).
- **Optimization**: 
    - Batch multiple operations into a single command buffer if they are independent.
    - Use **Push Constants** for small metadata (shapes, strides) instead of Uniform Buffers where possible.

## 4. Hardware Specifics (RADV / Bonaire)
- The user's GPU (AMD Radeon R7 200 Series / Bonaire) supports **AMD Smart Access Memory** (Resizable BAR) in some configurations on Linux.
- If BAR is small, we must prioritize what goes into Mappable VRAM.

## 5. Proposed "Quick Wins"
1. **Direct Mapping**: Use `CpuToGpu` for `Add` and `ReLU` output buffers if size is small.
2. **Fence-based Wait**: Replace `queue_wait_idle` with a fence wait in `backend.rs`.
3. **Persistent Mapping**: Ensure buffers are not mapped/unmapped repeatedly (already handled by `gpu-allocator`).
