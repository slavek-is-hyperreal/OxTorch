# Notes / "ADHD Parking Lot"

This file serves as a scratchpad and repository for architectural ideas that come to mind during sprints. We will periodically revisit them and decide whether to implement them into the main `roadmap.md` or discard them.

---

## 1. MSTS: Giant RAM-based FIFO "Capacitor" for SSDs
**Project Layer:** Sprint 4 / 5 (Offloading large LLM model inference)
**Added:** Late Phase 1 (Indexing)

### Problem:
The current MSTS ring buffer offloads weights pulled from the SSD directly into small areas optimized for the processor's cache (L2/L3). If the disk experiences an "I/O spike" (sudden drop in speed / paging latency), the computation thread becomes "starved" and waits for data. Simultaneously, in computers with several GB of free physical RAM, this memory is not used in any way during offloading.

### Solution:
- Build an asynchronous, massive FIFO queue (Look-ahead Prefetching Buffer) allocated in free physical RAM (e.g., occupying 4GB of 5.6GB free).
- The queue acts like a **capacitor** smoothing out irregularities in SSD transfer.
- A dedicated worker thread (e.g., using `io_uring`) reads data mechanically and consumes weights from the disk with maximum throughput, always trying to keep the RAM capacitor filled (e.g., with LLaMA weights for subsequent network calls).
- The processor's computation threads / CPU workers (and their cache) pull data without any waiting (zero-latency offloading) directly from the RAM, rather than directly from the memory-mapped disk.

### Benefits:
Guaranteed smoothing of SSD/NVMe performance drops. This allows for a constant number of *tokens per second* generated without sudden "hitches" in LLM inference resulting from operating system filesystem latencies.

---

## 2. Vulkan Descriptor Pool Memory Expansion
**Added:** Late Phase 1 (Indexing)

### Note:
Expanding the backend with more pipelines like `index_select` quickly drains `vk::DescriptorPoolCreateInfo`. We allocate pools in advance (`create_pool()`). Remember to continuously monitor `max_sets` (currently increased to 1024) and `descriptor_count` (increased to 4096 for STORAGE); otherwise, newly added operations will return `ERROR_OUT_OF_POOL_MEMORY` in production.
