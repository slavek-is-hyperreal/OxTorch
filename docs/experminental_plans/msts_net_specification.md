# MSTS Net: Horizontal Scalability for the Rest of Us

## Vision
To enable the execution of Frontier-class LLMs (e.g., Llama-3 70B, BitNet 2B/7B) on ubiquitous, older hardware by distributing computations across a local network of "legacy" nodes (i5-3450, AMD Bonaire, laptops with 8GB RAM).

## 1. Core Architecture: Tile-Based Dataflow
Inspired by the **MERA-400** and its asymmetric data handling, the system treats the entire cluster as a single, distributed "Mega-Processor".

### The Tile Unit
- **Size**: 1.0 MB (fixed) to balance PCIe/Network latency with compute throughput.
- **Format**: `vNN_BitNet_2bit` (weights) and `BF16` (activations).
- **Header (The Tag)**: Every tile carries a "Tagged-Token" containing:
  - `Layer_ID`, `Tile_Index`, `Sequence_ID`.
  - **P-Flag (Presence)**: Analogous to MERA's *STROB 1*.
  - **Q-Flag (Queued)**: Analogous to MERA's *ZG (Zgłoszenie)*.

## 2. Distributed Protocol (MERA-style Handshake)
MSTS Net implements a **Virtual Backplane** protocol:
1. **The Request (SZUKAJ)**: A node broadcasts a request for a specific tile tag.
2. **The Handshake (OK / EN)**: 
   - If a node has the tile and is free, it replies **OK** and "locks" the tile (Q=1).
   - If a node is busy, it replies **EN (Zajętość)**, and the requester tries the next peer.
3. **Asynchronous Completion**: No global clock. Tiles return whenever they are done, allowing a 10-year-old laptop to participate without slowing down a faster node.

## 3. Resilience: The "ZF/OFF" Logic
Inspired by MERA's power-fail recovery:
- **Heartbeat Timeout**: If a node claiming a tile (Q=1) stops sending heartbeats, the tile is "released" back to the network (Q=0, P=1).
- **Graceful Exit**: If a laptop lid is closed, the node sends a "ZF (Zezwolenie na odłączenie)" packet, releasing all held tiles before disconnecting.

## 3. Node Specialization
The "OxTorch Nodes" are heterogeneous:
- **GPU Nodes (Bonaire/GCN)**: Specialize in high-throughput `BitLinear` tiles.
- **CPU Nodes (i5/Ivy Bridge)**: Specialize in `Softmax`, `RMSNorm`, and `ReLU^2` (Fused kernels).
- **Memory Nodes**: Use `io_uring` to stream weights from SSD to the network "bus".

## 4. Why BitNet is the Key?
- **Standard FP16**: 14GB for 7B parameters. Too heavy for 1Gbps Ethernet.
- **BitNet 1.58b**: ~1.4GB for 7B parameters. 
- **Efficiency**: A 1Gbps network can transfer ~125MB/s. A BitNet model can be streamed at nearly full speed, allowing 500 laptops to maintain a collective "High-Speed Memory Bus" that exceeds any single modern GPU.

## 5. Deployment Scenario
"The Corporate Basement"
- 500 Laptops ($ \approx 2000 $ cores, 4TB RAM total).
- No RTX cards.
- **Output**: Real-time inference of a 70B model with <50ms latency per token.

## 6. Networking: Use of Custom Stack
To achieve "MERA-level" low latency, we **cannot** use standard TCP/IP streams.
- **Protocol**: **Lite-VNN (UDP-based)**. We bypass TCP's 3-way handshake and retransmission logic.
- **Kernel-Bypass**: Using `io_uring` (Linux) for zero-copy packet submission to the NIC.
- **Reliability**: Implemented at the **Tile-level**, not the Packet-level. If a tile is incomplete (fragment lost), the node simply returns "EN (Busy/Error)" to the Master, and the Master re-broadcasts.

## 7. Testing Strategy: "The 500-Node Loopback"
How to test 500 nodes without 500 laptops:
1. **Virtual Mesh**: Launch 50-100 `oxtorch-node` processes on a single i5-3450, communicating via `127.0.0.1`.
2. **Traffic Control (`tc/netem`)**:
   - `tc qdisc add dev lo root netem delay 10ms 2ms loss 1%`
   - This simulates real-world Ethernet jitter and packet loss locally.
3. **P/Q Stress Testing**: Randomly "killing" worker processes during a 70B MatMul to verify that the MERA-style tags correctly redistribute work to surviving nodes.
