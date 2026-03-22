# MSTS State Machine & Internal Visualization

The **Multi-Stage Tensor Streaming (MSTS)** system is the heart of OxTorch's performance on massive datasets. It coordinates asynchronous I/O with high-throughput compute.

## State Machine Visualization

Each **Tile** (typically 1MB) in the `CrookScheduler`'s ring buffer follows a strict lifecycle. This ensures that the Reader, Worker, and Writer never access the same memory concurrently.

```mermaid
stateDiagram-v2
    [*] --> EMPTY
    EMPTY --> READY_FOR_COMPUTE : Reader signs Tile
    READY_FOR_COMPUTE --> COMPUTING : Worker grabs Tile
    COMPUTING --> READY_FOR_WRITE : Worker signs Tile
    READY_FOR_WRITE --> EMPTY : Writer signs Tile
    EMPTY --> [*]
```

### State Definitions
| State | Responsibility | Description |
| :--- | :--- | :--- |
| `EMPTY` | **Reader** | The tile is a clean slate, waiting for data from the SSD. |
| `READY_FOR_COMPUTE` | **Worker** | The tile contains valid input data, waiting for the CPU/GPU kernel. |
| `COMPUTING` | **Worker** | The kernel is actively processing the tile's data. |
| `READY_FOR_WRITE` | **Writer** | The results are ready in the tile, waiting to be streamed back to disk. |

---

## The 3-Path Dispatch Strategy

To maximize performance across all tensor sizes, MSTS dynamically chooses the most efficient execution path at runtime.

```mermaid
graph TD
    Start[Execute Op] --> CheckSize{Tensor Size?}
    CheckSize -- "< DIRECT_MAX" --> PathA[Path A: Direct]
    CheckSize -- "< 32MB" --> PathB[Path B: Streaming Single]
    CheckSize -- "> 32MB" --> PathC[Path C: Streaming Parallel]

    PathA --> DirectIO[Direct I/O: Single Chunk]
    DirectIO --> SingleThread[Single-Threaded Kernel]
    SingleThread --> Done

    PathB --> StartSched[Start CrookScheduler]
    StartSched --> SchedLoop[1MB Tile Loop]
    SchedLoop --> LoopSingle[Single-Threaded Shared Worker]
    LoopSingle --> SchedLoop
    SchedLoop -- "Finished" --> Done

    PathC --> StartSchedParallel[Start CrookScheduler]
    StartSchedParallel --> SchedLoopParallel[1MB Tile Loop]
    SchedLoopParallel --> LoopParallel[Rayon Parallel Shared Worker]
    LoopParallel --> SchedLoopParallel
    SchedLoopParallel -- "Finished" --> Done
```

---

## Component Interaction Overview

The following diagram shows how the Host threads interact with the SSD using `io_uring` and the `CrookScheduler`.

```mermaid
sequenceDiagram
    participant SSD as NVMe SSD
    participant Reader as I/O Reader Thread
    participant Ring as CrookScheduler (Ring Buffer)
    participant Worker as Compute Workers (Host CPU)
    participant Writer as I/O Writer Thread

    Note over Ring: All tiles set to EMPTY
    Reader->>SSD: Submit io_uring Read Requests
    SSD-->>Reader: Data ready
    Reader->>Ring: Fill Tile 0, set READY_FOR_COMPUTE
    Worker->>Ring: Grab Tile 0, set COMPUTING
    Worker->>Worker: Run SIMD Kernels (Relu, Gelu, etc.)
    Worker->>Ring: Finished Tile 0, set READY_FOR_WRITE
    Writer->>Ring: Grab Tile 0
    Writer->>SSD: Submit io_uring Write Requests
    SSD-->>Writer: Write confirmed
    Writer->>Ring: Reset Tile 0 to EMPTY
```

## Related Documentation
- [MSTS Logic](msts_logic.md): Theoretical background.
- [SSD Format](ssd_format.md): Binary structure of the data.
- [CPU Backend](cpu_backend.md): Details on the SIMD kernels used by the Workers.
