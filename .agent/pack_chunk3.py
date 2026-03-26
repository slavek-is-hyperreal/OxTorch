import os

files = [
    "vulkannn_rusted/src/backend.rs",
    "vulkannn_rusted/src/buf_pool.rs",
    "vulkannn_rusted/src/cpu/conversions.rs",
    "vulkannn_rusted/src/cpu/mod.rs",
    "vulkannn_rusted/src/cpu/tiling_cpu.rs",
    "vulkannn_rusted/src/crook_scheduler.rs",
    "vulkannn_rusted/src/io_uring_engine.rs",
    "vulkannn_rusted/src/lib.rs",
    "vulkannn_rusted/src/models/bitnet.rs",
    "vulkannn_rusted/src/models/mod.rs",
    "vulkannn_rusted/src/prng.rs",
    "vulkannn_rusted/src/streaming.rs"
]

with open("/tmp/chunk3.txt", "w", encoding="utf-8") as out:
    for f_path in files:
        out.write(f"\n====================\nFILE: {f_path}\n====================\n")
        try:
            with open(f_path, "r", encoding="utf-8") as inf:
                out.write(inf.read()[:5000])
        except Exception as e:
            out.write(f"Error reading {f_path}: {e}\n")
