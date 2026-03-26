import os

files = [
    "tests/__init__.py",
    "tests/analyze_history.py",
    "tests/archive/diagnose_vram.py",
    "tests/archive/test_gguf_api.py",
    "tests/archive/test_taichi_smem.py",
    "tests/archive/test_vnn_rusted.py",
    "tests/chat_gemma_3n.py",
    "tests/chat_gemma_oom_safe.py",
    "tests/conftest.py",
    "tests/generate_chart.py",
    "tests/overnight_bench.py",
    "tests/utils.py",
    "vulkannn_rusted/Cargo.toml",
    "vulkannn_rusted/build.rs"
]

with open("/tmp/chunk2.txt", "w", encoding="utf-8") as out:
    for f_path in files:
        out.write(f"\n====================\nFILE: {f_path}\n====================\n")
        try:
            with open(f_path, "r", encoding="utf-8") as inf:
                out.write(inf.read()[:5000])
        except Exception as e:
            out.write(f"Error reading: {e}\n")
