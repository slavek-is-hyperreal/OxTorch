import os

files = [
    ".env",
    ".gitignore",
    "dump_weights.py",
    "examples/nanogpt_oxtorch_demo.py",
    "inspect_safetensors.py",
    "scripts/convert_gemma_to_f32.py",
    "scripts/vnn_adapter.py",
    "side_tools/gemini_to_markdown/__init__.py",
    "side_tools/gemini_to_markdown/gemini_to_md.py",
    "test_f16_linear_small.py",
    "test_f16_matmul_small.py",
    "test_f16_relu.py",
    "verify_msts.py"
]

with open("/tmp/chunk1.txt", "w", encoding="utf-8") as out:
    for f_path in files:
        out.write(f"\n====================\nFILE: {f_path}\n====================\n")
        try:
            with open(f_path, "r", encoding="utf-8") as inf:
                out.write(inf.read()[:5000]) # Protect against gigantic files just in case
        except Exception as e:
            out.write(f"Error reading: {e}\n")
