import os

TEMPLATE = """from ..base import BenchmarkBase

if __name__ == "__main__":
    bench = BenchmarkBase(
        name="{name}",
        op="{op}",
        shape={shape},
        mode="{mode}",
        dtype="{dtype}",
        inplace={inplace},
        is_ssd={is_ssd}
    )
    bench.run()
"""

# Config from unified_benchmark.py
dtypes = ["f32", "f16", "bf16", "int8"]
modes = ["cpu", "vulkan", "hybrid"]
ops = [
    ("MatMul", (2048, 2048)),
    ("Sum", (2048, 2048)),
    ("Softmax", (2048, 2048)),
    ("Mul", (2048, 2048)),
    ("Sub", (2048, 2048)),
    ("ScalarAdd", (2048, 2048)),
    ("ScalarMul", (2048, 2048)),
    ("GELU", (2048, 2048)),
    ("ReLU", (1000000,)),
    ("ReLU_15M", (15000000,))
]

base_dir = "/my_data/gaussian_room/tests/benchmarks"

for dtype in dtypes:
    for mode in modes:
        for op_name, shape in ops:
            # Handle ReLU_15M as op ReLU with larger shape
            actual_op = "ReLU" if "ReLU" in op_name else op_name
            name = f"{op_name}_{dtype}_{mode}"
            file_name = f"{op_name.lower()}_{mode}.py"
            file_path = os.path.join(base_dir, dtype, file_name)
            
            content = TEMPLATE.format(
                name=name,
                op=actual_op,
                shape=shape,
                mode=mode,
                dtype=dtype,
                inplace="False",
                is_ssd="False"
            )
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(content)

# Special case: ReLU Inplace (CPU only)
for dtype in dtypes:
    file_path = os.path.join(base_dir, dtype, "relu_inplace_cpu.py")
    content = TEMPLATE.format(
        name=f"ReLU_{dtype}_cpu_inplace",
        op="ReLU",
        shape=(1000000,),
        mode="cpu",
        dtype=dtype,
        inplace="True",
        is_ssd="False"
    )
    with open(file_path, 'w') as f:
        f.write(content)

# Special case: Monster ReLU SSD
file_path = os.path.join(base_dir, "monster", "relu_ssd_f32.py")
content = TEMPLATE.format(
    name="Monster_ReLU_F32_SSD",
    op="ReLU",
    shape=(4000000000,),
    mode="cpu",
    dtype="f32",
    inplace="False",
    is_ssd="True"
)
with open(file_path, 'w') as f:
    f.write(content)

print(f"Generated atomized benchmarks in {base_dir}")
