import os

with open(".agent/source_files.txt", "r") as f:
    files = [line.strip() for line in f if line.strip()]

filtered_files = []
cpu_ops_count = 0

for file in files:
    # Wyklucz pliki z samego skryptu .agent/
    if ".agent/" in file:
        continue
    # Wyklucz dema i stare legacy, żeby było spójne
    if "Python_Legacy" in file:
        continue
    # Wyklucz md
    if file.endswith(".md"):
        continue
    # Wyklucz benchmarki
    if "benchmark" in file.lower() or "perf" in file.lower():
        continue
    # Pogrupuj operacje CPU
    if "/cpu/ops/" in file:
        cpu_ops_count += 1
        continue
    
    filtered_files.append(file)

print("Kompaktowa lista:")
for f in filtered_files:
    print(f"- [ ] `{f}`")

print(f"- [ ] Grupa: Operacje CPU (`vulkannn_rusted/src/cpu/ops/*`) - {cpu_ops_count} plików")
