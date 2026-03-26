import os

with open(".agent/source_files.txt", "r") as f:
    files = [line.strip() for line in f if line.strip()]

filtered_files = []
cpu_ops_dirs = set()

for file in files:
    # Wyklucz pliki z samego skryptu .agent/
    if ".agent/" in file:
        continue
    # Wyklucz dema i stare legacy
    if "Python_Legacy" in file:
        continue
    # Wyklucz md
    if file.endswith(".md"):
        continue
    # Wyklucz benchmarki
    if "benchmark" in file.lower() or "perf" in file.lower():
        continue
        
    # Złap operacje CPU
    if "/cpu/ops/" in file:
        # np. ./vulkannn_rusted/src/cpu/ops/binary/add/add_bf16.rs
        # grupujemy wszystko do rodzica
        dir_name = os.path.dirname(file)
        # jeśli to sam plik w ops/ lub głębiej, bierzemy katalog
        cpu_ops_dirs.add(dir_name)
        continue
    
    filtered_files.append(file)

print("# Plan przeglądu i czyszczenia kodu")
print("Dla KAŻDEGO poniższego punktu zadaniem jest: (1) zapoznać się z kodem, (2) opisać w `files_desc.md`, (3) ocenić redundancję i czy usunąć.\n")

for f in sorted(filtered_files):
    print(f"- [ ] `{f}`")

print("\n## Operacje CPU (pogrupowane katalogami)")
for d in sorted(cpu_ops_dirs):
    print(f"- [ ] `{d}/` (wszystkie warianty precyzji wewnątrz katalogu)")
