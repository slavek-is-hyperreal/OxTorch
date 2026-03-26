import json
import os

with open(".agent/source_files.txt", "r") as f:
    files = [line.strip() for line in f if line.strip()]

# filter out empty or hidden files manually if needed
files = [f for f in files if f not in [".agent/source_files.txt", ".agent/source_files_updated.txt"]]

with open("files_desc.md", "w") as f_desc:
    f_desc.write("# Lista plików w projekcie\n\n")
    for file in files:
        f_desc.write(f"## `{file}`\n\n")
        f_desc.write("**Opis:**\n\n")
        f_desc.write("**Rola w projekcie:**\n\n")
        f_desc.write("**Przestarzały/redundantny kod:**\n\n")
        f_desc.write("**Kandydat do usunięcia:**\n\n")
        f_desc.write("---\n\n")

# Provide an artifact task.md. In the system, task artifacts should go to `.agent/task.md` or similar, 
# wait, actually the brain folder. Wait, I should use `write_to_file` to create the artifact, because
# I don't know the conversation ID path perfectly. I CAN know it: /home/slavekm/.gemini/antigravity/brain/993001e0-630e-4411-bb85-77c95671546e/task.md

task_file = "/home/slavekm/.gemini/antigravity/brain/993001e0-630e-4411-bb85-77c95671546e/task.md"
os.makedirs(os.path.dirname(task_file), exist_ok=True)

with open(task_file, "w") as t_desc:
    t_desc.write("# Plan przeglądu i czyszczenia kodu\n\n")
    t_desc.write("## 1. Wstępne zapoznanie z rdzeniem dokumentacji\n")
    t_desc.write("- [x] Przeczytanie WSZYSTKICH LINII `README.md`\n")
    t_desc.write("- [x] Przeczytanie WSZYSTKICH LINII `docs/doc_index.md`\n\n")
    
    t_desc.write("## 2. Przegląd `docs/`\n")
    doc_files = [f for f in files if f.startswith("./docs/")]
    for df in doc_files:
        t_desc.write(f"- [ ] Przeczytanie WSZYSTKICH LINII `{df}`\n")
        
    t_desc.write("\n## 3. Przegląd i ewaluacja wszystkich plików (`files_desc.md`)\n")
    for file in files:
        t_desc.write(f"- [ ] `{file}`\n")
        t_desc.write(f"  - [ ] zapoznaj się z CAŁOŚCIĄ danego pliku\n")
        t_desc.write(f"  - [ ] dodaj jego opis, rolę w projekcie, informację czy zawiera przestarzały i/lub redundantny kod, czy jest kandydatem do usunięcia (w `files_desc.md`)\n")

print(f"Generated files_desc.md and task.md with {len(files)} files.")
