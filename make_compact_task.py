import os
with open(".agent/source_files.txt", "r") as f:
    files = [line.strip() for line in f if line.strip()]
files = [f for f in files if f not in [".agent/source_files.txt", ".agent/source_files_updated.txt"]]

with open(".agent/compact_task.md", "w") as f_out:
    f_out.write("# Plan przeglądu i czyszczenia kodu\n\n")
    f_out.write("Dla KAŻDEGO poniższego pliku zadaniem jest: (1) zapoznać się z całością, (2) opisać w `files_desc.md`, (3) ocenić redundancję i czy usunąć.\n\n")
    for f in files:
        f_out.write(f"- [ ] `{f}`\n")
