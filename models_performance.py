# save_chestxray_backbones.py
import json, csv
from pathlib import Path

root = Path("saved_models")
rows = []
for d in sorted(root.iterdir()):
    if d.is_dir() and d.name.startswith("chestxray"):
        f = d / "args.txt"
        if f.exists():
            j = json.loads(f.read_text())
            rows.append((d.name, j.get("backbone",""), j.get("clip_name","")))

out = root / "chestxray_backbones.csv"
with out.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["model_dir","backbone","clip_name"])
    writer.writerows(rows)

print("Wrote", out)