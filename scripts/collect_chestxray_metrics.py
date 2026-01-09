#!/usr/bin/env python3
"""Collect backbone and selected metrics from chestxray saved_models.

Writes CSV to `saved_models/chestxray_model_metrics.csv` with columns:
model_dir,backbone,clip_name,loss_tr,acc_tr,loss_val,acc_val
"""
from pathlib import Path
import json
import csv
import argparse


def collect(root: Path):
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("chestxray"):
            continue
        backbone = ""
        clip_name = ""
        loss_tr = acc_tr = loss_val = acc_val = ""

        argsf = d / "args.txt"
        if argsf.exists():
            try:
                j = json.loads(argsf.read_text())
                backbone = j.get("backbone", "")
                clip_name = j.get("clip_name", "")
            except Exception:
                pass

        metricsf = d / "metrics.txt"
        if metricsf.exists():
            try:
                m = json.loads(metricsf.read_text())
                mm = m.get("metrics", m)
                loss_tr = mm.get("loss_tr", "")
                acc_tr = mm.get("acc_tr", "")
                loss_val = mm.get("loss_val", "")
                acc_val = mm.get("acc_val", "")
            except Exception:
                pass

        rows.append((d.name, backbone, clip_name, loss_tr, acc_tr, loss_val, acc_val))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="saved_models", help="Saved models root")
    p.add_argument("--out", default="saved_models/chestxray_model_metrics.csv", help="Output CSV path")
    args = p.parse_args()

    root = Path(args.root)
    rows = collect(root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline='', encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model_dir", "backbone", "clip_name", "loss_tr", "acc_tr", "loss_val", "acc_val"])
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
