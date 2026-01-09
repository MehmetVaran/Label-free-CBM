#!/usr/bin/env python3
"""Create per-disease .txt concept files from base chest x-ray JSON concept sets.

This script loads all JSON files from an input directory (default:
`data/concept_sets/base_chestxray`), merges concept lists for each disease
(across the JSON files), deduplicates while preserving order, and writes a
single .txt file per disease with one concept per line.

Usage examples:

python3 setup_base_concepts.py
python3 setup_base_concepts.py --input-dir data/concept_sets/base_chestxray --output-dir data/umls_concepts/generated_base --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def _clean_concept(s: str) -> str:
    s = s.strip()
    # Remove leading dash/bullet markers
    if s.startswith("- "):
        s = s[2:]
    elif s.startswith("-"):
        s = s[1:]
    return s.strip()


def merge_json_concept_sets(input_dir: Path) -> Dict[str, List[str]]:
    files = sorted(input_dir.glob("*.json"))
    disease_map: Dict[str, List[str]] = {}

    for jf in files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        for disease, concepts in data.items():
            if not isinstance(concepts, list):
                continue
            if disease not in disease_map:
                disease_map[disease] = []
            for c in concepts:
                if not isinstance(c, str):
                    continue
                cleaned = _clean_concept(c)
                if not cleaned:
                    continue
                disease_map[disease].append(cleaned)

    # Deduplicate while preserving order
    for disease, lst in disease_map.items():
        seen = set()
        deduped: List[str] = []
        for item in lst:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        disease_map[disease] = deduped

    return disease_map


def _safe_filename(name: str) -> str:
    # Replace problematic characters with underscore
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name.strip("_") or "disease"


def write_txt_files(disease_map: Dict[str, List[str]], out_dir: Path, overwrite: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for disease, concepts in disease_map.items():
        fname = _safe_filename(disease) + ".txt"
        out_path = out_dir / fname
        if out_path.exists() and not overwrite:
            continue
        out_path.write_text("\n".join(concepts), encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create per-disease .txt files from base JSON concept sets.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/concept_sets/base_chestxray",
        help="Directory containing JSON concept files (default: data/concept_sets/base_chestxray)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/umls_concepts/generated_base",
        help="Directory to write per-disease .txt files (default: data/umls_concepts/generated_base)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files in the output directory.",
    )

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    out_dir = Path(args.output_dir)

    disease_map = merge_json_concept_sets(input_dir)
    write_txt_files(disease_map, out_dir, overwrite=args.overwrite)

    print(f"Wrote {len(disease_map)} disease files to {out_dir}")


if __name__ == "__main__":
    main()
