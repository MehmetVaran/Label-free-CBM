#!/usr/bin/env python3
"""Create tables from saved_models/chestxray_model_metrics.csv
Writes:
- saved_models/chestxray_model_metrics_no_modeldir.csv (same rows without model_dir)
- saved_models/chestxray_model_metrics_summary.csv (grouped by backbone, clip_name: mean/std for metrics)
Also prints a markdown table of the grouped summary.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "saved_models" / "chestxray_model_metrics.csv"
OUT1 = ROOT / "saved_models" / "chestxray_model_metrics_no_modeldir.csv"
OUT2 = ROOT / "saved_models" / "chestxray_model_metrics_summary.csv"

if not IN.exists():
    raise SystemExit(f"Input file not found: {IN}")

df = pd.read_csv(IN)
if 'model_dir' in df.columns:
    df_no_model = df.drop(columns=['model_dir'])
else:
    df_no_model = df.copy()

# Save per-run table without model_dir
OUT1.parent.mkdir(parents=True, exist_ok=True)
df_no_model.to_csv(OUT1, index=False)

# Numeric columns to aggregate
num_cols = [c for c in df_no_model.columns if c not in ['backbone','clip_name']]

# group and compute mean and std
grp = df_no_model.groupby(['backbone','clip_name'])[num_cols]
mean = grp.mean().add_suffix('_mean')
std = grp.std(ddof=0).add_suffix('_std')  # population std (ddof=0) for consistency
summary = pd.concat([mean, std], axis=1).reset_index()

# reorder columns: backbone, clip_name, then pairs
cols = ['backbone','clip_name'] + [col for pair in zip(mean.columns, std.columns) for col in pair]
summary = summary[cols]
summary.to_csv(OUT2, index=False)

# Print markdown table: show mean ± std for each metric in columns
def fmt_pair(mean_v, std_v):
    return f"{mean_v:.4f} ± {std_v:.4f}"

metric_names = [c.replace('_mean','') for c in mean.columns]
md_lines = []
header = ['backbone','clip_name'] + metric_names
md_lines.append('| ' + ' | '.join(header) + ' |')
md_lines.append('| ' + ' | '.join(['---']*len(header)) + ' |')
for _, row in summary.iterrows():
    cells = [row['backbone'], row['clip_name']]
    for m in metric_names:
        mean_v = row[f'{m}_mean']
        std_v = row[f'{m}_std']
        cells.append(fmt_pair(mean_v, std_v))
    md_lines.append('| ' + ' | '.join(map(str, cells)) + ' |')

print('\n'.join(md_lines))
print('\nWrote:')
print(' -', OUT1)
print(' -', OUT2)
