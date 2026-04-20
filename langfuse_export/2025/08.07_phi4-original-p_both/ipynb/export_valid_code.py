# %% [markdown]
# # Export valid sketches from raw_*.json files in raw_export
# This notebook extracts code snippets stored in the `sketch` field where `status == "success"` and writes them as `.py` files under a new sibling folder `exported_valid_code`.
# 
# Outline:
# - Import dependencies
# - Resolve input/output paths
# - Enumerate raw JSON files
# - Parse and filter successful sketches
# - Write extracted code to files
# - Quick verification
# 

# %%
# Import dependencies
import os
import json
import csv
import glob
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

# %%
# Resolve input and output paths
# Derive base_dir relative to the notebook location to keep things portable.
# If run from the ipynb folder, base_dir is its parent; otherwise use CWD.
# cwd = Path.cwd().resolve()
# base_dir = cwd.parent if cwd.name == 'ipynb' else cwd
base_dir = Path("/home/han/Projects/benchmark-tinyml_llm-2026/langfuse_export/2025/07.30_abla-og-phi4")
raw_dir = base_dir / 'raw_export'
export_dir = base_dir / 'exported_valid_code'

# Derive a dataset suffix from the folder name (drop leading number prefix if present)
name_parts = base_dir.name.split('_', 1)
dataset_suffix = name_parts[1] if len(name_parts) > 1 else base_dir.name
# Sanitize suffix for filenames
_dataset_clean = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in dataset_suffix).strip('_')
dataset_suffix = _dataset_clean or 'dataset'

# Pre-create known prefix subfolders (sg, psg, tpusg); fallback created on demand
prefix_dirs = [export_dir / 'sg', export_dir / 'psg', export_dir / 'tpusg']
for d in prefix_dirs:
    d.mkdir(parents=True, exist_ok=True)

print(f'Base dir: {base_dir}')
print(f'Dataset suffix: {dataset_suffix}')
print(f'Raw dir: {raw_dir}')
print(f'Export root: {export_dir}')
print('Subfolders (precreated):')
for d in prefix_dirs:
    print(' -', d)

# %%
# Enumerate raw JSON files
raw_files = sorted(raw_dir.glob('raw_*.json'))
print(f'Found {len(raw_files)} raw files')
for f in raw_files:
    print(' -', f.name)

# %%
# Parse and filter successful sketches

def safe_name(text: str) -> str:
    # Keep alphanumerics, dash, underscore; replace others with underscore
    cleaned = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in text)
    return cleaned.strip('_') or 'unknown_item'

def detect_prefix(filename: str) -> str:
    name = filename.lower()
    if 'tpusg' in name:
        return 'tpusg'
    if 'psg' in name:
        return 'psg'
    if 'sg' in name:
        return 'sg'
    return 'unknown_item'

records = []
for fpath in raw_files:
    prefix = detect_prefix(fpath.name)
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        print(f'Failed to load {fpath.name}: {exc}')
        continue

    items: List[Dict[str, Any]] = payload.get('data', []) if isinstance(payload, dict) else []
    for entry in items:
        output = entry.get('output') if isinstance(entry, dict) else {}
        if not isinstance(output, dict):
            continue
        status = output.get('status')
        if not status or str(status).lower() != 'success':
            continue
        sketch = output.get('sketch')
        if not sketch:
            continue
        rec_id = entry.get('id') or entry.get('name') or fpath.stem
        records.append({
            'source_file': fpath.name,
            'id': rec_id,
            'sketch': sketch,
            'prefix': prefix,
        })

print(f'Collected {len(records)} sketches with status=="success"')

# %%
# Schema check: compare exported success counts with processed_data CSVs
processed_dir = base_dir / 'processed_data'
csv_files = sorted(processed_dir.rglob('*.csv'))

print(f'Found {len(csv_files)} processed CSV files')

record_counts = Counter([r['prefix'] for r in records])
aggregate_csv_success = Counter()
per_file_stats = []

def prefix_from_csv_name(name: str) -> str:
    n = name.lower()
    if 'tpusg' in n:
        return 'tpusg'
    if 'psg' in n:
        return 'psg'
    if 'sg' in n:
        return 'sg'
    return 'unknown_item'

for path in csv_files:
    prefix = prefix_from_csv_name(path.name)
    total_rows = 0
    success_rows = 0
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                if str(row.get('status', '')).lower() == 'success':
                    success_rows += 1
    except Exception as exc:
        print(f'Failed to read {path}: {exc}')
        continue
    aggregate_csv_success[prefix] += success_rows
    per_file_stats.append((path, prefix, success_rows, total_rows))

print('Success count comparison (records vs CSV):')
for prefix in ['tpusg', 'psg', 'sg', 'unknown_item']:
    print(f" {prefix}: records={record_counts.get(prefix, 0)} csv_success={aggregate_csv_success.get(prefix, 0)}")

print('\nPer-file CSV success counts:')
for path, prefix, success_rows, total_rows in per_file_stats:
    try:
        rel = path.relative_to(base_dir)
    except Exception:
        rel = path
    print(f' - {rel} [{prefix}] success={success_rows}/{total_rows}')

# %%
# Write extracted code to files (idempotent on reruns)
written = []
for idx, rec in enumerate(records, start=1):
    safe_id = safe_name(str(rec['id']))
    prefix = rec.get('prefix') or 'unknown_item'
    out_dir = export_dir / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f'{prefix}_{safe_id}_{dataset_suffix}' if safe_id else f'{prefix}_{idx}_{dataset_suffix}'
    ext = '.ino' if prefix == 'sg' else '.py'
    out_path = out_dir / f'{base_name}{ext}'

    sketch_content = rec['sketch']
    if out_path.exists():
        try:
            with open(out_path, 'r', encoding='utf-8') as fr:
                existing = fr.read()
            if existing == sketch_content:
                written.append(out_path)
                continue
        except Exception:
            pass

    with open(out_path, 'w', encoding='utf-8') as fw:
        fw.write(sketch_content)
    written.append(out_path)

print(f'Files accounted for under {export_dir}: {len(written)}')

# %%
# Trace-to-export mapping check
# Ensure every CSV success trace has a matching exported file and vice versa

# Handles filenames like sg_02cd0534_abla-l2-gpt5.ino or sg_02cd0534_abla-l2-gpt5_1.ino
# and psg/tpusg *.py equivalents.
def parse_trace_from_filename(path: Path):
    stem = path.stem
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None
    prefix = parts[0]
    # Handle optional numeric dedup suffix at end
    tail_parts = parts[1:]
    if tail_parts[-1].isdigit():
        tail_parts = tail_parts[:-1]
    # Expect the dataset suffix at the end
    if tail_parts and tail_parts[-1] == dataset_suffix:
        tail_parts = tail_parts[:-1]
    if not tail_parts:
        return None, None
    trace = '_'.join(tail_parts)
    return prefix, trace

# CSV success traces
csv_success_traces = set()
for path in csv_files:
    prefix = prefix_from_csv_name(path.name)
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get('status', '')).lower() == 'success':
                    trace = str(row.get('trace_id', '')).strip()
                    if trace:
                        csv_success_traces.add((prefix, trace))
    except Exception as exc:
        print(f'Failed to scan {path} for trace_id: {exc}')

# Records (success sketches parsed from raw_*)
record_traces = set((r['prefix'], safe_name(str(r['id']))) for r in records)

# Written files
written_traces = set()
for p in written:
    prefix, trace = parse_trace_from_filename(p)
    if prefix and trace:
        written_traces.add((prefix, trace))

missing_exports = csv_success_traces - written_traces
extra_exports = written_traces - csv_success_traces

print('Trace mapping check:')
print(f' CSV success traces: {len(csv_success_traces)}')
print(f' Exported files (parsed): {len(written_traces)}')
print(f' Missing exports for CSV success traces: {len(missing_exports)}')
print(f' Extra exports without CSV success rows: {len(extra_exports)}')

if missing_exports:
    print('\nMissing (prefix, trace_id):')
    for item in sorted(missing_exports):
        print(' -', item)

if extra_exports:
    print('\nExtra exports not found in CSV success:')
    for item in sorted(extra_exports):
        print(' -', item)

# %%
# Quick verification of exports
from itertools import islice

print('Example files:')
for p in written[:5]:
    try:
        rel = p.relative_to(base_dir)
    except Exception:
        rel = p
    print(' -', rel)

if written:
    sample = written[0]
    print(f"\nPreview of {sample.name}:")
    with open(sample, 'r', encoding='utf-8') as f:
        for line in islice(f, 10):
            print(line.rstrip())
else:
    print('No files written.')


