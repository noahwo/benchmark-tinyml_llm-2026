#!/usr/bin/env python3
"""Recover generation_count in all-MPU.csv from the original MPU records file."""

import argparse
import csv
import math
from pathlib import Path


def parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def parse_int(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def values_match(a, b):
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    return a == b


def load_original_rows(original_path):
    mapping = {}
    with original_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"trace_id", "latency", "total_tokens", "generation_count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Original CSV is missing columns: {missing}")

        for row in reader:
            trace_id = row.get("trace_id", "").strip()
            if not trace_id:
                continue
            mapping.setdefault(trace_id, []).append(row)

    return mapping


def load_gpt5_rows(gpt5_dir):
    mapping = {}
    gpt5_path = Path(gpt5_dir)
    if not gpt5_path.exists():
        print(f"WARNING: GPT-5 directory not found: {gpt5_path}")
        return mapping

    csv_files = sorted(gpt5_path.glob("*.csv"))
    for csv_file in csv_files:
        with csv_file.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"trace_id", "latency", "total_tokens", "generation_count"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"GPT-5 CSV {csv_file} is missing columns: {missing}")
            for row in reader:
                trace_id = row.get("trace_id", "").strip()
                if not trace_id:
                    continue
                mapping.setdefault(trace_id, []).append(row)

    return mapping


def find_matching_row(rows, target_latency, target_total_tokens):
    if not rows:
        return None
    exactly_matching = []
    for row in rows:
        orig_latency = parse_float(row.get("latency"))
        orig_total_tokens = parse_int(row.get("total_tokens"))
        if values_match(target_latency, orig_latency) and values_match(target_total_tokens, orig_total_tokens):
            exactly_matching.append(row)
    if len(exactly_matching) == 1:
        return exactly_matching[0]
    if len(exactly_matching) > 1:
        return exactly_matching[0]
    return None


def build_output_header(input_header):
    if "generation_count" in input_header:
        return input_header
    if "status" not in input_header:
        raise ValueError("Input CSV header does not contain 'status' column.")
    output_header = []
    for col in input_header:
        output_header.append(col)
        if col == "status":
            output_header.append("generation_count")
    return output_header


def recover_generation_counts(input_path, original_path, gpt5_dir, output_path, batch_size=60, inplace=False):
    original_map = load_original_rows(original_path)
    gpt5_map = load_gpt5_rows(gpt5_dir) if gpt5_dir else {}
    for trace_id, rows in gpt5_map.items():
        original_map.setdefault(trace_id, []).extend(rows)

    duplicate_trace_ids = [tid for tid, rows in original_map.items() if len(rows) > 1]
    if duplicate_trace_ids:
        print(f"WARNING: {len(duplicate_trace_ids)} trace_id values have multiple candidate source rows. Matching will validate latency and total_tokens.")

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        input_header = reader.fieldnames or []
        desired_header = [
            "name",
            "trace_id",
            "batch_id",
            "status",
            "latency",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
            "total_cost",
            "prompt_cost",
            "completion_cost",
            "parameters",
            "processor",
            "model",
        ]
        if input_header != desired_header:
            raise ValueError(
                f"Unexpected input header. Expected {desired_header!r}, got {input_header!r}."
            )

        output_header = build_output_header(input_header)
        rows = []
        stats = {
            "total": 0,
            "matched": 0,
            "missing_trace_id": 0,
            "no_original_match": 0,
            "mismatched": 0,
            "missing_generation_count": 0,
        }

        batch_rows = []
        for row in reader:
            stats["total"] += 1
            trace_id = row.get("trace_id", "").strip()
            row_out = {**row}
            row_out["generation_count"] = "N/A"

            if not trace_id:
                stats["missing_trace_id"] += 1
            else:
                candidate_rows = original_map.get(trace_id)
                if not candidate_rows:
                    stats["no_original_match"] += 1
                else:
                    in_latency = parse_float(row.get("latency"))
                    in_total_tokens = parse_int(row.get("total_tokens"))
                    match = find_matching_row(candidate_rows, in_latency, in_total_tokens)
                    if match is None:
                        stats["mismatched"] += 1
                    else:
                        generation_count = match.get("generation_count", "").strip()
                        if generation_count == "":
                            stats["missing_generation_count"] += 1
                        else:
                            row_out["generation_count"] = generation_count
                            stats["matched"] += 1

            rows.append(row_out)
            batch_rows.append(row_out)

            if stats["total"] % batch_size == 0:
                print_batch_summary(stats, batch_rows, stats["total"])
                batch_rows = []

        if batch_rows:
            print_batch_summary(stats, batch_rows, stats["total"], final=True)

    output_target = output_path if not inplace else input_path
    with output_target.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_header)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFinal summary:")
    print(f"  total rows processed: {stats['total']}")
    print(f"  matched rows: {stats['matched']}")
    print(f"  missing trace_id: {stats['missing_trace_id']}")
    print(f"  no original match: {stats['no_original_match']}")
    print(f"  mismatched latency/total_tokens: {stats['mismatched']}")
    print(f"  missing generation_count in original: {stats['missing_generation_count']}")
    if inplace:
        print(f"Updated in place: {output_target}")
    else:
        print(f"Written output CSV: {output_target}")


def print_batch_summary(stats, batch_rows, processed_count, final=False):
    label = "FINAL BATCH" if final else "BATCH"
    print(
        f"{label} processed up to row {processed_count}: "
        f"matched={stats['matched']}, "
        f"missing_trace_id={stats['missing_trace_id']}, "
        f"no_match={stats['no_original_match']}, "
        f"mismatched={stats['mismatched']}, "
        f"missing_generation_count={stats['missing_generation_count']}"
    )
    if stats["mismatched"]:
        print("  Note: some rows had matching trace_id but mismatched latency/total_tokens and were skipped.")


def main():
    parser = argparse.ArgumentParser(description="Recover generation_count in all-MPU.csv from original copy.")
    parser.add_argument("input_csv", type=Path, help="Path to cleaned all-MPU.csv")
    parser.add_argument("original_csv", type=Path, help="Path to original full MPU copy CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/all-MPU.recovered.csv"),
        help="Path to write the recovered CSV. Defaults to tmp/all-MPU.recovered.csv",
    )
    parser.add_argument(
        "--gpt5-dir",
        type=Path,
        default=Path("langfuse_export/2026/02.03_gpt5-MPU/processed_data/gpt-5-2025-08-07_40f1"),
        help="Path to the GPT-5 source CSV directory for recovering gpt-5 generation_count values.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input CSV in place instead of writing to a separate output file.",
    )
    args = parser.parse_args()

    if args.inplace:
        args.output = args.input_csv

    recover_generation_counts(args.input_csv, args.original_csv, args.gpt5_dir, args.output, inplace=args.inplace)


if __name__ == "__main__":
    main()
