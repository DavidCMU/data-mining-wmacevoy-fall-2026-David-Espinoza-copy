#!/usr/bin/env python3
"""
px.py — Simple Parquet Data Explorer (CLI)

Goals for absolute beginners:
- Inspect one or more Parquet files from the command line
- See columns, dtypes, schema, head/tail, and basic stats
- Optionally filter and select columns without writing extra code

Usage examples:
  python px.py data/09163500_dv_5y.parquet --info --head 10
  python px.py data/*.parquet --columns
  python px.py data/09095500_iv_7d.parquet --select time,discharge_cfs --head 5
  python px.py data/09095500_iv_7d.parquet --where "discharge_cfs > 1000" --head 10
  python px.py data/09095500_iv_7d.parquet --time-col time --start 2025-08-10 --end 2025-08-12 --head 20

Notes:
- This script uses pandas + pyarrow (already in this project).
- For filtering, you can use --where with pandas.query() syntax, and/or use
  --time-col with --start/--end ISO-like timestamps.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, List, Optional

import pandas as pd
from dateutil import parser as dtp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parquet data explorer (CLI)")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more file paths or globs (e.g., data/*.parquet)",
    )

    # What to print
    parser.add_argument("--schema", action="store_true", help="Print pyarrow-like schema")
    parser.add_argument("--columns", action="store_true", help="Print column names and dtypes")
    parser.add_argument("--info", action="store_true", help="Print basic info (rows, columns, size)")
    parser.add_argument("--describe", action="store_true", help="Print pandas describe() summary")

    # How much to show
    parser.add_argument("--head", type=int, default=0, help="Show first N rows")
    parser.add_argument("--tail", type=int, default=0, help="Show last N rows")
    parser.add_argument("--sample", type=int, default=0, help="Show a random sample of N rows")

    # Column selection and filtering
    parser.add_argument(
        "--select",
        type=str,
        default="",
        help="Comma-separated columns to include (e.g., time,discharge_cfs)",
    )
    parser.add_argument(
        "--where",
        type=str,
        default="",
        help="Row filter using pandas.query() syntax (e.g., 'discharge_cfs > 1000')",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="",
        help="Name of a datetime column to print min/max and apply --start/--end",
    )
    parser.add_argument("--start", type=str, default="", help="Filter rows where time-col >= this")
    parser.add_argument("--end", type=str, default="", help="Filter rows where time-col <= this")

    # Display formatting
    parser.add_argument("--max-rows", type=int, default=60, help="Max rows to print in tables")
    parser.add_argument("--max-cols", type=int, default=20, help="Max columns to print in tables")

    return parser.parse_args()


def expand_paths(paths: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for p in paths:
        # Accept globs (e.g., data/*.parquet)
        matches = glob.glob(p)
        expanded.extend(matches if matches else [p])
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for p in expanded:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def read_parquet_frame(file_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    # Use pandas with pyarrow engine; columns pruning if specified
    return pd.read_parquet(file_path, columns=columns)


def print_schema(df: pd.DataFrame) -> None:
    # Show columns and inferred dtype from pandas
    print("Columns and dtypes:")
    for name, dtype in df.dtypes.items():
        print(f"  - {name}: {dtype}")


def print_columns(df: pd.DataFrame) -> None:
    print(", ".join(df.columns))


def print_info(df: pd.DataFrame, file_path: str) -> None:
    num_rows, num_cols = df.shape
    size_bytes = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    print(f"File: {file_path}")
    print(f"  Size: {size_bytes:,} bytes")
    print(f"  Shape: {num_rows:,} rows x {num_cols:,} cols")


def ensure_datetime_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if column_name not in df.columns:
        return df
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        coerced = pd.to_datetime(df[column_name], errors="coerce")
        df = df.copy()
        df[column_name] = coerced
    return df


def main() -> None:
    args = parse_args()

    # Configure pandas display for terminal readability
    pd.set_option("display.max_rows", args.max_rows)
    pd.set_option("display.max_columns", args.max_cols)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 80)

    paths = expand_paths(args.paths)
    if not paths:
        print("No files matched.")
        return

    selected_columns: Optional[List[str]] = None
    if args.select.strip():
        selected_columns = [c.strip() for c in args.select.split(",") if c.strip()]

    for file_path in paths:
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue

        try:
            df = read_parquet_frame(file_path, columns=selected_columns)
        except Exception as exc:
            print(f"Failed to read {file_path}: {exc}")
            continue

        print("=" * 80)
        print_info(df, file_path)

        # If requested, try to interpret a time column and show min/max
        if args.time_col:
            df = ensure_datetime_column(df, args.time_col)
            if args.time_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[args.time_col]):
                tmin = df[args.time_col].min()
                tmax = df[args.time_col].max()
                print(f"  {args.time_col} range: {tmin} .. {tmax}")

        # Optional time-based filtering
        if args.time_col and (args.start or args.end):
            df = df.copy()
            if args.start:
                try:
                    t0 = dtp.isoparse(args.start)
                    df = df[df[args.time_col] >= t0]
                except Exception:
                    print(f"  WARN: could not parse --start={args.start}")
            if args.end:
                try:
                    t1 = dtp.isoparse(args.end)
                    df = df[df[args.time_col] <= t1]
                except Exception:
                    print(f"  WARN: could not parse --end={args.end}")

        # Optional query filtering
        if args.where.strip():
            try:
                df = df.query(args.where)
            except Exception as exc:
                print(f"  WARN: query failed ({exc}); skipping --where")

        if args.schema:
            print_schema(df)

        if args.columns:
            print("Columns:")
            print_columns(df)

        if args.describe:
            # include='all' to also show non-numeric summaries
            try:
                desc = df.describe(include="all", datetime_is_numeric=True)
            except TypeError:
                # Older pandas may not support datetime_is_numeric
                desc = df.describe(include="all")
            print("\nDescribe():")
            print(desc)

        # Row views
        if args.head > 0:
            print(f"\nHead({args.head}):")
            print(df.head(args.head))
        if args.tail > 0:
            print(f"\nTail({args.tail}):")
            print(df.tail(args.tail))
        if args.sample > 0 and len(df) > 0:
            print(f"\nSample({args.sample}):")
            print(df.sample(min(args.sample, len(df)), random_state=0))

    print("=" * 80)


if __name__ == "__main__":
    main()


