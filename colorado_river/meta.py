#!/usr/bin/env python3
"""
meta.py — List USGS data sources

Usage:
  # List USGS sites discovered via the USGS site service (default)
  python meta.py [--format table|json] [--state CO [--state UT ...]] [--parameter 00060,00065] [--site-status active|all]

  # List only the configured sources from config.json
  python meta.py --mode config [--format table|json]

Defaults to pretty table output.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List

import httpx

from usgs import SITE_CATALOG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List available USGS sources")
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format",
    )
    parser.add_argument(
        "--mode",
        choices=["usgs", "config"],
        default="usgs",
        help="Data source: discover from USGS site service, or show configured only",
    )
    parser.add_argument(
        "--state",
        action="append",
        dest="states",
        default=["CO"],
        help="State codes to include (repeatable). Default: CO",
    )
    parser.add_argument(
        "--parameter",
        default="00060,00065",
        help="Comma-separated USGS parameter codes to filter (default: 00060,00065)",
    )
    parser.add_argument(
        "--site-status",
        choices=["active", "all"],
        default="active",
        help="Filter by site status (default: active)",
    )
    return parser.parse_args()


def load_sources() -> Dict[str, str]:
    # SITE_CATALOG is already loaded from config with fallback in usgs.py
    return SITE_CATALOG


def _parse_rdb(text: str) -> List[Dict[str, str]]:
    """Parse USGS RDB (tab-delimited) format into a list of dict rows."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    # Second line is data types or dashes; skip it
    rows: List[Dict[str, str]] = []
    for ln in lines[2:]:
        parts = ln.split("\t")
        if len(parts) != len(header):
            # Skip malformed lines
            continue
        rows.append({header[i]: parts[i] for i in range(len(header))})
    return rows


def discover_usgs_sites(states: Iterable[str], parameter: str, site_status: str) -> Dict[str, str]:
    """Query the USGS site service and return a mapping of label -> site_no.

    We request RDB format for robust parsing. We restrict to siteType=ST (stream),
    filter by state(s), parameter code(s), and site status.
    """
    base_url = "https://waterservices.usgs.gov/nwis/site/"
    # Normalize and deduplicate state codes
    norm_states = sorted({s.strip().upper() for s in states if s and s.strip()})

    params = {
        "format": "rdb",
        "siteType": "ST",
        "parameterCd": parameter,
        "siteStatus": site_status,
        # Using comma-separated states is supported by the service
        "stateCd": ",".join(norm_states),
    }
    with httpx.Client(timeout=60) as client:
        r = client.get(base_url, params=params)
        r.raise_for_status()
        rows = _parse_rdb(r.text)

    out: Dict[str, str] = {}
    for row in rows:
        site_no = row.get("site_no") or row.get("site_no ")
        station_nm = row.get("station_nm") or row.get("station_nm ")
        state_cd = row.get("state_cd") or row.get("state_cd ")
        if site_no and station_nm:
            label = f"{station_nm} ({site_no})"
            if state_cd:
                label = f"{station_nm}, {state_cd} ({site_no})"
            out[label] = site_no
    return out


def print_table(sources: Dict[str, str]) -> None:
    if not sources:
        print("No sources configured.")
        return
    # Simple fixed-width columns
    label_width = max(len(k) for k in sources.keys())
    code_width = max(len(v) for v in sources.values())
    header_label = "Label"
    header_code = "USGS Site Code"
    label_width = max(label_width, len(header_label))
    code_width = max(code_width, len(header_code))
    print(f"{header_label:<{label_width}}  {header_code:<{code_width}}")
    print(f"{'-' * label_width}  {'-' * code_width}")
    for label, code in sources.items():
        print(f"{label:<{label_width}}  {code:<{code_width}}")


def main() -> None:
    args = parse_args()
    if args.mode == "config":
        sources = load_sources()
    else:
        sources = discover_usgs_sites(args.states, args.parameter, args.site_status)
    if args.format == "json":
        print(json.dumps({"usgs_sources": sources}, indent=2))
    else:
        print_table(sources)


if __name__ == "__main__":
    main()


