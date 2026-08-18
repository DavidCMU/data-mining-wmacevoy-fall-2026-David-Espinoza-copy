"""
USGS client (beginner-friendly walkthrough)

This module talks to the USGS Water Services API. Two flavors:
- IV (Instantaneous Values): high-frequency measurements.
- DV (Daily Values): daily aggregates (mean/min/max).

We use httpx to download JSON and pandas to shape the data into DataFrames.
Caching: we save results as Parquet files in ./data so we don't refetch each time.

Important concepts:
- A pandas DataFrame uses an index (like row labels). For time series, the
  index is often a DatetimeIndex so that sorting and resampling are easy.
- Timezones matter. USGS returns timestamps that may include offsets.
  We parse them and normalize to UTC internally for consistency.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import httpx
import pandas as pd
from dateutil import parser as dtp

# Base endpoints for the USGS Water Services API
USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
USGS_DV_URL = "https://waterservices.usgs.gov/nwis/dv/"

# Parameter codes we care about (see USGS parameter catalog)
DISCHARGE_CFS = "00060"  # discharge in cubic feet per second
STAGE_FT = "00065"        # gage height in feet
WATER_TEMP_C = "00010"    # water temperature in degrees Celsius
SALINITY_PPT = "00480"    # salinity, parts per thousand (ppt)

def _load_site_catalog() -> Dict[str, str]:
    """Load site catalog from config.json if present, else fallback.

    The config format is expected to be:
      { "usgs_sources": { "Label": "site_code", ... } }
    """
    default_catalog: Dict[str, str] = {
        "Colorado River near Cameo (09095500)": "09095500",
        "Gunnison River near Grand Junction (09152500)": "09152500",
        "Colorado River at CO–UT State Line (09163500)": "09163500",
    }
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            sources = cfg.get("usgs_sources")
            if isinstance(sources, dict) and sources:
                # Ensure keys/values are strings
                cleaned: Dict[str, str] = {}
                for k, v in sources.items():
                    if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                        cleaned[k] = v
                if cleaned:
                    return cleaned
    except Exception:
        # Fall back silently on any error
        pass
    return default_catalog


# Catalog of nearby gauges, potentially loaded from config.json
SITE_CATALOG: Dict[str, str] = _load_site_catalog()

# Where we cache data locally. Parquet is compact and fast with pandas>=2.
# Anchored to this file rather than the current working directory, so the cache
# always lands in colorado_river/data/ no matter where the process was started.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _nwis_request(url: str, params: Dict) -> Dict:
    """Make an HTTP GET request and return JSON.

    Beginner notes:
    - We use a context-managed httpx.Client so the connection is cleaned up.
    - raise_for_status() makes HTTP errors become Python exceptions, which is
      helpful for debugging.
    """
    with httpx.Client(timeout=60) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def fetch_iv(
    site: str,
    start: datetime,
    end: datetime,
    parameters: Tuple[str, ...] = (DISCHARGE_CFS, STAGE_FT, WATER_TEMP_C, SALINITY_PPT),
) -> pd.DataFrame:
    """Fetch Instantaneous Values (IV) for a time window, return as DataFrame.

    What you get:
    - A DataFrame indexed by UTC timestamps.
    - Columns such as `discharge_cfs` (flow) and `stage_ft` (gage height).
    """
    params = {
        "format": "json",
        "sites": site,
        "parameterCd": ",".join(parameters),
        # USGS accepts ISO-like strings; Z indicates UTC.
        "startDT": start.strftime("%Y-%m-%dT%H:%MZ"),
        "endDT": end.strftime("%Y-%m-%dT%H:%MZ"),
        "siteStatus": "all",
    }
    js = _nwis_request(USGS_IV_URL, params)

    frames = []
    # USGS JSON groups data by variable. We turn each variable's list of
    # measurements into a small DataFrame, then concatenate them side-by-side.
    for ts in js.get("value", {}).get("timeSeries", []):
        variable_code = ts["variable"]["variableCode"][0]["value"]
        values = ts["values"][0]["value"]
        recs = []
        for v in values:
            # Parse date string and normalize to UTC (timezone-aware)
            t = dtp.isoparse(v["dateTime"]).astimezone(timezone.utc)
            # Convert missing/empty strings to None; otherwise float
            val = float(v["value"]) if v["value"] not in ("", None) else None
            recs.append({"time": t, variable_code: val})
        if recs:
            frames.append(pd.DataFrame(recs).set_index("time").sort_index())

    if not frames:
        # Return an empty, tz-aware frame for consistency in callers
        return pd.DataFrame(index=pd.DatetimeIndex([], tz=timezone.utc))

    out = pd.concat(frames, axis=1).sort_index()
    # Replace numeric parameter codes with friendly column names
    rename = {
        DISCHARGE_CFS: "discharge_cfs",
        STAGE_FT: "stage_ft",
        WATER_TEMP_C: "water_temp_c",
        SALINITY_PPT: "salinity_ppt",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


def fetch_dv(
    site: str,
    start: datetime,
    end: datetime,
    stat_code: str = "00003",  # 00003 = mean; 00001=min; 00002=max
    parameter: str = DISCHARGE_CFS,
) -> pd.DataFrame:
    """Fetch Daily Values (DV) for a time window, return as DataFrame.

    What you get:
    - A DataFrame indexed by date (no timezone) with a single column
      (e.g., `discharge_cfs`).
    """
    params = {
        "format": "json",
        "sites": site,
        "parameterCd": parameter,
        "statCd": stat_code,
        "startDT": start.strftime("%Y-%m-%d"),
        "endDT": end.strftime("%Y-%m-%d"),
    }
    js = _nwis_request(USGS_DV_URL, params)

    frames = []
    for ts in js.get("value", {}).get("timeSeries", []):
        variable_code = ts["variable"]["variableCode"][0]["value"]
        values = ts["values"][0]["value"]
        recs = []
        for v in values:
            # DV uses date-only strings (no timezone)
            d = dtp.isoparse(v["dateTime"]).date()
            val = float(v["value"]) if v["value"] not in ("", None) else None
            recs.append({"date": d, variable_code: val})
        if recs:
            frames.append(pd.DataFrame(recs).set_index("date").sort_index())

    if not frames:
        return pd.DataFrame(index=pd.Index([], name="date"))

    out = pd.concat(frames, axis=1).sort_index()
    rename = {
        DISCHARGE_CFS: "discharge_cfs",
        WATER_TEMP_C: "water_temp_c",
        SALINITY_PPT: "salinity_ppt",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    return out


# ------------------------ Debug helpers ------------------------
def fetch_iv_json(
    site: str,
    start: datetime,
    end: datetime,
    parameters: Tuple[str, ...] = (DISCHARGE_CFS, STAGE_FT),
) -> Dict:
    """Return the raw JSON payload for IV for debugging/inspection."""
    params = {
        "format": "json",
        "sites": site,
        "parameterCd": ",".join(parameters),
        "startDT": start.strftime("%Y-%m-%dT%H:%MZ"),
        "endDT": end.strftime("%Y-%m-%dT%H:%MZ"),
        "siteStatus": "all",
    }
    return _nwis_request(USGS_IV_URL, params)


def fetch_dv_json(
    site: str,
    start: datetime,
    end: datetime,
    stat_code: str = "00003",
    parameter: str = DISCHARGE_CFS,
) -> Dict:
    """Return the raw JSON payload for DV for debugging/inspection."""
    params = {
        "format": "json",
        "sites": site,
        "parameterCd": parameter,
        "statCd": stat_code,
        "startDT": start.strftime("%Y-%m-%d"),
        "endDT": end.strftime("%Y-%m-%d"),
    }
    return _nwis_request(USGS_DV_URL, params)


def _cache_path(site: str, tag: str) -> str:
    """Build a cache filename based on site and a tag (e.g., 'iv_7d').

    Creates the cache directory on demand. Importing this module should not
    have side effects, so the directory is made here rather than at import.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    safe = site.replace("/", "-")
    return os.path.join(DATA_DIR, f"{safe}_{tag}.parquet")


def load_or_fetch_iv(site: str, days: int = 7) -> pd.DataFrame:
    """Fetch IV data, caching to Parquet (./data/)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    # Include 'all' to reflect the expanded parameter set in cache key
    path = _cache_path(site, f"iv_all_{days}d")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = fetch_iv(site, start, end)
    df.to_parquet(path)
    return df


def load_or_fetch_dv(site: str, years: int = 5, parameter: str = DISCHARGE_CFS, stat_code: str = "00003") -> pd.DataFrame:
    """Fetch DV data (for a single parameter), caching to Parquet (./data/)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)
    path = _cache_path(site, f"dv_{parameter}_{stat_code}_{years}y")
    if os.path.exists(path):
        return pd.read_parquet(path)
    df = fetch_dv(site, start, end, stat_code=stat_code, parameter=parameter)
    df.to_parquet(path)
    return df
