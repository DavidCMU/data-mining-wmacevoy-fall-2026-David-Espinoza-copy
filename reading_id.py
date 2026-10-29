"""
reading_id.py — deterministic, content-addressed IDs for sensor readings.

This is the core trick that makes the ingress endpoint idempotent even when
a device/simulator uses plain POST instead of a client-supplied PUT id: the
ID is *derived from the reading's own content*, so submitting the exact same
reading twice (e.g. a retried request after a dropped response) always maps
to the same document instead of creating a duplicate.

Two readings are considered "the same" if they share:
- sensor_id   (which physical/simulated device produced it)
- metric      (what's being measured, e.g. "temperature_c")
- timestamp   (when it was measured — ISO 8601 string, second precision)
- value       (rounded to a fixed precision to avoid float-formatting noise)

Changing the measurement (same sensor/metric/time but a corrected value)
intentionally produces a *different* id — that's a new, distinct event, not
a duplicate of the old one.
"""
from __future__ import annotations

import hashlib

# Round floats to this many decimal places before hashing, so that
# 21.400000000000002 and 21.4 (same reading, different float noise) hash the
# same way.
VALUE_PRECISION = 6


def compute_reading_id(sensor_id: str, metric: str, timestamp: str, value: float) -> str:
    """Return a short, deterministic, content-addressed id for a reading.

    `timestamp` must already be a normalized ISO 8601 string (callers should
    use `datetime.isoformat()` on a timezone-aware UTC datetime) so that two
    equivalent timestamps written slightly differently don't produce
    different ids.
    """
    canonical_value = f"{round(float(value), VALUE_PRECISION):.{VALUE_PRECISION}f}"
    canonical = "|".join([sensor_id, metric, timestamp, canonical_value])
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    # Full sha256 hex (64 chars) is overkill for a demo but collision-proof;
    # trim to 32 chars for shorter URLs/ids while keeping it effectively
    # unique for this use case.
    return digest[:32]
