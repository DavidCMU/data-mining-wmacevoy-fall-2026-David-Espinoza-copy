"""
simulator.py — stand-in for a physical Qwiic sensor (e.g. a SparkFun
Qwiic BME280 temperature/humidity/pressure sensor over I2C).

Real hardware is optional for this assignment — this script generates
readings with the same shape a real Qwiic sensor driver would produce, and
pushes them through the REST ingress API exactly like a real device would.
It also demonstrates idempotency directly: it can replay the last reading
it sent and show that the ingress API reports it as a duplicate rather than
creating a new record.

Usage:
  pixi run sensor-ingress-sim                      # 5 readings, 2s apart
  pixi run sensor-ingress-sim -- --count 20 --interval 0.5
  pixi run sensor-ingress-sim -- --url https://<cloud-run-url> --replay-last
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone

import httpx

SENSOR_ID = "qwiic-sim-01"
METRICS = [
    ("temperature_c", "celsius", 20.0, 2.0),
    ("humidity_pct", "percent", 45.0, 5.0),
    ("pressure_hpa", "hectopascal", 1013.0, 3.0),
]


def _fake_reading(metric: str, unit: str, mean: float, spread: float) -> dict:
    return {
        "sensor_id": SENSOR_ID,
        "metric": metric,
        "unit": unit,
        "value": round(random.gauss(mean, spread), 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run(base_url: str, count: int, interval: float, replay_last: bool) -> None:
    last_payload: dict | None = None
    with httpx.Client(base_url=base_url, timeout=10) as client:
        for i in range(count):
            for metric, unit, mean, spread in METRICS:
                payload = _fake_reading(metric, unit, mean, spread)
                resp = client.post("/readings", json=payload)
                resp.raise_for_status()
                body = resp.json()
                print(
                    f"[{i + 1}/{count}] POST /readings -> {resp.status_code} "
                    f"{body['metric']}={body['value']}{' (new)' if body['newly_created'] else ' (duplicate, unchanged)'} "
                    f"id={body['reading_id']}"
                )
                last_payload = payload
            if i < count - 1:
                time.sleep(interval)

        if replay_last and last_payload is not None:
            print("\nReplaying the last reading unchanged to demonstrate idempotency...")
            resp = client.post("/readings", json=last_payload)
            resp.raise_for_status()
            body = resp.json()
            assert body["newly_created"] is False, "expected the replay to be recognized as a duplicate"
            print(f"POST /readings (replay) -> {resp.status_code}, newly_created={body['newly_created']} (as expected)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a Qwiic-style multi-sensor device.")
    parser.add_argument("--url", default="http://localhost:8080", help="Ingress API base URL")
    parser.add_argument("--count", type=int, default=5, help="Number of reading rounds to send")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between rounds")
    parser.add_argument(
        "--replay-last",
        action="store_true",
        help="After sending, resend the last reading to demonstrate idempotent dedup",
    )
    args = parser.parse_args()
    run(args.url, args.count, args.interval, args.replay_last)


if __name__ == "__main__":
    main()
