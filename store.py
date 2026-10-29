"""
store.py — idempotent storage for sensor readings, with two backends.

Both backends expose the exact same interface (`ReadingStore`), so
`app/main.py` and the GraphQL resolvers never need to know which one is
active. That's what lets the whole thing be developed and tested entirely
offline, then pointed at real Firestore just by setting an environment
variable before deploying.

- LocalJSONStore: a dict in memory, snapshotted to a local JSON file after
  every write. Used for local dev (`pixi run sensor-ingress`) and for the
  test suite — no cloud credentials required.
- FirestoreStore: the real backend used once this is deployed to Cloud Run.
  Idempotency comes directly from Firestore's `document(id).set(data)`:
  writing the same id with the same data twice leaves Firestore in the same
  state either way — no duplicate documents, no error on the second write.

Which backend is active is controlled by the INGRESS_STORE_BACKEND
environment variable ("local" or "firestore"); see `get_store()` at the
bottom.
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.models import Reading, ReadingIn
from app.reading_id import compute_reading_id

DEFAULT_LOCAL_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "readings.local.json"


class ReadingStore(ABC):
    """Storage interface implemented by both backends."""

    @abstractmethod
    def upsert(self, reading_id: str, reading_in: ReadingIn) -> Reading:
        """Idempotently write a reading under `reading_id`.

        Calling this twice with the same `reading_id` and equal field values
        must leave storage in the same state and must NOT create a second
        record. `Reading.newly_created` tells the caller whether this call
        was the first write (True) or a replay of an existing one (False).
        """

    @abstractmethod
    def get(self, reading_id: str) -> Optional[Reading]:
        ...

    @abstractmethod
    def list(
        self,
        sensor_id: Optional[str] = None,
        metric: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Reading]:
        """Return readings, newest first, optionally filtered."""

    @abstractmethod
    def sensor_ids(self) -> list[str]:
        """Return the distinct sensor_ids seen so far."""


def _reading_from_stored(data: dict) -> Reading:
    return Reading(**data)


class LocalJSONStore(ReadingStore):
    """In-memory store, persisted to a single JSON file for durability.

    Not meant for concurrent multi-process use (that's what Firestore is
    for) — a thread lock is enough to make it safe within one process, which
    is all the local dev server and the test suite need.
    """

    def __init__(self, path: Optional[Path] = None):
        # Resolved at call time (not bound as a default-argument value at
        # import time), so tests can monkeypatch the module-level
        # DEFAULT_LOCAL_STORE_PATH and have it actually take effect here.
        self._path = path if path is not None else DEFAULT_LOCAL_STORE_PATH
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    def upsert(self, reading_id: str, reading_in: ReadingIn) -> Reading:
        with self._lock:
            existing = self._data.get(reading_id)
            if existing is not None:
                # Idempotent replay: return what's already there unchanged,
                # don't touch ingested_at, don't write to disk again.
                existing_reading = _reading_from_stored(existing)
                return existing_reading.model_copy(update={"newly_created": False})

            reading = Reading(
                **reading_in.model_dump(),
                reading_id=reading_id,
                ingested_at=datetime.now(timezone.utc),
                newly_created=True,
            )
            self._data[reading_id] = json.loads(reading.model_dump_json())
            self._save()
            return reading

    def get(self, reading_id: str) -> Optional[Reading]:
        with self._lock:
            data = self._data.get(reading_id)
            return _reading_from_stored(data) if data else None

    def list(
        self,
        sensor_id: Optional[str] = None,
        metric: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Reading]:
        with self._lock:
            readings = [_reading_from_stored(d) for d in self._data.values()]
        if sensor_id is not None:
            readings = [r for r in readings if r.sensor_id == sensor_id]
        if metric is not None:
            readings = [r for r in readings if r.metric == metric]
        if since is not None:
            readings = [r for r in readings if r.timestamp >= since]
        readings.sort(key=lambda r: r.timestamp, reverse=True)
        return readings[:limit]

    def sensor_ids(self) -> list[str]:
        with self._lock:
            return sorted({d["sensor_id"] for d in self._data.values()})


class FirestoreStore(ReadingStore):
    """Firestore-backed store, used once deployed to Cloud Run.

    Imports `google.cloud.firestore` lazily (inside __init__) so that the
    local dev / test path never needs the `google-cloud-firestore` package
    installed or any GCP credentials configured.
    """

    COLLECTION = "readings"

    def __init__(self, project: Optional[str] = None):
        from google.cloud import firestore  # noqa: PLC0415 (intentional lazy import)

        self._client = firestore.Client(project=project)
        self._collection = self._client.collection(self.COLLECTION)

    def upsert(self, reading_id: str, reading_in: ReadingIn) -> Reading:
        doc_ref = self._collection.document(reading_id)
        existing = doc_ref.get()
        if existing.exists:
            existing_reading = _reading_from_stored(existing.to_dict())
            return existing_reading.model_copy(update={"newly_created": False})

        reading = Reading(
            **reading_in.model_dump(),
            reading_id=reading_id,
            ingested_at=datetime.now(timezone.utc),
            newly_created=True,
        )
        # .set() is the idempotent primitive here: writing the same
        # reading_id with the same payload again is a no-op in effect (we
        # never reach this line on a replay because of the .exists check
        # above, but even without that check, .set() would just overwrite
        # with identical data rather than erroring or duplicating).
        doc_ref.set(json.loads(reading.model_dump_json()))
        return reading

    def get(self, reading_id: str) -> Optional[Reading]:
        doc = self._collection.document(reading_id).get()
        return _reading_from_stored(doc.to_dict()) if doc.exists else None

    def list(
        self,
        sensor_id: Optional[str] = None,
        metric: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Reading]:
        query = self._collection
        if sensor_id is not None:
            query = query.where("sensor_id", "==", sensor_id)
        if metric is not None:
            query = query.where("metric", "==", metric)
        if since is not None:
            query = query.where("timestamp", ">=", since.isoformat())
        query = query.order_by("timestamp", direction="DESCENDING").limit(limit)
        return [_reading_from_stored(doc.to_dict()) for doc in query.stream()]

    def sensor_ids(self) -> list[str]:
        # Small-scale demo approach: pull distinct values client-side.
        # At real scale you'd maintain a separate "sensors" collection
        # instead of scanning every reading.
        docs = self._collection.select(["sensor_id"]).stream()
        return sorted({doc.to_dict()["sensor_id"] for doc in docs})


_store_instance: Optional[ReadingStore] = None
_store_lock = threading.Lock()


def get_store() -> ReadingStore:
    """Return the process-wide store, chosen by INGRESS_STORE_BACKEND.

    - "local" (default): LocalJSONStore — no external dependencies.
    - "firestore": FirestoreStore — used on Cloud Run (see Dockerfile / terraform).
    """
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            backend = os.environ.get("INGRESS_STORE_BACKEND", "local").lower()
            if backend == "firestore":
                _store_instance = FirestoreStore(project=os.environ.get("GCP_PROJECT"))
            elif backend == "local":
                _store_instance = LocalJSONStore()
            else:
                raise ValueError(f"Unknown INGRESS_STORE_BACKEND: {backend!r}")
        return _store_instance


def reset_store_for_tests() -> None:
    """Force a fresh store on the next get_store() call. Test-only."""
    global _store_instance
    with _store_lock:
        _store_instance = None
