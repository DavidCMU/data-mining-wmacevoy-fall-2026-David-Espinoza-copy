"""
graphql_schema.py — read-only GraphQL API for exploring ingested readings.

Ingestion stays on the REST side (PUT/POST /readings) on purpose: REST with
explicit idempotency semantics is the right tool for writes, GraphQL is the
right tool for flexible, client-shaped reads. This schema only ever reads
from the store.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import strawberry

from app.store import get_store


@strawberry.type
class Reading:
    reading_id: str
    sensor_id: str
    metric: str
    value: float
    unit: str
    timestamp: dt.datetime
    ingested_at: dt.datetime


def _to_gql(reading) -> Reading:
    return Reading(
        reading_id=reading.reading_id,
        sensor_id=reading.sensor_id,
        metric=reading.metric,
        value=reading.value,
        unit=reading.unit,
        timestamp=reading.timestamp,
        ingested_at=reading.ingested_at,
    )


@strawberry.type
class Query:
    @strawberry.field(description="Distinct sensor IDs seen so far.")
    def sensors(self) -> list[str]:
        return get_store().sensor_ids()

    @strawberry.field(description="Look up one reading by its id.")
    def reading(self, reading_id: str) -> Optional[Reading]:
        found = get_store().get(reading_id)
        return _to_gql(found) if found else None

    @strawberry.field(
        description=(
            "Readings, newest first. Filter by sensor_id and/or metric, "
            "and/or only readings at/after `since`. `limit` caps the count."
        )
    )
    def readings(
        self,
        sensor_id: Optional[str] = None,
        metric: Optional[str] = None,
        since: Optional[dt.datetime] = None,
        limit: int = 100,
    ) -> list[Reading]:
        results = get_store().list(sensor_id=sensor_id, metric=metric, since=since, limit=limit)
        return [_to_gql(r) for r in results]


schema = strawberry.Schema(query=Query)
