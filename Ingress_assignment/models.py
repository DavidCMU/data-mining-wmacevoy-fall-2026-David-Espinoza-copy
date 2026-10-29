"""
models.py — the shape of one sensor reading, shared by the REST ingress,
the storage layer, and the GraphQL API.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class ReadingIn(BaseModel):
    """What a device/simulator sends to the ingress API.

    `timestamp` defaults to "now" so a simulator can omit it, but a real
    device should always send the time it took the measurement, not the
    time it happened to get a network connection to send it.
    """

    sensor_id: str = Field(..., examples=["qwiic-sim-01"])
    metric: str = Field(..., examples=["temperature_c"])
    value: float = Field(..., examples=[21.4])
    unit: str = Field(..., examples=["celsius"])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("timestamp")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime) -> datetime:
        # Naive datetimes are a classic source of silent bugs once this data
        # gets compared against other sources (see colorado_river/eda.py and
        # blue_mesa_reservoir's monthly resampling for the same lesson).
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class Reading(ReadingIn):
    """A reading as stored: the input fields plus server-assigned metadata."""

    reading_id: str
    ingested_at: datetime
    # True the *first* time this reading_id was written; False on every
    # later idempotent replay of the same id. Lets a caller tell "this is
    # new data" apart from "this is a retry of something I already have."
    newly_created: bool = True
