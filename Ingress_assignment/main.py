"""
main.py — idempotent sensor-data ingress REST API + GraphQL exploration API.

Endpoints:
  PUT  /readings/{reading_id}   Idempotent ingress: caller names the id.
  POST /readings                Idempotent ingress: server derives the id
                                 from the reading's own content (sensor_id +
                                 metric + timestamp + value), so a retried/
                                 duplicate POST lands on the same record
                                 instead of creating a new one.
  GET  /readings/{reading_id}   Fetch one reading.
  GET  /healthz                 Liveness check (used by Cloud Run).
  POST /graphql, GET /graphql   GraphQL API (GraphiQL IDE in a browser;
                                 see app/graphql_schema.py for the schema).

Run locally:
  pixi run sensor-ingress
Then: POST/PUT to http://localhost:8080/readings..., browse
http://localhost:8080/graphql for the GraphiQL explorer.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Response, status
from strawberry.fastapi import GraphQLRouter

from app.graphql_schema import schema
from app.models import Reading, ReadingIn
from app.reading_id import compute_reading_id
from app.store import get_store

app = FastAPI(
    title="Sensor Ingress API",
    description="Idempotent ingress for simulated Qwiic-style sensor readings.",
    version="1.0.0",
)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.put("/readings/{reading_id}", response_model=Reading)
def put_reading(reading_id: str, reading_in: ReadingIn, response: Response) -> Reading:
    """Idempotent ingress with a caller-supplied id.

    This is the textbook idempotent REST verb: PUT the same `reading_id`
    with the same body any number of times and the end state — and the
    response — is identical. `response.status_code` is 201 the first time
    and 200 on every replay, which is the conventional way to surface "was
    this actually new?" without changing the body shape.
    """
    stored = get_store().upsert(reading_id, reading_in)
    response.status_code = status.HTTP_201_CREATED if stored.newly_created else status.HTTP_200_OK
    return stored


@app.post("/readings", response_model=Reading)
def post_reading(reading_in: ReadingIn, response: Response) -> Reading:
    """Idempotent ingress with a server-derived id (no id required from the
    caller). The id is a content hash of sensor_id + metric + timestamp +
    value, so re-sending the exact same reading — e.g. because a device
    retried after a timed-out response — is automatically deduplicated.
    """
    reading_id = compute_reading_id(
        reading_in.sensor_id,
        reading_in.metric,
        reading_in.timestamp.isoformat(),
        reading_in.value,
    )
    stored = get_store().upsert(reading_id, reading_in)
    response.status_code = status.HTTP_201_CREATED if stored.newly_created else status.HTTP_200_OK
    return stored


@app.get("/readings/{reading_id}", response_model=Reading)
def get_reading(reading_id: str) -> Reading:
    found = get_store().get(reading_id)
    if found is None:
        raise HTTPException(status_code=404, detail="reading not found")
    return found


@app.get("/readings", response_model=list[Reading])
def list_readings(
    sensor_id: Optional[str] = None,
    metric: Optional[str] = None,
    limit: int = 100,
) -> list[Reading]:
    return get_store().list(sensor_id=sensor_id, metric=metric, limit=limit)
