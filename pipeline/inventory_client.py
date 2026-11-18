"""Client for the inventory service, plus a local mock of it.

The real service returns live remaining seats and the current selling price
per event and tier. It is slow and flaky, so callers need to cope with
timeouts and errors.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import (
    API_ERROR_RATE,
    API_MAX_RETRIES,
    API_TIMEOUT_RATE,
    API_TIMEOUT_SECONDS,
    SEED,
)

log = logging.getLogger(__name__)


class InventoryAPIError(Exception):
    pass


@dataclass
class Response:
    status_code: int
    body: dict

    def json(self) -> dict:
        return self.body


class MockInventoryAPI:
    """Serves `inventory_snapshot.json` and fails some of the time, like the real thing."""

    def __init__(
        self,
        snapshot_path: Path,
        seed: int = SEED,
        timeout_rate: float = API_TIMEOUT_RATE,
        error_rate: float = API_ERROR_RATE,
    ):
        snapshot = json.loads(Path(snapshot_path).read_text())
        self._items = {(int(item["event_id"]), item["tier"]): item for item in snapshot["items"]}
        self._rng = random.Random(seed)
        self.timeout_rate = timeout_rate
        self.error_rate = error_rate
        self.calls = 0

    def get(self, path: str, timeout: float) -> Response:
        self.calls += 1
        _, resource, event_id, tier = path.split("/")
        roll = self._rng.random()
        if roll < self.timeout_rate:
            raise TimeoutError(f"no response from {path} within {timeout}s")
        if roll < self.timeout_rate + self.error_rate:
            return Response(500, {"error": "internal server error"})

        if resource != "inventory":
            return Response(404, {"error": f"no such resource {resource}"})
        try:
            key = (int(event_id), tier)
        except ValueError:
            return Response(400, {"error": f"event_id must be an integer, got {event_id!r}"})
        if key not in self._items:
            return Response(400, {"error": f"unknown event/tier {event_id}/{tier}"})
        return Response(200, dict(self._items[key]))


class InventoryClient:
    def __init__(self, transport, max_retries: int = API_MAX_RETRIES, timeout: float = API_TIMEOUT_SECONDS):
        self.transport = transport
        self.max_retries = max_retries
        self.timeout = timeout

    def get_inventory(self, event_id, tier: str) -> dict:
        """Return {"remaining": int, "current_price": float, ...} for one event/tier."""
        path = f"/inventory/{event_id}/{tier}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            started = time.monotonic()
            try:
                response = self.transport.get(path, timeout=self.timeout)
            except TimeoutError as exc:
                last_error = exc
                log.debug("attempt %d for %s timed out", attempt, path)
                continue

            elapsed_ms = (time.monotonic() - started) * 1000
            if response.status_code == 200:
                log.debug("%s ok in %.1fms after %d attempt(s)", path, elapsed_ms, attempt)
                return response.json()

            last_error = InventoryAPIError(f"{path} returned {response.status_code}: {response.body}")
            log.debug("attempt %d for %s failed with %d", attempt, path, response.status_code)

        raise InventoryAPIError(f"giving up on {path} after {self.max_retries} attempts") from last_error
