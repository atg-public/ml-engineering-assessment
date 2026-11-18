import pytest

from pipeline import inventory_client
from pipeline.inventory_client import InventoryAPIError, InventoryClient, Response


class ScriptedTransport:
    """Returns (or raises) the scripted outcomes in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, path, timeout):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


OK = Response(200, {"event_id": 101, "tier": "A", "remaining": 40, "current_price": 95.0})


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(inventory_client.time, "sleep", recorded.append)
    return recorded


def test_returns_payload_on_success(sleeps):
    client = InventoryClient(ScriptedTransport(OK))
    assert client.get_inventory(101, "A")["remaining"] == 40


def test_retries_timeouts_and_server_errors(sleeps):
    transport = ScriptedTransport(TimeoutError(), Response(500, {}), OK)
    client = InventoryClient(transport, max_retries=5)

    assert client.get_inventory(101, "A")["current_price"] == 95.0
    assert transport.calls == 3


def test_gives_up_after_max_retries(sleeps):
    transport = ScriptedTransport(*[Response(503, {})] * 3)
    client = InventoryClient(transport, max_retries=3)

    with pytest.raises(InventoryAPIError):
        client.get_inventory(101, "A")
    assert transport.calls == 3


def test_does_not_retry_client_errors(sleeps):
    transport = ScriptedTransport(*[Response(400, {"error": "bad request"})] * 5)
    client = InventoryClient(transport, max_retries=5)

    with pytest.raises(InventoryAPIError):
        client.get_inventory(101, "A")
    assert transport.calls == 1


def test_backs_off_between_retries(sleeps):
    transport = ScriptedTransport(Response(500, {}), Response(500, {}), OK)
    client = InventoryClient(transport, max_retries=5)

    client.get_inventory(101, "A")

    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0] > 0
