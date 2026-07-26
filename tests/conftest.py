import socket

import pytest


@pytest.fixture(autouse=True)
def block_network_and_remove_api_key(monkeypatch):
    """Make every test fail immediately if it attempts network access."""

    def denied(*args, **kwargs):
        raise AssertionError("network access is forbidden in the test suite")

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
