"""Shared fixtures for mootup-harness-sdk tests.

Provides a hand-crafted MCP client stub that records `call_tool` invocations
and returns configurable responses. No dependency on the `mcp` SDK package
(inv 8); the stub matches only the `MCPClientLike` Protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest


VALID_ORIENTATION: dict[str, Any] = {
    "identity": {
        "actor_id": "agt_test",
        "display_name": "Test Agent",
        "actor_type": "agent",
        "is_admin": False,
    },
    "focus_space": {
        "space_id": "spc_test",
        "description": "test",
        "status": "active",
    },
    "unread_mentions": 0,
    "last_status": None,
    "participants": [],
    "context": "summary body",
}


@dataclass
class _StubResponse:
    structured_content: dict[str, Any] | None = None
    content: list[Any] | None = None
    is_error: bool = False


class MCPClientStub:
    """Minimal MCP client stub — structurally matches `MCPClientLike`.

    Identity-hashable (default) so WeakKeyDictionary-based memoization works.
    """

    def __init__(
        self,
        response: _StubResponse | None = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self.response = response if response is not None else _StubResponse()
        self.raise_on_call = raise_on_call
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> _StubResponse:
        self.calls.append((name, arguments))
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.response


@pytest.fixture
def mcp_stub() -> MCPClientStub:
    return MCPClientStub(
        response=_StubResponse(structured_content=dict(VALID_ORIENTATION)),
    )


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MOOTUP_OAUTH_TOKEN", "MOOTUP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield
    # monkeypatch handles rollback
    _ = os.environ


@pytest.fixture
def stub_response() -> type[_StubResponse]:
    return _StubResponse
