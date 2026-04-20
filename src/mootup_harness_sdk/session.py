"""Session types + error classes for mootup-harness-sdk.

Duck-typed MCPClientLike Protocol avoids a runtime dependency on the `mcp`
package (inv 8) — callers bring their own MCP client instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MCPClientLike(Protocol):
    """Minimal MCP client surface the SDK uses.

    Implementations may be mcp.client.Client, a custom wrapper, or a test
    stub — the SDK only calls `.call_tool(name, arguments)`.
    """

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:  # pragma: no cover — structural type
        ...


class MootupNotOrientedError(Exception):
    """Raised when session state is requested before setup resolves."""

    def __init__(
        self,
        message: str = "Session not yet resolved — await setup_mootup() first",
    ) -> None:
        super().__init__(message)


@dataclass(frozen=True)
class Session:
    """Typed session returned by `setup_mootup()` / `MootupAgent.setup()`.

    Frozen (T-4) — mutation raises `dataclasses.FrozenInstanceError`.
    """

    participant_id: str
    space_id: str | None
    orientation_summary: str
