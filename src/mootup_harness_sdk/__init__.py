"""mootup-harness-sdk — thin Python SDK for agent harness integrations.

Call `setup_mootup(mcp_client, base_url=..., auth=...)` (or instantiate
`MootupAgent`) to invoke the convo `orientation` MCP tool on connect and
receive a frozen, typed `Session`.
"""

from __future__ import annotations

from .agent import MootupAgent, setup_mootup
from .session import MCPClientLike, MootupNotOrientedError, Session

__all__ = [
    "MCPClientLike",
    "MootupAgent",
    "MootupNotOrientedError",
    "Session",
    "setup_mootup",
]
