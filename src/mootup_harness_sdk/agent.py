"""MootupAgent + setup_mootup — thin harness SDK entrypoints.

Calls the shipped convo `orientation` MCP tool via a caller-supplied MCP
client; extracts structured content into a frozen `Session`; memoizes per
MCP client instance (inv 10).
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

from .session import MCPClientLike, MootupNotOrientedError, Session

_JWT_SHAPE_RE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$")
_REDACTION_SUBSTRINGS = ("Bearer ", "Authorization", "api_key", "token")

# Per-process per-client session cache (inv 10). WeakKeyDictionary so the
# cache does not pin client objects in memory.
_session_cache: "WeakKeyDictionary[object, Session]" = WeakKeyDictionary()


def _redact_error(message: str) -> str:
    redacted = message
    for needle in _REDACTION_SUBSTRINGS:
        if needle.lower() in redacted.lower():
            pattern = re.compile(re.escape(needle), re.IGNORECASE)
            redacted = pattern.sub("<redacted>", redacted)
    return redacted


def _resolve_auth(explicit: str | None) -> str:
    if explicit:
        return explicit
    for var in ("MOOTUP_OAUTH_TOKEN", "MOOTUP_API_KEY"):
        value = os.environ.get(var)
        if value:
            return value
    raise MootupNotOrientedError(
        "No auth supplied (kwarg > MOOTUP_OAUTH_TOKEN > MOOTUP_API_KEY > error)",
    )


def _validate_base_url_origin(base_url: str, token: str) -> None:
    """T-1: when token is JWT-shaped, compare its `iss` claim origin to base_url."""
    if not _JWT_SHAPE_RE.match(token):
        return
    try:
        claims_b64 = token.split(".")[1]
        pad_len = (-len(claims_b64)) % 4
        claims_bytes = base64.urlsafe_b64decode(claims_b64 + ("=" * pad_len))
        claims = json.loads(claims_bytes)
    except (ValueError, json.JSONDecodeError):
        return
    iss = claims.get("iss") if isinstance(claims, dict) else None
    if not iss:
        return
    try:
        iss_parsed = urlparse(iss)
        base_parsed = urlparse(base_url)
    except ValueError as exc:
        raise ValueError("base_url or OAuth issuer URL is malformed") from exc
    iss_origin = f"{iss_parsed.scheme}://{iss_parsed.netloc}"
    base_origin = f"{base_parsed.scheme}://{base_parsed.netloc}"
    if iss_origin != base_origin:
        raise ValueError(
            f"base_url origin ({base_origin}) does not match OAuth issuer "
            f"origin ({iss_origin})",
        )


def _extract_structured(response: Any) -> dict[str, Any]:
    # MCP SDK responses may expose either attribute or mapping-style access;
    # accept both shapes to avoid a hard MCP SDK dep (inv 8).
    is_error = getattr(response, "is_error", None) or getattr(
        response, "isError", None
    )
    if is_error:
        raise MootupNotOrientedError("orientation tool returned is_error=True")
    structured = getattr(response, "structured_content", None) or getattr(
        response, "structuredContent", None
    )
    if structured is None and isinstance(response, dict):
        structured = response.get("structured_content") or response.get(
            "structuredContent"
        )
    if isinstance(structured, dict):
        return structured
    # Fallback: parse JSON text block if structured_content absent.
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content")
    if content:
        for block in content:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                try:
                    parsed = json.loads(text)
                except (ValueError, json.JSONDecodeError):
                    continue
                if isinstance(parsed, dict):
                    return parsed
    raise MootupNotOrientedError(
        "orientation response missing structured content (requires convo ≥AH-g)",
    )


async def setup_mootup(
    mcp_client: MCPClientLike,
    *,
    base_url: str,
    auth: str | None = None,
) -> Session:
    """Invoke convo `orientation` + return a frozen typed Session.

    Memoized per MCP client — second call on the same client returns the
    cached Session without re-invoking `orientation` (inv 10).
    """
    cached = _session_cache.get(mcp_client)
    if cached is not None:
        return cached

    token = _resolve_auth(auth)
    try:
        _validate_base_url_origin(base_url, token)
    except ValueError as exc:
        raise ValueError(_redact_error(str(exc))) from None

    try:
        response = await mcp_client.call_tool("orientation", {})
    except MootupNotOrientedError:
        raise
    except Exception as exc:
        raise RuntimeError(
            _redact_error(f"orientation call failed: {exc}"),
        ) from None

    structured = _extract_structured(response)
    identity = structured.get("identity")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("actor_id"), str
    ):
        raise MootupNotOrientedError(
            "orientation response missing identity.actor_id",
        )
    focus = structured.get("focus_space")
    space_id = (
        focus.get("space_id")
        if isinstance(focus, dict) and isinstance(focus.get("space_id"), str)
        else None
    )
    context_raw = structured.get("context")
    orientation_summary = context_raw if isinstance(context_raw, str) else ""

    session = Session(
        participant_id=identity["actor_id"],
        space_id=space_id,
        orientation_summary=orientation_summary,
    )
    _session_cache[mcp_client] = session
    return session


class MootupAgent:
    """Thin wrapper around `setup_mootup()` for harnesses that prefer OOP.

    Instantiate, `await agent.setup()`, then access `agent.session`. Before
    setup resolves, `agent.session` raises `MootupNotOrientedError`.
    """

    def __init__(
        self,
        mcp_client: MCPClientLike,
        *,
        base_url: str,
        auth: str | None = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.base_url = base_url
        self.auth = auth
        self._session: Session | None = None

    async def setup(self) -> Session:
        self._session = await setup_mootup(
            self.mcp_client,
            base_url=self.base_url,
            auth=self.auth,
        )
        return self._session

    @property
    def session(self) -> Session:
        if self._session is None:
            raise MootupNotOrientedError()
        return self._session
