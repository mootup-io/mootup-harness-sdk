"""AH-g Required tests R9–R14: setup_mootup + MootupAgent behavior."""

from __future__ import annotations

import base64
import dataclasses
import json
from typing import Any

import pytest

from mootup_harness_sdk import (
    MootupAgent,
    MootupNotOrientedError,
    Session,
    setup_mootup,
)

from .conftest import VALID_ORIENTATION, MCPClientStub


def _jwt_with_issuer(iss: str) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').rstrip(b"=")
    claims = base64.urlsafe_b64encode(json.dumps({"iss": iss}).encode()).rstrip(b"=")
    return f"{header.decode()}.{claims.decode()}.sig"


@pytest.mark.asyncio
async def test_r9_setup_mootup_happy_path(mcp_stub: MCPClientStub) -> None:
    """R9 — session fields extracted correctly from structured content."""
    session = await setup_mootup(
        mcp_stub,
        base_url="http://convo.test",
        auth="mootup_pat_abc",
    )
    assert isinstance(session, Session)
    assert session.participant_id == "agt_test"
    assert session.space_id == "spc_test"
    assert session.orientation_summary == "summary body"
    assert len(mcp_stub.calls) == 1
    assert mcp_stub.calls[0][0] == "orientation"


@pytest.mark.parametrize(
    ("explicit", "env_oauth", "env_api", "expect_ok"),
    [
        ("explicit_token", "env_oauth", "env_api", True),  # explicit wins
        (None, "env_oauth", "env_api", True),              # oauth env wins over api
        (None, None, "env_api", True),                      # api-key fallback
        (None, None, None, False),                           # no auth → error
    ],
    ids=["explicit-wins", "oauth-env", "api-key-env", "no-auth"],
)
@pytest.mark.asyncio
async def test_r10_auth_precedence(
    monkeypatch: pytest.MonkeyPatch,
    mcp_stub: MCPClientStub,
    explicit: str | None,
    env_oauth: str | None,
    env_api: str | None,
    expect_ok: bool,
) -> None:
    """R10 parametrized — OQ-G5 precedence: kwarg > OAUTH env > API_KEY env > error."""
    if env_oauth:
        monkeypatch.setenv("MOOTUP_OAUTH_TOKEN", env_oauth)
    if env_api:
        monkeypatch.setenv("MOOTUP_API_KEY", env_api)
    if expect_ok:
        session = await setup_mootup(
            mcp_stub,
            base_url="http://convo.test",
            auth=explicit,
        )
        assert session.participant_id == "agt_test"
    else:
        with pytest.raises(MootupNotOrientedError):
            await setup_mootup(mcp_stub, base_url="http://convo.test", auth=None)


@pytest.mark.asyncio
async def test_r11_missing_structured_content_raises(
    mcp_stub: MCPClientStub,
) -> None:
    """R11 — pre-AH-g convo (text-only response) triggers MootupNotOrientedError."""
    mcp_stub.response.structured_content = None
    mcp_stub.response.content = [{"type": "text", "text": "**Identity:** ..."}]
    with pytest.raises(MootupNotOrientedError, match="structured content"):
        await setup_mootup(
            mcp_stub,
            base_url="http://convo.test",
            auth="mootup_pat_abc",
        )


@pytest.mark.asyncio
async def test_r11_fallback_when_text_block_is_json(
    mcp_stub: MCPClientStub,
) -> None:
    """R11 tangent — if text block is valid JSON, extract fields from it."""
    mcp_stub.response.structured_content = None
    mcp_stub.response.content = [
        {"type": "text", "text": json.dumps(VALID_ORIENTATION)}
    ]
    session = await setup_mootup(
        mcp_stub,
        base_url="http://convo.test",
        auth="mootup_pat_abc",
    )
    assert session.participant_id == "agt_test"


@pytest.mark.asyncio
async def test_r12_session_memoization_per_client(
    mcp_stub: MCPClientStub,
) -> None:
    """R12 — 2nd call on same client returns cached session; call_tool fires once."""
    first = await setup_mootup(
        mcp_stub,
        base_url="http://convo.test",
        auth="mootup_pat_abc",
    )
    second = await setup_mootup(
        mcp_stub,
        base_url="http://convo.test",
        auth="mootup_pat_abc",
    )
    assert second is first
    assert len(mcp_stub.calls) == 1


@pytest.mark.asyncio
async def test_r13_session_dataclass_frozen(mcp_stub: MCPClientStub) -> None:
    """R13 — Session is frozen; mutation raises FrozenInstanceError."""
    session = await setup_mootup(
        mcp_stub,
        base_url="http://convo.test",
        auth="mootup_pat_abc",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        session.participant_id = "spoofed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_r14_mootup_agent_delegates_to_setup_mootup(
    mcp_stub: MCPClientStub,
) -> None:
    """R14 — MootupAgent.setup() matches setup_mootup() output; session raises pre-setup."""
    agent = MootupAgent(
        mcp_stub,
        base_url="http://convo.test",
        auth="mootup_pat_abc",
    )
    # Pre-setup access raises MootupNotOrientedError.
    with pytest.raises(MootupNotOrientedError):
        _ = agent.session
    resolved = await agent.setup()
    assert resolved.participant_id == "agt_test"
    assert agent.session is resolved


@pytest.mark.asyncio
async def test_r8_python_analog_origin_validation_rejects_jwt_issuer_mismatch(
    mcp_stub: MCPClientStub,
) -> None:
    """Python analog of R8 — OAuth issuer-origin mismatch rejected."""
    jwt_attacker = _jwt_with_issuer(
        "https://attacker.test/oauth/authorization-server"
    )
    with pytest.raises(ValueError, match="base_url origin"):
        await setup_mootup(
            mcp_stub,
            base_url="http://convo.test",
            auth=jwt_attacker,
        )


@pytest.mark.asyncio
async def test_non_jwt_token_skips_origin_check(mcp_stub: MCPClientStub) -> None:
    """Non-JWT tokens (PATs, API keys) skip the origin-check step."""
    session = await setup_mootup(
        mcp_stub,
        base_url="http://convo.test",
        auth="mootup_pat_xyz",
    )
    assert session.participant_id == "agt_test"


@pytest.mark.asyncio
async def test_is_error_response_raises(mcp_stub: MCPClientStub) -> None:
    """`is_error=True` on the response propagates as MootupNotOrientedError."""
    mcp_stub.response.is_error = True
    with pytest.raises(MootupNotOrientedError, match="is_error"):
        await setup_mootup(
            mcp_stub,
            base_url="http://convo.test",
            auth="mootup_pat_abc",
        )


@pytest.mark.asyncio
async def test_redacts_auth_header_from_propagated_errors() -> None:
    """Inv 11 — errors from call_tool that embed auth substrings are redacted."""
    leaky = RuntimeError("Bearer mootup_pat_secret leaked; api_key=xyz")
    stub = MCPClientStub(raise_on_call=leaky)
    err: Exception | None = None
    try:
        await setup_mootup(
            stub,
            base_url="http://convo.test",
            auth="mootup_pat_secret",
        )
    except Exception as exc:
        err = exc
    assert err is not None
    text = str(err)
    assert "Bearer " not in text
    assert "api_key" not in text.lower()
