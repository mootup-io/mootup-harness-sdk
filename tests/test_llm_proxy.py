"""R1-R8 tests for the multi-provider LLM proxy (Run D-1)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from mootup_harness_sdk.llm_proxy.server import app
from mootup_harness_sdk.llm_proxy.translators import openai_format

PROXY_SECRET = "test-shared-secret-d1-do-not-use-in-prod"


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROXY_SHARED_SECRET", PROXY_SECRET)
    monkeypatch.setenv("ANTHROPIC_API_KEY_UPSTREAM", "sk-ant-api-upstream-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oai-test-key")


@pytest.fixture
async def client() -> Any:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://proxy"
    ) as c:
        yield c


# ── R1 token-rejection ──
@pytest.mark.parametrize(
    "token,model,expected_status",
    [
        ("sk-ant-oat01-XXX", "claude-sonnet-4-6", 403),
        ("sk-unknown-prefix-XXX", "claude-sonnet-4-6", 403),
        ("", "claude-sonnet-4-6", 401),
    ],
)
async def test_r1_token_rejection(
    client: AsyncClient, token: str, model: str, expected_status: int
) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = await client.post(
        "/v1/messages",
        json={"model": model, "max_tokens": 32, "messages": []},
        headers=headers,
    )
    assert r.status_code == expected_status


# ── R2 token-acceptance ──
async def test_r2_proxy_secret_accepted(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        return httpx.Response(
            200,
            json={
                "id": "msg_x",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "model": "claude-sonnet-4-6",
                "role": "assistant",
                "type": "message",
                "stop_sequence": None,
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    r = await client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200


# ── R3 anthropic passthrough ──
async def test_r3_anthropic_passthrough(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        # URL-discriminate: only intercept upstream (absolute https) calls,
        # let the test's relative POST to the proxy app pass through.
        if str(url).startswith("http"):
            captured["url"] = str(url)
            captured["json"] = kw.get("json")
            captured["headers"] = kw.get("headers")
            return httpx.Response(
                200,
                json={
                    "id": "msg_a",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "text", "text": "verbatim"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            )
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    r = await client.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "test"}],
        },
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-api-upstream-test"


# ── R4 + R5 OpenAI-compat translation (parameterized over DeepSeek + Fireworks) ──
@pytest.mark.parametrize(
    "model,expected_url,expected_key",
    [
        (
            "deepseek-chat",
            "https://api.deepseek.com/v1/chat/completions",
            "ds-test-key",
        ),
        (
            "fireworks/llama-v3p1-70b-instruct",
            "https://api.fireworks.ai/inference/v1/chat/completions",
            "fw-test-key",
        ),
        (
            "gpt-4o",
            "https://api.openai.com/v1/chat/completions",
            "oai-test-key",
        ),
    ],
)
async def test_r4_r5_openai_compat_translation(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_url: str,
    expected_key: str,
) -> None:
    captured: dict[str, Any] = {}
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            captured["url"] = str(url)
            captured["json"] = kw.get("json")
            captured["headers"] = kw.get("headers")
            return httpx.Response(
                200,
                json={
                    "id": "cmpl_x",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "translated"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            )
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    r = await client.post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "test"}],
        },
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    assert captured["url"] == expected_url
    assert captured["headers"]["Authorization"] == f"Bearer {expected_key}"
    body = r.json()
    assert body["content"][0]["text"] == "translated"
    assert body["stop_reason"] == "end_turn"
    assert body["model"] == model


# ── R6 streaming SSE translation ──
async def test_r6_streaming_translation(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}, "finish_reason":"stop"}], "usage":{"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    class _MockStream:
        async def aiter_lines(self):  # type: ignore[no-untyped-def]
            for c in sse_chunks:
                yield c.decode().strip()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    def mock_stream(self, *a, **kw):  # type: ignore[no-untyped-def]
        return _MockStream()

    monkeypatch.setattr(httpx.AsyncClient, "stream", mock_stream)

    r = await client.post(
        "/v1/messages",
        json={
            "model": "deepseek-chat",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    body = r.text
    assert "message_start" in body
    assert "content_block_start" in body
    assert "content_block_delta" in body
    assert "message_stop" in body


# ── R7 tool-use round-trip ──
def test_r7_tool_use_round_trip() -> None:
    anth_req = {
        "model": "deepseek-chat",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "weather please"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Look up weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }
    oai_req = openai_format.anthropic_to_openai_request(anth_req)
    assert oai_req["tools"][0]["function"]["name"] == "get_weather"
    assert oai_req["tools"][0]["function"]["parameters"]["required"] == ["city"]

    oai_resp = {
        "id": "cmpl_y",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }
    anth_resp = openai_format.openai_to_anthropic_response(
        oai_resp, original_model="deepseek-chat"
    )
    tool_block = anth_resp["content"][0]
    assert tool_block["type"] == "tool_use"
    assert tool_block["name"] == "get_weather"
    assert tool_block["input"] == {"city": "SF"}
    assert anth_resp["stop_reason"] == "tool_use"


# ── R8 unsupported model ──
async def test_r8_unsupported_model(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/messages",
        json={"model": "gemini-1.5-pro", "max_tokens": 32, "messages": []},
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 400
    assert "unsupported_model" in r.text


# ── R9 count_tokens pass-through for providers that support it (claude, deepseek) ──
@pytest.mark.parametrize(
    "model,expected_url,auth_header,auth_value",
    [
        (
            "claude-sonnet-4-6",
            "https://api.anthropic.com/v1/messages/count_tokens",
            "x-api-key",
            "sk-ant-api-upstream-test",
        ),
        (
            "deepseek-v4-pro",
            "https://api.deepseek.com/anthropic/v1/messages/count_tokens",
            "Authorization",
            "Bearer ds-test-key",
        ),
    ],
)
async def test_r9_count_tokens_passthrough(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_url: str,
    auth_header: str,
    auth_value: str,
) -> None:
    captured: dict[str, Any] = {}
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            captured["url"] = str(url)
            captured["json"] = kw.get("json")
            captured["headers"] = kw.get("headers")
            return httpx.Response(200, json={"input_tokens": 42})
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    r = await client.post(
        # Claude Code appends ?beta=true; the route must still match.
        "/v1/messages/count_tokens?beta=true",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    assert r.json()["input_tokens"] == 42
    assert captured["url"] == expected_url
    # Body forwarded verbatim (Anthropic-shaped; no OpenAI translation).
    assert captured["json"]["model"] == model
    assert captured["json"]["messages"][0]["content"] == "hi"
    assert captured["headers"][auth_header] == auth_value


# ── R9b Fireworks has no upstream count_tokens → local estimate, no upstream call ──
async def test_r9b_count_tokens_fireworks_estimates(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mootup_harness_sdk.llm_proxy import tokens

    called = {"upstream": False}
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            called["upstream"] = True
            return httpx.Response(404, json={"error": "no route"})
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    body = {
        "model": "accounts/fireworks/models/glm-5p2",
        "messages": [{"role": "user", "content": "hello world, size me up"}],
    }
    r = await client.post(
        "/v1/messages/count_tokens",
        json=body,
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    assert r.json()["input_tokens"] == tokens.estimate_input_tokens(body)
    assert called["upstream"] is False  # no wasted upstream round-trip


# ── R9c DeepSeek falls back to a local estimate if upstream count_tokens fails ──
async def test_r9c_count_tokens_fallback_on_upstream_failure(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mootup_harness_sdk.llm_proxy import tokens

    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            return httpx.Response(404, json={"error": "no route"})
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    body = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "estimate me"}],
    }
    r = await client.post(
        "/v1/messages/count_tokens",
        json=body,
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    assert r.json()["input_tokens"] == tokens.estimate_input_tokens(body)


# ── R10 count_tokens shares the auth gate (rejects unknown tokens) ──
async def test_r10_count_tokens_rejects_unknown_token(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/messages/count_tokens",
        json={"model": "deepseek-v4-pro", "messages": []},
        headers={"Authorization": "Bearer sk-unknown-prefix-XXX"},
    )
    assert r.status_code == 403


# ── R11 reasoning round-trip (non-stream response + request echo) ──
def test_r11_reasoning_response_prepends_thinking() -> None:
    oai_resp = {
        "id": "cmpl_r",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "let me think...",
                    "content": "the answer is 4",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5},
    }
    anth = openai_format.openai_to_anthropic_response(
        oai_resp, original_model="deepseek-v4-pro"
    )
    # Thinking comes first, with a non-empty (synthesized) signature.
    assert anth["content"][0]["type"] == "thinking"
    assert anth["content"][0]["thinking"] == "let me think..."
    assert anth["content"][0]["signature"]
    assert anth["content"][1] == {"type": "text", "text": "the answer is 4"}


def test_r11_reasoning_request_echoes_reasoning_content() -> None:
    # Turn 2: replay the assistant's thinking + tool_use. The thinking MUST come
    # back to DeepSeek as reasoning_content on the assistant message or it 400s.
    anth_req = {
        "model": "deepseek-v4-pro",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "call the tool", "signature": "x"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "SF"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "sunny",
                    }
                ],
            },
        ],
    }
    oai = openai_format.anthropic_to_openai_request(anth_req)
    asst = next(
        m
        for m in oai["messages"]
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    assert asst["reasoning_content"] == "call the tool"
    assert asst["tool_calls"][0]["function"]["name"] == "get_weather"


# ── R12 streaming reasoning → thinking block events ──
async def test_r12_streaming_reasoning_block(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sse_chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking "}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"hard"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"done"}, "finish_reason":"stop"}], "usage":{"completion_tokens":2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    class _MockStream:
        async def aiter_lines(self):  # type: ignore[no-untyped-def]
            for c in sse_chunks:
                yield c.decode().strip()

        async def __aenter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __aexit__(self, *args):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr(
        httpx.AsyncClient, "stream", lambda self, *a, **kw: _MockStream()
    )
    r = await client.post(
        "/v1/messages",
        json={
            "model": "deepseek-v4-pro",
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    body = r.text
    # A thinking block opens, streams a thinking_delta, then closes with a
    # signature_delta before the text block.
    assert "thinking" in body
    assert "thinking_delta" in body
    assert "signature_delta" in body
    assert "text_delta" in body
    assert body.index("thinking_delta") < body.index("text_delta")


# ── R13 debug capture records both sides of the translation ──
async def test_r13_capture_records_both_sides(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("LLM_PROXY_CAPTURE_DIR", str(tmp_path))
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            return httpx.Response(
                200,
                json={
                    "id": "cmpl",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "hi there"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                },
            )
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    r = await client.post(
        "/v1/messages",
        json={
            "model": "deepseek-v4-pro",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200

    lines = (tmp_path / "llm-proxy-capture.ndjson").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["provider"] == "deepseek" and rec["stream"] is False
    assert rec["upstream_status"] == 200
    # Both sides of the request translation are present.
    assert rec["anthropic_request"]["messages"][0]["content"] == "hello"
    assert (
        rec["upstream_request"]["model"] == "deepseek-v4-pro"
    )  # translated OpenAI body
    # Translated Anthropic response too.
    assert rec["anthropic_response"]["content"][0]["text"] == "hi there"


def test_r13_capture_disabled_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("LLM_PROXY_CAPTURE_DIR", raising=False)
    from mootup_harness_sdk.llm_proxy import capture

    assert capture.enabled() is False
    capture.record(provider="x", anthropic_request={"k": "v"})  # no-op, must not raise
    assert not (tmp_path / "llm-proxy-capture.ndjson").exists()


# ── OpenAI provider: routing + request shaping (chat vs reasoning models) ──
def test_openai_routing() -> None:
    from mootup_harness_sdk.llm_proxy.router import select_provider

    assert select_provider("gpt-4o") == "openai"
    assert select_provider("gpt-5-chat-latest") == "openai"
    assert select_provider("o1") == "openai"
    assert select_provider("o3-mini") == "openai"
    assert select_provider("o4-mini") == "openai"
    assert select_provider("claude-sonnet-4-6") == "anthropic"
    assert select_provider("deepseek-v4-pro") == "deepseek"


async def _capture_openai_upstream(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]
) -> dict[str, Any]:
    """POST `body` to the proxy, mock the OpenAI upstream, return the captured
    upstream (OpenAI-format) request body."""
    captured: dict[str, Any] = {}
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            captured["json"] = kw.get("json")
            return httpx.Response(
                200,
                json={
                    "id": "cmpl_x",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            )
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    r = await client.post(
        "/v1/messages", json=body, headers={"Authorization": f"Bearer {PROXY_SECRET}"}
    )
    assert r.status_code == 200
    return captured["json"]


async def test_openai_chat_model_uses_max_completion_tokens(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = await _capture_openai_upstream(
        client,
        monkeypatch,
        {
            "model": "gpt-4o",
            "max_tokens": 32,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    # max_tokens is renamed to max_completion_tokens; chat models keep temperature.
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 32
    assert sent["temperature"] == 0.7
    assert "reasoning_effort" not in sent


async def test_openai_reasoning_model_drops_temperature(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = await _capture_openai_upstream(
        client,
        monkeypatch,
        {
            "model": "o3-mini",
            "max_tokens": 64,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 64
    # Reasoning models reject a non-default temperature → dropped.
    assert "temperature" not in sent


async def test_openai_reasoning_effort_from_thinking_budget(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = await _capture_openai_upstream(
        client,
        monkeypatch,
        {
            "model": "gpt-5",
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 20000},
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert sent["reasoning_effort"] == "high"


async def test_openai_count_tokens_estimates_without_upstream(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mootup_harness_sdk.llm_proxy import tokens

    called = {"upstream": False}
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            called["upstream"] = True
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "estimate me"}]}
    r = await client.post(
        "/v1/messages/count_tokens",
        json=body,
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    assert called["upstream"] is False  # OpenAI has no count_tokens endpoint
    assert r.json()["input_tokens"] == tokens.estimate_input_tokens(body)


# ── Codex/OpenAI-compatible proxy surface ──
async def test_codex_chat_completions_routes_openai_compatible_body(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    _original_post = httpx.AsyncClient.post

    async def mock_post(self, url, **kw):  # type: ignore[no-untyped-def]
        if str(url).startswith("http"):
            captured["url"] = str(url)
            captured["json"] = kw.get("json")
            captured["headers"] = kw.get("headers")
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl_codex",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
                headers={"content-type": "application/json"},
            )
        return await _original_post(self, url, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    body = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    r = await client.post(
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 200
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["json"] == body
    assert captured["headers"]["Authorization"] == "Bearer ds-test-key"
    assert r.json()["choices"][0]["message"]["content"] == "ok"


async def test_codex_chat_completions_rejects_anthropic_model(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "claude-sonnet-4-6", "messages": []},
        headers={"Authorization": f"Bearer {PROXY_SECRET}"},
    )
    assert r.status_code == 400
    assert "unsupported_openai_compat_model" in r.text


async def test_codex_chat_completions_requires_proxy_secret(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer sk-ant-api-user"},
    )
    assert r.status_code == 403
