"""OpenAI route — Chat Completions API at api.openai.com.

OpenAI's Chat Completions schema is the proxy's translation target, so this
mirrors the Fireworks route. The OpenAI-specific wrinkle is the reasoning
models (o-series, gpt-5): they require ``max_completion_tokens`` instead of the
deprecated ``max_tokens``, reject a non-default ``temperature``/``top_p``, and
hide their reasoning text (no ``reasoning_content`` to round-trip, unlike
DeepSeek). ``_adapt_for_openai`` applies those request fixups post-translation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from fastapi import HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse

from .. import capture, streaming, tokens
from ..translators import openai_format

logger = logging.getLogger("convo.llm_proxy.providers.openai")

UPSTREAM_URL = "https://api.openai.com/v1/chat/completions"

# o-series models are reasoning models; so is the gpt-5 family except its
# `*-chat*` variants (e.g. gpt-5-chat-latest, a standard chat model).
_REASONING_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    if m.startswith(_REASONING_PREFIXES):
        return True
    return m.startswith("gpt-5") and "chat" not in m


def _reasoning_effort(anthropic_body: dict[str, Any]) -> str | None:
    """Map an Anthropic extended-thinking budget to OpenAI ``reasoning_effort``."""
    thinking = anthropic_body.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return None
    budget = thinking.get("budget_tokens")
    if not isinstance(budget, int):
        return None
    if budget <= 4096:
        return "low"
    if budget <= 16384:
        return "medium"
    return "high"


def _adapt_for_openai(
    openai_body: dict[str, Any], anthropic_body: dict[str, Any], model: str
) -> dict[str, Any]:
    """Apply OpenAI-specific request fixups (mutates + returns ``openai_body``).

    - ``max_tokens`` is deprecated in favor of ``max_completion_tokens``, which
      reasoning models *require* and chat models also accept — so use it
      universally.
    - Reasoning models reject a non-default ``temperature``/``top_p``; drop them.
    - Map an extended-thinking budget to ``reasoning_effort``.
    """
    max_tokens = openai_body.pop("max_tokens", None)
    if max_tokens is not None:
        openai_body["max_completion_tokens"] = max_tokens
    if _is_reasoning_model(model):
        openai_body.pop("temperature", None)
        openai_body.pop("top_p", None)
        effort = _reasoning_effort(anthropic_body)
        if effort is not None:
            openai_body["reasoning_effort"] = effort
    return openai_body


async def count_tokens(
    *, body: dict[str, Any], request: Request, token_class: str
) -> Response:
    """Estimate input tokens locally. OpenAI's Chat Completions API has no
    Anthropic-compatible ``count_tokens`` route, and Claude Code uses it only to
    size context before a turn, so a local estimate keeps the turn unblocked."""
    return Response(
        content=tokens.estimate_count_tokens_body(body),
        status_code=200,
        media_type="application/json",
    )


async def forward(
    *, body: dict[str, Any], request: Request, token_class: str
) -> Response:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="openai provider not configured")

    openai_body = _adapt_for_openai(
        openai_format.anthropic_to_openai_request(body), body, body["model"]
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    stream = bool(body.get("stream"))
    if not stream:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(UPSTREAM_URL, json=openai_body, headers=headers)
        ok = resp.status_code < 400
        anthropic_body = (
            openai_format.openai_to_anthropic_response(
                resp.json(), original_model=body["model"]
            )
            if ok
            else None
        )
        capture.record(
            provider="openai",
            model=body.get("model"),
            stream=False,
            upstream_url=UPSTREAM_URL,
            upstream_status=resp.status_code,
            anthropic_request=body,
            upstream_request=openai_body,
            upstream_response=resp.text,
            anthropic_response=anthropic_body,
        )
        if not ok:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        return Response(
            content=json.dumps(anthropic_body).encode(),
            status_code=200,
            media_type="application/json",
        )

    return EventSourceResponse(
        streaming.openai_chunks_to_anthropic_sse(
            UPSTREAM_URL,
            openai_body,
            headers,
            body["model"],
            capture_meta={
                "provider": "openai",
                "model": body.get("model"),
                "upstream_url": UPSTREAM_URL,
                "anthropic_request": body,
                "upstream_request": openai_body,
            },
        ),
        media_type="text/event-stream",
    )
