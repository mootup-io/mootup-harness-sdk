"""Anthropic passthrough route — forwards request body verbatim."""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger("convo.llm_proxy.providers.anthropic")

UPSTREAM_URL = "https://api.anthropic.com/v1/messages"
COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"


async def count_tokens(
    *, body: dict[str, Any], request: Request, token_class: str
) -> Response:
    """Forward an Anthropic ``count_tokens`` probe verbatim to Anthropic.

    Claude Code calls this before each turn; it is Anthropic-shaped in and out,
    so no translation is needed. Same auth as :func:`forward`.
    """
    if token_class == "anthropic_api_key":
        api_key = request.headers["Authorization"].removeprefix("Bearer ")
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY_UPSTREAM", "")
        if not api_key:
            raise HTTPException(
                status_code=503, detail="anthropic provider not configured"
            )
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(COUNT_TOKENS_URL, json=body, headers=headers)
    if resp.status_code == 200:
        return Response(
            content=resp.content, status_code=200, media_type="application/json"
        )
    # Fallback: estimate locally so the turn isn't blocked if the upstream
    # count_tokens call fails (the real error surfaces on /v1/messages).
    from .. import tokens

    return Response(
        content=tokens.estimate_count_tokens_body(body),
        status_code=200,
        media_type="application/json",
    )


async def forward(
    *, body: dict[str, Any], request: Request, token_class: str
) -> Response:
    """Pass body through to Anthropic Messages upstream.

    Auth: if token_class is anthropic_api_key, agent's own key is reused;
    if proxy_secret, ANTHROPIC_API_KEY_UPSTREAM env is substituted.
    """
    if token_class == "anthropic_api_key":
        api_key = request.headers["Authorization"].removeprefix("Bearer ")
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY_UPSTREAM", "")
        if not api_key:
            raise HTTPException(
                status_code=503, detail="anthropic provider not configured"
            )

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    stream = bool(body.get("stream"))
    if not stream:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(UPSTREAM_URL, json=body, headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )

    async def upstream_sse() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", UPSTREAM_URL, json=body, headers=headers
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return EventSourceResponse(upstream_sse(), media_type="text/event-stream")
