"""DeepSeek route — OpenAI-compatible API at api.deepseek.com."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from fastapi import HTTPException, Request, Response
from sse_starlette.sse import EventSourceResponse

from .. import streaming, tokens
from ..translators import openai_format

logger = logging.getLogger("convo.llm_proxy.providers.deepseek")

UPSTREAM_URL = "https://api.deepseek.com/v1/chat/completions"
COUNT_TOKENS_URL = "https://api.deepseek.com/anthropic/v1/messages/count_tokens"


async def count_tokens(
    *, body: dict[str, Any], request: Request, token_class: str
) -> Response:
    """Forward an Anthropic ``count_tokens`` probe to DeepSeek's
    Anthropic-compatible endpoint. count_tokens is Anthropic-shaped both ways,
    so the body passes through verbatim (no OpenAI translation)."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="deepseek provider not configured")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(COUNT_TOKENS_URL, json=body, headers=headers)
    if resp.status_code == 200:
        return Response(
            content=resp.content, status_code=200, media_type="application/json"
        )
    # Upstream lacks/refused count_tokens — estimate locally so the turn isn't
    # blocked (the real error, if any, surfaces on the actual /v1/messages call).
    return Response(
        content=tokens.estimate_count_tokens_body(body),
        status_code=200,
        media_type="application/json",
    )


async def forward(
    *, body: dict[str, Any], request: Request, token_class: str
) -> Response:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="deepseek provider not configured")

    openai_body = openai_format.anthropic_to_openai_request(body)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    stream = bool(body.get("stream"))
    if not stream:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(UPSTREAM_URL, json=openai_body, headers=headers)
        if resp.status_code >= 400:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type="application/json",
            )
        anthropic_body = openai_format.openai_to_anthropic_response(
            resp.json(), original_model=body["model"]
        )
        return Response(
            content=json.dumps(anthropic_body).encode(),
            status_code=200,
            media_type="application/json",
        )

    return EventSourceResponse(
        streaming.openai_chunks_to_anthropic_sse(
            UPSTREAM_URL, openai_body, headers, body["model"]
        ),
        media_type="text/event-stream",
    )
