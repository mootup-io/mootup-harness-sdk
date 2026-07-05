"""FastAPI app entry — POST /v1/messages + GET /healthz routes."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from pydantic import BaseModel

from . import auth, router

logger = logging.getLogger("convo.llm_proxy")

app = FastAPI(title="Convo LLM Proxy", version="0.1.0")


OPENAI_COMPAT_UPSTREAMS = {
    "deepseek": ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY"),
    "fireworks": (
        "https://api.fireworks.ai/inference/v1/chat/completions",
        "FIREWORKS_API_KEY",
    ),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
}


class HealthzResponse(BaseModel):
    status: str
    providers: dict[str, bool]


@app.get("/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    return HealthzResponse(
        status="ok",
        providers={
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY_UPSTREAM")),
            "deepseek": bool(os.getenv("DEEPSEEK_API_KEY")),
            "fireworks": bool(os.getenv("FIREWORKS_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
        },
    )


def _authorize_and_route(
    request: Request, body: dict[str, Any]
) -> tuple[router.Provider, str]:
    """Shared gate for the message routes: validate the model + bearer token,
    reject subscription/unknown tokens, and resolve the provider. Returns
    ``(provider, token_class)`` or raises the matching ``HTTPException``."""
    model = body.get("model")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'model' field")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    token = auth_header.removeprefix("Bearer ")
    token_class = auth.classify_bearer(token, model)
    if token_class == "rejected":
        auth.log_rejection(token, model)
        raise HTTPException(
            status_code=403, detail="subscription_or_unknown_token_rejected"
        )

    provider = router.select_provider(model)
    if provider == "unsupported":
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_model",
                "allowed_prefixes": router.ALLOWED_PREFIXES,
            },
        )
    return provider, token_class


@app.post("/v1/messages")
async def v1_messages(request: Request) -> Response:
    body: dict[str, Any] = await request.json()
    provider, token_class = _authorize_and_route(request, body)
    handler = router.dispatch(provider)
    return await handler(body=body, request=request, token_class=token_class)


@app.post("/v1/messages/count_tokens")
async def v1_messages_count_tokens(request: Request) -> Response:
    """Claude Code probes token usage before each turn. Without this route the
    proxy 404s and the turn returns empty. Same auth + prefix routing as
    ``/v1/messages``; forwards to each provider's Anthropic count_tokens
    endpoint (Anthropic-shaped both ways, so no translation)."""
    body: dict[str, Any] = await request.json()
    provider, token_class = _authorize_and_route(request, body)
    handler = router.dispatch_count_tokens(provider)
    return await handler(body=body, request=request, token_class=token_class)


def _authorize_openai_compat(request: Request, body: dict[str, Any]) -> router.Provider:
    """Validate an OpenAI-compatible request and resolve its upstream provider.

    This route is for Codex/custom OpenAI-compatible clients. It accepts only
    the proxy shared secret, because Anthropic API keys are valid only for the
    Anthropic-shaped Claude Code route.
    """
    model = body.get("model")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'model' field")

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    token = auth_header.removeprefix("Bearer ")
    if auth.classify_bearer(token, model) != "proxy_secret":
        auth.log_rejection(token, model)
        raise HTTPException(
            status_code=403, detail="subscription_or_unknown_token_rejected"
        )

    provider = router.select_provider(model)
    if provider not in OPENAI_COMPAT_UPSTREAMS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_openai_compat_model",
                "allowed_prefixes": [
                    p
                    for p in router.ALLOWED_PREFIXES
                    if p != "claude-"
                ],
            },
        )
    return provider


async def _forward_openai_compat(body: dict[str, Any], provider: router.Provider) -> Response:
    upstream = OPENAI_COMPAT_UPSTREAMS[provider]
    upstream_url, key_env = upstream
    api_key = os.getenv(key_env, "")
    if not api_key:
        raise HTTPException(status_code=503, detail=f"{provider} provider not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    stream = bool(body.get("stream"))
    if not stream:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(upstream_url, json=body, headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    client = httpx.AsyncClient(timeout=None)
    req = client.build_request("POST", upstream_url, json=body, headers=headers)
    resp = await client.send(req, stream=True)
    if resp.status_code >= 400:
        content = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return Response(
            content=content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    async def cleanup() -> None:
        await resp.aclose()
        await client.aclose()

    return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
        background=BackgroundTask(cleanup),
    )


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request) -> Response:
    body: dict[str, Any] = await request.json()
    provider = _authorize_openai_compat(request, body)
    return await _forward_openai_compat(body, provider)
