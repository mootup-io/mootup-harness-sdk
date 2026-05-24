"""FastAPI app entry — POST /v1/messages + GET /healthz routes."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from . import auth, router

logger = logging.getLogger("convo.llm_proxy")

app = FastAPI(title="Convo LLM Proxy", version="0.1.0")


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
        },
    )


@app.post("/v1/messages")
async def v1_messages(request: Request) -> Response:
    body: dict[str, Any] = await request.json()
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

    handler = router.dispatch(provider)
    return await handler(body=body, request=request, token_class=token_class)
