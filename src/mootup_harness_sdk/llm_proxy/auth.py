"""Bearer extraction + classification per D-1-PROXY-AUTH-MODEL + D-MBT-PROXY-DEFAULT-DENY-SUBSCRIPTION-TOKENS."""

from __future__ import annotations

import logging
import os
from typing import Literal

logger = logging.getLogger("convo.llm_proxy.auth")

TokenClass = Literal["proxy_secret", "anthropic_api_key", "rejected"]

ANTHROPIC_API_KEY_PREFIX = "sk-ant-api"
ANTHROPIC_SUBSCRIPTION_PREFIX = "sk-ant-oat"


def classify_bearer(token: str, model: str) -> TokenClass:
    """Return the auth class for an inbound bearer token.

    proxy_secret: exact match against LLM_PROXY_SHARED_SECRET. Allowed any route.
    anthropic_api_key: sk-ant-api* prefix; allowed iff model is claude-*.
    rejected: subscription tokens (sk-ant-oat*) or unknown shapes.
    """
    shared_secret = os.getenv("LLM_PROXY_SHARED_SECRET", "")
    if shared_secret and token == shared_secret:
        return "proxy_secret"
    if token.startswith(ANTHROPIC_API_KEY_PREFIX) and model.startswith("claude-"):
        return "anthropic_api_key"
    return "rejected"


def log_rejection(token: str, model: str) -> None:
    """Log rejection with prefix-only (NOT full token) per D-MBT guardrail."""
    prefix = token[:12] if len(token) >= 12 else token
    logger.warning(
        "Rejected bearer token (model=%s, prefix=%r)",
        model,
        prefix,
    )
