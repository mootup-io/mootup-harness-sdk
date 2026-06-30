"""Model-prefix classifier per D-1-ROUTING-BY-MODEL-FIELD."""

from __future__ import annotations

from typing import Awaitable, Callable, Literal

from fastapi import Response

Provider = Literal["anthropic", "deepseek", "fireworks", "openai", "unsupported"]

# OpenAI native model IDs are forwarded verbatim: gpt-* (gpt-4o, gpt-4.1,
# gpt-5, gpt-5-chat-*) and the o-series reasoning models (o1/o3/o4*).
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")

ALLOWED_PREFIXES: list[str] = [
    "claude-",
    "deepseek-",
    "fireworks/",
    "accounts/fireworks/",
    *_OPENAI_PREFIXES,
]


def select_provider(model: str) -> Provider:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("deepseek-"):
        return "deepseek"
    if model.startswith("fireworks/") or model.startswith("accounts/fireworks/"):
        return "fireworks"
    if model.startswith(_OPENAI_PREFIXES):
        return "openai"
    return "unsupported"


Handler = Callable[..., Awaitable[Response]]


def dispatch(provider: Provider) -> Handler:
    from .providers import anthropic, deepseek, fireworks, openai

    if provider == "anthropic":
        return anthropic.forward
    if provider == "deepseek":
        return deepseek.forward
    if provider == "fireworks":
        return fireworks.forward
    if provider == "openai":
        return openai.forward
    raise ValueError(f"No handler for provider {provider!r}")


def dispatch_count_tokens(provider: Provider) -> Handler:
    from .providers import anthropic, deepseek, fireworks, openai

    if provider == "anthropic":
        return anthropic.count_tokens
    if provider == "deepseek":
        return deepseek.count_tokens
    if provider == "fireworks":
        return fireworks.count_tokens
    if provider == "openai":
        return openai.count_tokens
    raise ValueError(f"No count_tokens handler for provider {provider!r}")
