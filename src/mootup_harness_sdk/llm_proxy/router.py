"""Model-prefix classifier per D-1-ROUTING-BY-MODEL-FIELD."""

from __future__ import annotations

from typing import Awaitable, Callable, Literal

from fastapi import Response

Provider = Literal["anthropic", "deepseek", "fireworks", "unsupported"]

ALLOWED_PREFIXES: list[str] = [
    "claude-",
    "deepseek-",
    "fireworks/",
    "accounts/fireworks/",
]


def select_provider(model: str) -> Provider:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("deepseek-"):
        return "deepseek"
    if model.startswith("fireworks/") or model.startswith("accounts/fireworks/"):
        return "fireworks"
    return "unsupported"


Handler = Callable[..., Awaitable[Response]]


def dispatch(provider: Provider) -> Handler:
    from .providers import anthropic, deepseek, fireworks

    if provider == "anthropic":
        return anthropic.forward
    if provider == "deepseek":
        return deepseek.forward
    if provider == "fireworks":
        return fireworks.forward
    raise ValueError(f"No handler for provider {provider!r}")


def dispatch_count_tokens(provider: Provider) -> Handler:
    from .providers import anthropic, deepseek, fireworks

    if provider == "anthropic":
        return anthropic.count_tokens
    if provider == "deepseek":
        return deepseek.count_tokens
    if provider == "fireworks":
        return fireworks.count_tokens
    raise ValueError(f"No count_tokens handler for provider {provider!r}")
