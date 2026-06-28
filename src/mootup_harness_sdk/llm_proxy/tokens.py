"""Local input-token estimation.

Used as a count_tokens fallback for providers whose Anthropic-compatible
endpoint lacks a ``/v1/messages/count_tokens`` route (e.g. Fireworks 404s it).
Claude Code calls count_tokens only to size context before a turn, so a rough
character-based estimate keeps the turn unblocked. Providers that DO support
count_tokens (Anthropic, DeepSeek) forward for an exact count and use this only
if the upstream fails.
"""

from __future__ import annotations

import json
from typing import Any

# Rough bytes/chars-per-token. Real tokenizers vary by model; ~4 chars/token is
# the standard back-of-envelope figure and is plenty for context sizing.
_CHARS_PER_TOKEN = 4


def estimate_input_tokens(body: dict[str, Any]) -> int:
    """Estimate the input token count of an Anthropic Messages request body."""
    chars = 0

    system = body.get("system")
    if isinstance(system, str):
        chars += len(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict):
                chars += len(str(block.get("text", "")))

    for message in body.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    chars += len(str(block.get("text", "")))
                    chars += len(str(block.get("content", "")))
                    if block.get("input") is not None:
                        chars += len(json.dumps(block["input"]))

    for tool in body.get("tools", []) or []:
        chars += len(json.dumps(tool))

    return max(1, chars // _CHARS_PER_TOKEN)


def estimate_count_tokens_body(body: dict[str, Any]) -> bytes:
    """JSON-encoded Anthropic count_tokens response for a local estimate."""
    return json.dumps({"input_tokens": estimate_input_tokens(body)}).encode()
