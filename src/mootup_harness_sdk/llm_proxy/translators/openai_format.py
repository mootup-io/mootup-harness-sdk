"""Bidirectional translator: Anthropic Messages ↔ OpenAI Chat Completions."""

from __future__ import annotations

import json
import uuid
from typing import Any

# Synthesized signature for thinking blocks. OpenAI-style reasoning providers
# (DeepSeek) don't emit Anthropic thinking signatures, but Claude Code needs a
# non-empty signature to store + echo a thinking block back. It is never
# cryptographically verified — it round-trips through this proxy, which maps the
# thinking block back to reasoning_content — so any stable non-empty string works.
THINKING_SIGNATURE = "moot-proxy-reasoning"


def anthropic_to_openai_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages request into OpenAI Chat Completions shape."""
    messages: list[dict[str, Any]] = []

    system = body.get("system")
    if isinstance(system, str):
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text_parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        if text_parts:
            messages.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in body.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text_parts = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for block in content or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                # Reasoning models (e.g. deepseek-v4-pro) require the prior
                # turn's reasoning to be echoed back, or they 400. Carry the
                # thinking text into reasoning_content on the OpenAI message.
                thinking_parts.append(block.get("thinking", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif btype == "tool_result":
                tool_content = block.get("content", "")
                if isinstance(tool_content, list):
                    tool_content = "".join(
                        b.get("text", "")
                        for b in tool_content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": tool_content,
                    }
                )
        text_content = "\n".join(text_parts) if text_parts else ""
        reasoning_content = "\n".join(p for p in thinking_parts if p) or None
        if role == "assistant" and tool_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls,
            }
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            messages.append(assistant_msg)
        elif tool_results:
            messages.extend(tool_results)
        else:
            plain_msg: dict[str, Any] = {"role": role, "content": text_content}
            if role == "assistant" and reasoning_content:
                plain_msg["reasoning_content"] = reasoning_content
            messages.append(plain_msg)

    out: dict[str, Any] = {
        "model": body["model"],
        "messages": messages,
        "max_tokens": body.get("max_tokens"),
        "stream": bool(body.get("stream")),
    }
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    if "tools" in body:
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in body["tools"]
        ]
    return out


def openai_to_anthropic_response(
    resp: dict[str, Any], *, original_model: str
) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response into Anthropic Messages shape."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    content_blocks: list[dict[str, Any]] = []
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        # Thinking must come first in Anthropic content. Synthesize a signature
        # so Claude Code stores + echoes it (round-tripped back to DeepSeek as
        # reasoning_content on the next turn).
        content_blocks.append(
            {"type": "thinking", "thinking": reasoning, "signature": THINKING_SIGNATURE}
        )
    text = msg.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "input": args,
            }
        )
    finish = choice.get("finish_reason")
    finish_key = finish if isinstance(finish, str) else ""
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(finish_key, "end_turn")
    usage_raw = resp.get("usage")
    usage: dict[str, Any] = usage_raw if isinstance(usage_raw, dict) else {}
    return {
        "id": resp.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
