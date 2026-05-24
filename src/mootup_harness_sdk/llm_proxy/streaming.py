"""OpenAI SSE chunk → Anthropic SSE event translator (async generator)."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx


async def openai_chunks_to_anthropic_sse(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    original_model: str,
) -> AsyncIterator[dict[str, str]]:
    """Yield Anthropic-shape SSE events translated from OpenAI streaming chunks.

    Each yielded dict is a sse-starlette EventSourceResponse payload:
    {"event": "<type>", "data": "<json-encoded>"}.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield {
        "event": "message_start",
        "data": json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": original_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }
        ),
    }

    text_block_open = False
    tool_block_open = False
    tool_index = 0
    final_finish: str | None = None
    output_tokens = 0

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line.removeprefix("data: ").strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                fin = choice.get("finish_reason")
                if fin:
                    final_finish = fin
                text = delta.get("content")
                if isinstance(text, str) and text:
                    if not text_block_open:
                        yield {
                            "event": "content_block_start",
                            "data": json.dumps(
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                }
                            ),
                        }
                        text_block_open = True
                    yield {
                        "event": "content_block_delta",
                        "data": json.dumps(
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {"type": "text_delta", "text": text},
                            }
                        ),
                    }
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    if not tool_block_open:
                        if text_block_open:
                            yield {
                                "event": "content_block_stop",
                                "data": json.dumps(
                                    {"type": "content_block_stop", "index": 0}
                                ),
                            }
                            text_block_open = False
                        tool_index = 0
                        yield {
                            "event": "content_block_start",
                            "data": json.dumps(
                                {
                                    "type": "content_block_start",
                                    "index": tool_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tc.get("id")
                                        or f"toolu_{uuid.uuid4().hex[:24]}",
                                        "name": fn.get("name", ""),
                                        "input": {},
                                    },
                                }
                            ),
                        }
                        tool_block_open = True
                    args_frag = fn.get("arguments")
                    if isinstance(args_frag, str) and args_frag:
                        yield {
                            "event": "content_block_delta",
                            "data": json.dumps(
                                {
                                    "type": "content_block_delta",
                                    "index": tool_index,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args_frag,
                                    },
                                }
                            ),
                        }
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    output_tokens = usage.get("completion_tokens", output_tokens)

    if text_block_open:
        yield {
            "event": "content_block_stop",
            "data": json.dumps({"type": "content_block_stop", "index": 0}),
        }
    if tool_block_open:
        yield {
            "event": "content_block_stop",
            "data": json.dumps({"type": "content_block_stop", "index": tool_index}),
        }
    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(final_finish or "stop", "end_turn")
    yield {
        "event": "message_delta",
        "data": json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            }
        ),
    }
    yield {
        "event": "message_stop",
        "data": json.dumps({"type": "message_stop"}),
    }
