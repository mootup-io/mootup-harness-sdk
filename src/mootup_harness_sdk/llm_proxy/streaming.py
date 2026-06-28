"""OpenAI SSE chunk → Anthropic SSE event translator (async generator)."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from . import capture
from .translators.openai_format import THINKING_SIGNATURE


def _event(event_type: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event_type, "data": json.dumps(data)}


async def openai_chunks_to_anthropic_sse(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    original_model: str,
    *,
    capture_meta: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, str]]:
    """Yield Anthropic-shape SSE events translated from OpenAI streaming chunks.

    Handles reasoning models: a leading run of ``delta.reasoning_content`` is
    emitted as an Anthropic ``thinking`` content block, closed with a synthesized
    ``signature_delta`` when real ``content``/``tool_calls`` begin. Block indices
    are assigned monotonically (thinking, then text, then tool_use), so Claude
    Code can store the thinking block and echo it back next turn.

    Each yielded dict is a sse-starlette EventSourceResponse payload:
    {"event": "<type>", "data": "<json-encoded>"}.
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _event(
        "message_start",
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
        },
    )

    next_index = 0
    thinking_index: int | None = None
    thinking_closed = False
    text_index: int | None = None
    text_closed = False
    tool_index: int | None = None
    final_finish: str | None = None
    output_tokens = 0

    # Optional debug capture: tee the raw upstream SSE (bounded) so a turn's
    # provider-side output can be inspected next to the translated request.
    cap_on = capture_meta is not None and capture.enabled()
    raw_lines: list[str] = []
    raw_len = 0
    upstream_status: int | None = None

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if cap_on:
                upstream_status = getattr(resp, "status_code", None)
            async for line in resp.aiter_lines():
                if cap_on and line and raw_len < 32768:
                    raw_lines.append(line)
                    raw_len += len(line)
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

                # 1) reasoning_content → leading thinking block.
                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    if thinking_index is None:
                        thinking_index = next_index
                        next_index += 1
                        yield _event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": thinking_index,
                                "content_block": {"type": "thinking", "thinking": ""},
                            },
                        )
                    yield _event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": thinking_index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        },
                    )

                text = delta.get("content")
                tool_calls = delta.get("tool_calls") or []
                has_text = isinstance(text, str) and bool(text)

                # Close the thinking block once real output begins.
                if (
                    (has_text or tool_calls)
                    and thinking_index is not None
                    and not thinking_closed
                ):
                    yield _event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": thinking_index,
                            "delta": {
                                "type": "signature_delta",
                                "signature": THINKING_SIGNATURE,
                            },
                        },
                    )
                    yield _event(
                        "content_block_stop",
                        {"type": "content_block_stop", "index": thinking_index},
                    )
                    thinking_closed = True

                # 2) text block.
                if has_text:
                    if text_index is None:
                        text_index = next_index
                        next_index += 1
                        yield _event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": text_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    yield _event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )

                # 3) tool_use block (single block; arg fragments stream in).
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    if tool_index is None:
                        if text_index is not None and not text_closed:
                            yield _event(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": text_index},
                            )
                            text_closed = True
                        tool_index = next_index
                        next_index += 1
                        yield _event(
                            "content_block_start",
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
                            },
                        )
                    args_frag = fn.get("arguments")
                    if isinstance(args_frag, str) and args_frag:
                        yield _event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": tool_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args_frag,
                                },
                            },
                        )

                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    output_tokens = usage.get("completion_tokens", output_tokens)

    if cap_on and capture_meta is not None:
        capture.record(
            stream=True,
            upstream_status=upstream_status,
            upstream_response="\n".join(raw_lines),
            anthropic_response=None,  # translated SSE not buffered
            **capture_meta,
        )

    # Close any blocks still open (e.g. a reasoning-only or text-only turn).
    if thinking_index is not None and not thinking_closed:
        yield _event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": thinking_index,
                "delta": {"type": "signature_delta", "signature": THINKING_SIGNATURE},
            },
        )
        yield _event(
            "content_block_stop",
            {"type": "content_block_stop", "index": thinking_index},
        )
    if text_index is not None and not text_closed:
        yield _event(
            "content_block_stop", {"type": "content_block_stop", "index": text_index}
        )
    if tool_index is not None:
        yield _event(
            "content_block_stop", {"type": "content_block_stop", "index": tool_index}
        )

    stop_reason = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }.get(final_finish or "stop", "end_turn")
    yield _event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    yield _event("message_stop", {"type": "message_stop"})
