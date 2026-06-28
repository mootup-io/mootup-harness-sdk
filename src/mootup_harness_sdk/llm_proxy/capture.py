"""Optional request/response capture for debugging proxy translation.

Enabled by setting ``LLM_PROXY_CAPTURE_DIR`` (the ``--capture-requests`` CLI flag
sets a default). While enabled, each exchange appends one NDJSON record to
``<dir>/llm-proxy-capture.ndjson`` capturing **both sides** of the translation —
the inbound Anthropic body, the translated upstream (OpenAI/provider) body, the
upstream status + response, and the translated Anthropic response — plus model,
provider, stream, and latency. Bearer tokens and provider API keys are redacted;
capture is best-effort and never raises (it must not break the proxy).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("convo.llm_proxy.capture")

_CAPTURE_FILE = "llm-proxy-capture.ndjson"
_REDACT_HEADER_KEYS = {"authorization", "x-api-key"}
_MAX_FIELD_CHARS = 64 * 1024  # clip any single captured body


def capture_dir() -> str | None:
    """The capture directory, or None when capture is disabled."""
    value = os.getenv("LLM_PROXY_CAPTURE_DIR", "").strip()
    return value or None


def enabled() -> bool:
    return capture_dir() is not None


def _redact_headers(headers: dict[str, str] | None) -> dict[str, str]:
    return {
        k: ("<redacted>" if k.lower() in _REDACT_HEADER_KEYS else v)
        for k, v in (headers or {}).items()
    }


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_FIELD_CHARS:
        return (
            value[:_MAX_FIELD_CHARS] + f"…<+{len(value) - _MAX_FIELD_CHARS}B clipped>"
        )
    return value


def record(**fields: Any) -> None:
    """Append one NDJSON capture record. No-op when capture is disabled.

    ``upstream_headers`` are redacted; ``upstream_response`` /
    ``anthropic_response`` are clipped. Never raises.
    """
    directory = capture_dir()
    if not directory:
        return
    try:
        rec = dict(fields)
        if "upstream_headers" in rec:
            rec["upstream_headers"] = _redact_headers(rec.get("upstream_headers"))
        for key in ("upstream_response", "anthropic_response"):
            if key in rec:
                rec[key] = _clip(rec[key])
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, _CAPTURE_FILE), "a") as handle:
            handle.write(json.dumps(rec, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 — capture must never break the proxy
        logger.warning("capture.record failed (ignored): %s", exc)
