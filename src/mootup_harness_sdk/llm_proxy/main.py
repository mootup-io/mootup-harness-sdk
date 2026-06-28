"""uvicorn entrypoint — `python -m mootup_harness_sdk.llm_proxy.main`."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-proxy")
    parser.add_argument(
        "--capture-requests",
        nargs="?",
        const="/tmp/llm-proxy-capture",
        default=None,
        metavar="DIR",
        help=(
            "Capture each request/response (both translation sides) as NDJSON in "
            "DIR (default /tmp/llm-proxy-capture) for debugging. Off by default; "
            "secrets are redacted. Also enabled via the LLM_PROXY_CAPTURE_DIR env."
        ),
    )
    args = parser.parse_args()
    if args.capture_requests:
        os.environ["LLM_PROXY_CAPTURE_DIR"] = args.capture_requests

    capture_dir = os.getenv("LLM_PROXY_CAPTURE_DIR")
    if capture_dir:
        print(
            f"llm-proxy: capturing requests to "
            f"{capture_dir}/llm-proxy-capture.ndjson",
            flush=True,
        )

    port = int(os.getenv("LLM_PROXY_PORT", "8090"))
    uvicorn.run(
        "mootup_harness_sdk.llm_proxy.server:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
