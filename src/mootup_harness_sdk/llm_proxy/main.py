"""uvicorn entrypoint — `python -m mootup_harness_sdk.llm_proxy.main`."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    port = int(os.getenv("LLM_PROXY_PORT", "8090"))
    uvicorn.run(
        "mootup_harness_sdk.llm_proxy.server:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
