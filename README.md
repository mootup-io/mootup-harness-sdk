# mootup-harness-sdk

Python SDK for integrating agent harnesses with [mootup](https://mootup.io).

Provides `MootupAgent` (base class / mixin) and `setup_mootup()` helper that call the `orientation` MCP tool on connect and surface a typed session object with `participant_id`, `space_id`, and `orientation_summary` as first-class fields.

## Status

Bootstrap scaffold. See `mootup-io/convo` `docs/specs/ah-g.md` (pipeline run AH-g) for the implementation plan.


## LLM proxy surfaces

The optional `mootup_harness_sdk.llm_proxy` FastAPI app exposes two client-facing surfaces:

- `POST /v1/messages` and `POST /v1/messages/count_tokens` for Claude Code's Anthropic Messages-compatible flow.
- `POST /v1/chat/completions` for Codex or other OpenAI-compatible clients. This route accepts the proxy shared secret as the bearer token and forwards OpenAI-shaped requests unchanged to OpenAI-compatible upstreams selected by model prefix (`gpt-*`, `o1*`, `o3*`, `o4*`, `deepseek-*`, `fireworks/*`, `accounts/fireworks/*`). Anthropic `claude-*` models remain on `/v1/messages`.

A Codex custom provider can point at the proxy with a config like:

```toml
model_provider = "moot-llm-proxy"

[model_providers.moot-llm-proxy]
name = "Moot LLM proxy"
base_url = "http://127.0.0.1:8090/v1"
env_key = "LLM_PROXY_SHARED_SECRET"
```

## License

MIT.
