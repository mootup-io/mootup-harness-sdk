"""Multi-provider LLM proxy (Run D-1).

Devcontainer-local standalone FastAPI app that accepts Anthropic Messages
API requests, routes by inbound `model` field to Anthropic / DeepSeek /
Fireworks upstreams, and translates non-Anthropic responses to Anthropic
shape. See docs/specs/d-1-llm-proxy.md.
"""
