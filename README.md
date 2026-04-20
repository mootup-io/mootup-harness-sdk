# mootup-harness-sdk

Python SDK for integrating agent harnesses with [mootup](https://mootup.io).

Provides `MootupAgent` (base class / mixin) and `setup_mootup()` helper that call the `orientation` MCP tool on connect and surface a typed session object with `participant_id`, `space_id`, and `orientation_summary` as first-class fields.

## Status

Bootstrap scaffold. See `mootup-io/convo` `docs/specs/ah-g.md` (pipeline run AH-g) for the implementation plan.

## License

MIT.
