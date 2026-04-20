"""AH-g parity — Session fields must match canonical contract.

Contract at ``<convo>/docs/api/sdk-harness-contract.json`` (mirrored to
``tests/fixtures/sdk-harness-contract.json`` via ``scripts/sync-contract.py``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from mootup_harness_sdk import MootupNotOrientedError, Session

CONTRACT_PATH = Path(__file__).parent / "fixtures" / "sdk-harness-contract.json"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def test_session_field_names_match_contract() -> None:
    """R-parity-py — Session dataclass exposes exactly the Python-named fields."""
    contract = _contract()
    expected = set(contract["naming"]["python"].values())
    actual = {f.name for f in dataclasses.fields(Session)}
    assert actual == expected, (
        f"Session fields drift from contract: expected={expected}, actual={actual}"
    )


def test_error_class_name_matches_contract() -> None:
    """Error class name matches contract `errors.not_oriented.python`."""
    contract = _contract()
    assert MootupNotOrientedError.__name__ == contract["errors"]["not_oriented"]["python"]


def test_no_mcp_sdk_dep_in_pyproject() -> None:
    """Inv 8 — pyproject.toml declares no dependency on the `mcp` SDK package."""
    pyproject = (
        Path(__file__).parent.parent / "pyproject.toml"
    ).read_text()
    # Split by sections and scan only [project]'s dependencies lines.
    # Simple grep — the explicit dependency array must not mention `mcp` as a dist name.
    # Allowed: comments, headers, module references.
    suspicious: list[str] = []
    for line in pyproject.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if (
            stripped.startswith('"mcp')
            or stripped.startswith("'mcp")
            or stripped.startswith('"modelcontextprotocol')
            or stripped.startswith("'modelcontextprotocol")
        ):
            suspicious.append(line)
    assert suspicious == [], (
        f"pyproject.toml declares forbidden MCP SDK dependency: {suspicious}"
    )


def test_contract_stamps_oas_version_ref() -> None:
    """Inv 4 — contract JSON stamps which OAS version it derives from."""
    contract = _contract()
    oas_version = contract.get("oas_version_ref")
    assert isinstance(oas_version, str)
    parts = oas_version.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
