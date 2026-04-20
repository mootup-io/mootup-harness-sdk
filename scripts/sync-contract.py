#!/usr/bin/env python3
"""AH-g: sync the canonical sdk-harness contract from convo into test fixtures.

Canonical source: ``<parent>/convo/docs/api/sdk-harness-contract.json`` — the
mirrored layout used by moot-cli-js's ``sync-contract.mjs``.

Target: ``tests/fixtures/sdk-harness-contract.json``. Parity tests consume the
target; the committed copy is the source of truth for CI. Re-run this script
after the canonical contract changes in convo.

Inv 4: also asserts that this SDK's ``pyproject.toml`` version prefix is
greater than or equal to the contract's ``oas_version_ref``. Build-fail on
mismatch so operators catch drift early.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
SOURCE = PKG_ROOT.parent / "convo" / "docs" / "api" / "sdk-harness-contract.json"
TARGET = PKG_ROOT / "tests" / "fixtures" / "sdk-harness-contract.json"


def _parse_semver_prefix(version: str) -> tuple[int, ...]:
    # Strip PEP 440 pre-release suffix (`rc0`, `a1`, etc.) + everything after.
    base = re.split(r"[^0-9.]", version)[0].rstrip(".")
    return tuple(int(part or 0) for part in base.split("."))


def _project_version(pyproject: Path) -> str:
    text = pyproject.read_text()
    match = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find version in {pyproject}")
    return match.group(1)


def main() -> int:
    if not SOURCE.exists():
        print(
            f"sync-contract — canonical contract not found at {SOURCE}",
            file=sys.stderr,
        )
        print(
            "Expected layout: <parent>/convo and <parent>/mootup-harness-sdk as siblings.",
            file=sys.stderr,
        )
        print(
            "If the convo repo lives elsewhere, copy the file manually or adjust this script.",
            file=sys.stderr,
        )
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, TARGET)
    print(f"sync-contract — copied {SOURCE}\n                → {TARGET}")

    contract = json.loads(TARGET.read_text())
    oas_version_ref = contract.get("oas_version_ref")
    pkg_version = _project_version(PKG_ROOT / "pyproject.toml")
    if oas_version_ref and _parse_semver_prefix(pkg_version) < _parse_semver_prefix(
        oas_version_ref
    ):
        print(
            f"sync-contract — SDK version {pkg_version} is below contract "
            f"oas_version_ref {oas_version_ref}.",
            file=sys.stderr,
        )
        print(
            "Bump the SDK version to match (or post-date) the convo OAS before syncing.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
