#!/usr/bin/env python3
"""Classify a revision range for the Coeus CI test path.

The hook contract is independent of Rust compilation.  A range containing
only the installed hook files and their Python checker contract can therefore
run the hook tests directly.  A range containing only Markdown/license
artifacts (PM boards, ADRs, the changelog, README) cannot affect Rust
behavior either -- nothing under `crates/` or the build/toolchain manifests
reads them at compile or test time -- so it skips native coverage entirely
instead of paying a full workspace compile for a change no Rust build could
observe.  Every other range keeps the native and doctest coverage.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import PurePosixPath
from typing import Iterable

HOOK_CONTRACT_PATHS = frozenset({
    "scripts/lockfile.py",
    "scripts/tests/test_hooks.py",
})

# Suffixes/paths that can only hold prose or licensing text: no crate, build
# script, or CI job reads these at compile or test time. Kept narrow and
# additive -- a path outside this set keeps native coverage by default.
DOC_ONLY_SUFFIXES = (".md",)
DOC_ONLY_PATHS = frozenset({
    "LICENSE-APACHE",
    "LICENSE-MIT",
})


def is_hook_contract_path(path: str) -> bool:
    """Return whether ``path`` belongs to the hook implementation contract."""

    normalized = str(PurePosixPath(path))
    return normalized.startswith(".githooks/") or normalized in HOOK_CONTRACT_PATHS


def is_doc_only_path(path: str) -> bool:
    """Return whether ``path`` can only affect prose/licensing, never a build."""

    normalized = str(PurePosixPath(path))
    return normalized in DOC_ONLY_PATHS or normalized.endswith(DOC_ONLY_SUFFIXES)


def classify(paths: Iterable[str]) -> str:
    """Return ``hook-only``, ``skip``, or ``native`` for the changed paths."""

    changed = tuple(path for path in paths if path)
    if not changed:
        return "native"
    if all(is_hook_contract_path(path) for path in changed):
        return "hook-only"
    if all(is_doc_only_path(path) for path in changed):
        return "skip"
    return "native"


def changed_paths(base: str, head: str) -> tuple[str, ...]:
    """Read the committed paths changed between two revisions."""

    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base}..{head}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in result.stdout.splitlines() if line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base revision")
    parser.add_argument("--head", required=True, help="head revision")
    args = parser.parse_args()
    paths = changed_paths(args.base, args.head)
    selection = classify(paths)
    print(f"changed paths: {len(paths)}")
    print(f"test path: {selection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
