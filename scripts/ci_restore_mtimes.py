#!/usr/bin/env python3
"""Restore tracked file mtimes to the commit time that last touched them.

`actions/checkout` sets every checked-out file's mtime to the checkout
instant, so a content-identical file gets a different mtime on every run.
Cargo's fingerprint freshness check for a local (workspace) crate compares
each input file's *current* mtime against the value recorded the last time
that crate built; any difference -- not just "newer" -- marks the crate
dirty. A fresh checkout therefore makes every workspace crate look dirty on
every run, regardless of whether its content changed, which defeats
Swatinem/rust-cache's `cache-workspace-crates` option: the cached `target/`
directory restores the old build outputs, but the mismatched source mtime
still forces a full rebuild on top of them.

Deriving each tracked file's mtime from git history instead -- the commit
time of the most recent commit that touched it, reachable from HEAD -- makes
the mtime a function of content and history alone. Two checkouts of the same
file content (on any branch, at any later moment) then produce the identical
mtime, so Cargo's exact-match freshness check sees genuinely unchanged
crates as unchanged and skips recompiling them; a file a branch actually
modified gets a new, different timestamp from its modifying commit, so
Cargo still (correctly) rebuilds whatever that touches.

This walks `git log` once (rather than once per file) and requires full
history (`fetch-depth: 0`) -- a shallow checkout has only its HEAD commit
available, which would stamp every tracked file with the same timestamp and
give unrelated branches unrelated stamps for identical content, defeating
the scheme.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def last_commit_mtimes(root: Path) -> dict[str, int]:
    """Return ``{path: unix_timestamp}`` for the commit that most recently
    touched each path reachable from HEAD.

    `git log` visits commits newest-first, so the first time a path appears
    in the walk is its most recent touch; ``setdefault`` keeps only that
    first (latest) timestamp per path.
    """

    result = subprocess.run(
        ["git", "log", "--name-only", "--no-renames", "--format=commit %ct"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    mtimes: dict[str, int] = {}
    current_ts: int | None = None
    for line in result.stdout.splitlines():
        if line.startswith("commit "):
            current_ts = int(line.split(" ", 1)[1])
        elif line and current_ts is not None:
            mtimes.setdefault(line, current_ts)
    return mtimes


def tracked_paths(root: Path) -> tuple[str, ...]:
    """Read the paths Git tracks in the current worktree."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
        cwd=root,
    )
    return tuple(p.decode() for p in result.stdout.split(b"\0") if p)


def restore(root: Path) -> int:
    """Set each tracked file's mtime to its last-commit time; return the
    count of files whose mtime was set."""

    mtimes = last_commit_mtimes(root)
    applied = 0
    for rel_path in tracked_paths(root):
        timestamp = mtimes.get(rel_path)
        if timestamp is None:
            # Reachable from the index but not from any walked commit (e.g. a
            # merge conflict marker path); leave the checkout mtime in place.
            continue
        path = root / rel_path
        try:
            os.utime(path, (timestamp, timestamp))
            applied += 1
        except FileNotFoundError:
            # A tracked path absent from the worktree (submodule gitlink,
            # symlink target moved) -- nothing to stamp.
            continue
    return applied


def main() -> int:
    root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    root = Path(root_result.stdout.strip())
    applied = restore(root)
    print(f"restored mtimes: {applied}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
