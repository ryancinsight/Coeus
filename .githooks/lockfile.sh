#!/usr/bin/env bash
# Git entry points supply their diagnostic name and staged/full checker mode.
# Install with: git config core.hooksPath .githooks
set -euo pipefail

hook="$1"
mode="$2"
repo_root="$(git rev-parse --show-toplevel)"
checker="$repo_root/scripts/lockfile.py"

if [ ! -f "$checker" ]; then
  echo "${hook}: scripts/lockfile.py not present; lockfile not verified" >&2
  exit 1
fi

python_bin="${PYTHON:-}"
if [ -z "$python_bin" ]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
fi

if [ -z "$python_bin" ]; then
  echo "${hook}: no python interpreter found; lockfile not verified" >&2
  exit 1
fi

# Preserve the checker's exit status and diagnostic without reclassifying faults.
exec "$python_bin" "$checker" "$mode"
