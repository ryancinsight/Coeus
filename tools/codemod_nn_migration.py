#!/usr/bin/env python3
"""
Simple codemod for the Coeus workspace to assist NN crate API migration.
- Dry-run by default: prints diffs to stdout
- --apply to write changes
- Carefully scoped: only updates Tensor<T> -> Tensor<T, CpuBackend> where no comma exists in angle brackets
- Adds CpuBackend::default() to Tensor::from_vec calls missing backend

Usage:
  python tools/codemod_nn_migration.py --dry-run --root d:/coeus/nn/src
  python tools/codemod_nn_migration.py --apply --root d:/coeus/nn/src

Limitations / Safety:
- Regex-based; intended for small automatic edits + human review; must not be used blindly across entire repo without code review.
- Produces unified diffs; each file change should be split into module-level PRs.

"""
import argparse
import re
import sys
from pathlib import Path
from difflib import unified_diff

TENSOR_TYPE_RE = re.compile(r"\bTensor\s*<\s*([A-Za-z0-9_:]+)\s*>")  # only simple single-identifier dtype match
FROM_VEC_RE = re.compile(r"Tensor::from_vec\s*\((.*?)\)\s*")


def transform_content(text: str) -> (str, list):
    """Return transformed text and list of notes (why changed)."""
    notes = []
    # 1) Replace simple Tensor<T> -> Tensor<T, CpuBackend> only when bracket contains no comma
    def tensor_repl(m):
        inner = m.group(1)
        if ',' in inner:
            return m.group(0)  # already has backend or complex
        repl = f"Tensor<{inner}, CpuBackend>"
        notes.append(f"Replaced Tensor<{inner}> -> {repl}")
        return repl

    new = TENSOR_TYPE_RE.sub(tensor_repl, text)

    # 2) Replace Tensor::from_vec(data, shape) -> Tensor::from_vec(CpuBackend::default(), data, shape).unwrap()
    def from_vec_repl(m):
        args = m.group(1)
        # Safely split top-level args by comma; avoid nested commas by simple heuristic
        parts = [p.strip() for p in args.split(',')]
