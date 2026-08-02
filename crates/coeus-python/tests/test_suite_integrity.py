"""Static integrity checks for the Python binding test suite."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


def test_pytorch_parity_has_unique_test_names() -> None:
    """Every top-level parity test must survive Python module construction."""
    parity_path = Path(__file__).with_name("test_pytorch_parity.py")
    module = ast.parse(parity_path.read_text(encoding="utf-8"), filename=str(parity_path))
    names = [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    assert duplicates == [], f"duplicate top-level pytest names: {duplicates}"
