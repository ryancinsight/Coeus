#!/usr/bin/env python3
"""
Codemod to prefix local variables that are likely intentionally unused with an underscore
Only touches nn/src files and targets variable names matching these patterns:
 - *_data
 - *_broadcast
 - positions, input_data, new_seq_len, n_embd, rnn (local test vars)

Behavior:
 - Dry-run by default: prints unified diffs
 - --apply to mutate files
 - Conservative: only replaces let <name>: patterns or let <name> = <expr>; where name matches the pattern and not already prefixed with '_'

Usage:
  python tools/codemod_prefix_unused.py --dry-run --root d:/coeus/nn/src
  python tools/codemod_prefix_unused.py --apply --root d:/coeus/nn/src

Safety:
 - Regex-based and conservative to reduce false positives. Always review diffs.
"""
import argparse
import re
from pathlib import Path
from difflib import unified_diff

VAR_RE = re.compile(r"\blet\s+(?P<name>[A-Za-z0-9_]+)\b")
TARGET_SUFFIXES = ("_data", "_broadcast")
TARGET_NAMES = set(["positions","input_data","new_seq_len","n_embd","rnn"])


def transform_file(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    changed = False
    new_lines = []
    for line in lines:
        m = VAR_RE.search(line)
        if not m:
            new_lines.append(line)
            continue
        name = m.group('name')
        if name.startswith('_'):
            new_lines.append(line)
            continue
        if any(name.endswith(suf) for suf in TARGET_SUFFIXES) or name in TARGET_NAMES:
            # Replace first occurrence of 'let name' with 'let _name'
            new_line = line.replace(f"let {name}", f"let _{name}", 1)
            new_lines.append(new_line)
            changed = True
        else:
            new_lines.append(line)
    if not changed:
        return None
    return ''.join(new_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    files = list(root.rglob('*.rs'))
    for f in files:
        new = transform_file(f)
        if new is None:
            continue
        old = f.read_text(encoding='utf-8')
        diff = ''.join(unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=str(f), tofile=str(f)+".new"))
        print(diff)
        if args.apply:
            f.write_text(new, encoding='utf-8')

if __name__ == '__main__':
    main()
