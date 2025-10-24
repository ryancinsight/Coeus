#!/usr/bin/env python3
import json
import sys
from collections import defaultdict

# Read JSON input from stdin or file
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        data = f.read()
else:
    data = sys.stdin.read()

warnings = defaultdict(lambda: defaultdict(int))
warnings_by_crate = defaultdict(int)
total_warnings = 0

for line in data.strip().split('\n'):
    if not line.strip():
        continue
    try:
        obj = json.loads(line)
        if (obj.get('reason') == 'compiler-message' and
            obj.get('message', {}).get('level') == 'warning'):
            code = obj['message'].get('code', 'UNKNOWN')
            if code:
                package_id = obj.get('package_id', 'unknown')
                warnings[code][package_id] += 1
                warnings_by_crate[package_id] += 1
                total_warnings += 1
    except json.JSONDecodeError as e:
        continue

# Output results
print(f"Total warnings: {total_warnings}")
print()

print("Warnings by category:")
for code, packages in sorted(warnings.items()):
    count = sum(packages.values())
    print(f"  {code}: {count}")
print()

print("Warnings by crate:")
for crate, count in sorted(warnings_by_crate.items()):
    print(f"  {crate}: {count}")
print()

print("Detailed breakdown:")
for code in sorted(warnings.keys()):
    print(f"{code}:")
    for package, count in sorted(warnings[code].items()):
        print(f"  - {package}: {count}")
