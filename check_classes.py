#!/usr/bin/env python3
"""Check what classes are available in PyCoeus"""

import pycoeus as pc

print("Available attributes in pycoeus:")
for attr in sorted(dir(pc)):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\nChecking specific classes:")
classes_to_check = ['PyLinear', 'Linear', 'PyConv2d', 'Conv2d', 'PyReLU', 'ReLU', 'PyMSELoss', 'MSELoss']

for cls_name in classes_to_check:
    if hasattr(pc, cls_name):
        print(f"✅ {cls_name}: Available")
    else:
        print(f"❌ {cls_name}: Not found")
