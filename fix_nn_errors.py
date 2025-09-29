#!/usr/bin/env python3
"""
Script to systematically fix Tensor<T> → Tensor<T, CpuBackend> migration errors in nn crate.

Fixes:
1. Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), ...) → Tensor::from_vec(CpuBackend::default(), ...)
2. .unwrap().unwrap() → .unwrap()
3. Malformed .to_vec()) patterns
"""

import os
import re
import sys

def fix_tensor_from_vec(content):
    """Fix Tensor::from_vec calls with wrong number of arguments."""
    # Pattern 1: Tensor::from_vec(CpuBackend::default(), CpuBackend::default(), expr, vec![...])
    pattern1 = r'Tensor::from_vec\(CpuBackend::default\(\),\s*CpuBackend::default\(\),\s*([^,]+),\s*([^)]+)\)'
    replacement1 = r'Tensor::from_vec(CpuBackend::default(), \1, \2)'

    # Pattern 2: Tensor::from_vec(CpuBackend::default(), CpuBackend::default().unwrap(), expr, vec![...])
    pattern2 = r'Tensor::from_vec\(CpuBackend::default\(\),\s*CpuBackend::default\(\)\.unwrap\(\),\s*([^,]+),\s*([^)]+)\)'
    replacement2 = r'Tensor::from_vec(CpuBackend::default(), \1, \2)'

    content = re.sub(pattern1, replacement1, content)
    content = re.sub(pattern2, replacement2, content)

    return content

def fix_broken_to_vec_patterns(content):
    """Fix broken .to_vec()) patterns that my script created."""
    # Pattern: Tensor::from_vec(CpuBackend::default(), CpuBackend::default().unwrap()).unwrap().to_vec()).unwrap()
    # This is malformed - should be: Tensor::from_vec(CpuBackend::default(), data, shape)

    # Find patterns where we have the broken syntax and fix them
    # This is a complex pattern, let's handle it line by line

    lines = content.split('\n')
    fixed_lines = []

    for line in lines:
        # Fix the malformed pattern: Tensor::from_vec(CpuBackend::default(), CpuBackend::default().unwrap()).unwrap().to_vec()).unwrap()
        if 'Tensor::from_vec(CpuBackend::default(), CpuBackend::default().unwrap()).unwrap().to_vec()).unwrap()' in line:
            # This is too broken, skip for now and we'll handle manually
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_double_unwrap(content):
    """Fix double unwrap calls."""
    # Pattern: .unwrap().unwrap()
    # Replace with: .unwrap()
    return content.replace('.unwrap().unwrap()', '.unwrap()')

def fix_unwrap_calls(content):
    """Fix various unwrap call issues."""
    # Fix Tensor::zeros unwrap calls
    content = re.sub(r'Tensor::zeros\(([^)]+)\)\.unwrap\(\)', r'Tensor::zeros(\1).unwrap()', content)

    # Fix broken vec! patterns: vec![number).unwrap() → vec![number], vec![1]).unwrap()
    content = re.sub(r'vec!\[([^]]+)\)\.unwrap\(\)', r'vec![\1], vec![1]).unwrap()', content)

    # Fix malformed item() calls: loss.item(], vec![1]).unwrap() → loss.item().unwrap()
    content = re.sub(r'loss\.item\(\], vec!\[1\]\)\.unwrap\(\)', r'loss.item().unwrap()', content)

    # Fix malformed forward calls: forward(&input, &target], vec![1]).unwrap() → forward(&input, &target).unwrap()
    content = re.sub(r'\.forward\(&[^,]+, &[^)]+\], vec!\[1\]\)\.unwrap\(\)', lambda m: m.group(0).replace('], vec![1]).unwrap()', ').unwrap()'), content)

    # Fix other unwrap issues
    return content

def main():
    nn_src_dir = 'nn/src'

    if not os.path.exists(nn_src_dir):
        print(f"Directory {nn_src_dir} not found")
        sys.exit(1)

    files_fixed = 0

    for root, dirs, files in os.walk(nn_src_dir):
        for file in files:
            if file.endswith('.rs'):
                filepath = os.path.join(root, file)

                with open(filepath, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                modified_content = original_content

                # Apply fixes
                modified_content = fix_tensor_from_vec(modified_content)
                modified_content = fix_double_unwrap(modified_content)
                modified_content = fix_unwrap_calls(modified_content)
                modified_content = fix_broken_to_vec_patterns(modified_content)

                # Only write if content changed
                if modified_content != original_content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    print(f"Fixed: {filepath}")
                    files_fixed += 1

    print(f"\nFixed {files_fixed} files")

if __name__ == '__main__':
    main()
