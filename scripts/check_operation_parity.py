#!/usr/bin/env python3
"""
Operation Parity Checker

This script checks for parity across different operation categories and formats
(dense, sparse CSR/CSC/COO, quantized) by comparing file structures.

Usage:
    python scripts/check_operation_parity.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def get_operation_files(base_path: Path, category: str) -> Set[str]:
    """
    Get all operation files for a specific category.
    
    Args:
        base_path: Path to the base directory
        category: Category name (e.g., 'arithmetic', 'linear_algebra')
        
    Returns:
        Set of operation file names (without .rs extension)
    """
    category_path = base_path / category
    if not category_path.exists():
        return set()
    
    operations = set()
    for file in category_path.glob('*.rs'):
        if file.name != 'mod.rs':
            operations.add(file.stem)
    
    return operations


def check_sparse_format_parity(sparse_path: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Check parity across sparse formats (CSR, CSC, COO).
    
    Args:
        sparse_path: Path to the sparse/src/formats directory
        
    Returns:
        Dictionary mapping format names to their operation categories
    """
    formats = ['csr', 'csc', 'coo']
    categories = ['arithmetic', 'conversion', 'indexing']
    
    format_ops = {}
    
    for fmt in formats:
        format_ops[fmt] = {}
        format_path = sparse_path / fmt
        
        for category in categories:
            format_ops[fmt][category] = get_operation_files(format_path, category)
    
    return format_ops


def check_backend_operation_parity(backend_path: Path) -> Dict[str, Dict[str, Set[str]]]:
    """
    Check parity across backend operation categories.
    
    Args:
        backend_path: Path to the backend/src directory
        
    Returns:
        Dictionary mapping backend names to their operation categories
    """
    backends = ['cpu', 'gpu', 'tpu', 'npu']
    categories = ['arithmetic', 'linear_algebra', 'activation', 'reduction']
    
    backend_ops = {}
    
    for backend in backends:
        backend_ops[backend] = {}
        backend_base = backend_path / backend
        
        for category in categories:
            backend_ops[backend][category] = get_operation_files(backend_base, category)
    
    return backend_ops


def find_category_gaps(ops_by_entity: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Find missing operations in each category across entities.
    
    Args:
        ops_by_entity: Dictionary mapping entity names to their operation categories
        
    Returns:
        Dictionary mapping entity names to missing operations by category
    """
    # Find the union of all operations in each category
    all_ops_by_category = defaultdict(set)
    for entity_ops in ops_by_entity.values():
        for category, ops in entity_ops.items():
            all_ops_by_category[category].update(ops)
    
    # Find missing operations for each entity
    gaps = {}
    for entity, entity_ops in ops_by_entity.items():
        gaps[entity] = {}
        for category, all_ops in all_ops_by_category.items():
            entity_category_ops = entity_ops.get(category, set())
            missing = all_ops - entity_category_ops
            if missing:
                gaps[entity][category] = sorted(list(missing))
    
    return gaps


def generate_operation_parity_report(
    sparse_ops: Dict[str, Dict[str, Set[str]]],
    sparse_gaps: Dict[str, Dict[str, List[str]]],
    backend_ops: Dict[str, Dict[str, Set[str]]],
    backend_gaps: Dict[str, Dict[str, List[str]]]
) -> str:
    """
    Generate a human-readable operation parity report.
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("Operation Parity Report")
    report.append("=" * 80)
    report.append("")
    
    # Sparse format parity
    report.append("Sparse Format Parity:")
    report.append("-" * 80)
    for fmt, categories in sorted(sparse_ops.items()):
        total_ops = sum(len(ops) for ops in categories.values())
        report.append(f"\n{fmt.upper()}: {total_ops} total operations")
        for category, ops in sorted(categories.items()):
            report.append(f"  {category}: {len(ops)} operations")
    
    if sparse_gaps:
        report.append("\nSparse Format Gaps:")
        for fmt, categories in sorted(sparse_gaps.items()):
            if categories:
                report.append(f"\n{fmt.upper()}:")
                for category, missing in sorted(categories.items()):
                    report.append(f"  {category} (missing {len(missing)}):")
                    for op in missing:
                        report.append(f"    - {op}")
    else:
        report.append("\n✓ All sparse formats have complete parity!")
    
    report.append("")
    
    # Backend operation parity
    report.append("Backend Operation Parity:")
    report.append("-" * 80)
    for backend, categories in sorted(backend_ops.items()):
        total_ops = sum(len(ops) for ops in categories.values())
        report.append(f"\n{backend.upper()}: {total_ops} total operations")
        for category, ops in sorted(categories.items()):
            report.append(f"  {category}: {len(ops)} operations")
    
    if backend_gaps:
        report.append("\nBackend Operation Gaps:")
        for backend, categories in sorted(backend_gaps.items()):
            if categories:
                report.append(f"\n{backend.upper()}:")
                for category, missing in sorted(categories.items()):
                    report.append(f"  {category} (missing {len(missing)}):")
                    for op in missing:
                        report.append(f"    - {op}")
    else:
        report.append("\n✓ All backends have complete parity!")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main entry point for the operation parity checker."""
    # Get the workspace root
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    
    sparse_formats = workspace_root / "sparse" / "src" / "formats"
    backend_src = workspace_root / "backend" / "src"
    
    print("Checking operation parity...")
    print()
    
    # Check sparse format parity
    sparse_ops = {}
    sparse_gaps = {}
    if sparse_formats.exists():
        sparse_ops = check_sparse_format_parity(sparse_formats)
        sparse_gaps = find_category_gaps(sparse_ops)
    else:
        print(f"Warning: Sparse formats directory not found: {sparse_formats}", file=sys.stderr)
    
    # Check backend operation parity
    backend_ops = {}
    backend_gaps = {}
    if backend_src.exists():
        backend_ops = check_backend_operation_parity(backend_src)
        backend_gaps = find_category_gaps(backend_ops)
    else:
        print(f"Warning: Backend source directory not found: {backend_src}", file=sys.stderr)
    
    # Generate and print report
    report = generate_operation_parity_report(sparse_ops, sparse_gaps, backend_ops, backend_gaps)
    print(report)
    
    # Exit with error code if there are gaps
    has_gaps = False
    for categories in sparse_gaps.values():
        if categories:
            has_gaps = True
            break
    
    if not has_gaps:
        for categories in backend_gaps.values():
            if categories:
                has_gaps = True
                break
    
    if has_gaps:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
