#!/usr/bin/env python3
"""
Dispatch Chain Verification Script

Analyzes the dispatch architecture from tensor → dense/sparse → backend
to ensure correct layering and no crossover violations.

Usage:
    python scripts/verify_dispatch_chain.py
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def find_trait_methods(file_path: Path) -> Dict[str, List[str]]:
    """Extract trait method definitions from a Rust file."""
    traits = {}
    current_trait = None
    
    if not file_path.exists():
        return traits
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find trait definitions
    trait_pattern = re.compile(r'pub trait (\w+)<[^>]*>?\s*\{', re.MULTILINE)
    fn_pattern = re.compile(r'fn (\w+)\s*<[^>]*>?\s*\(', re.MULTILINE)
    
    # Simple trait extraction
    lines = content.split('\n')
    brace_depth = 0
    
    for line in lines:
        # Check for trait definition
        trait_match = re.search(r'pub trait (\w+)', line)
        if trait_match and '{' in line:
            current_trait = trait_match.group(1)
            traits[current_trait] = []
            brace_depth = 1
            continue
        
        if current_trait:
            brace_depth += line.count('{') - line.count('}')
            
            # Extract function names
            fn_match = re.search(r'fn (\w+)\s*[<(]', line)
            if fn_match:
                traits[current_trait].append(fn_match.group(1))
            
            if brace_depth <= 0:
                current_trait = None
    
    return traits


def check_dispatch_implementation(file_path: Path) -> Dict[str, List[str]]:
    """Check how dispatch.rs implements trait methods."""
    implementations = {}
    
    if not file_path.exists():
        return implementations
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find impl blocks
    impl_pattern = re.compile(
        r'impl<[^>]+>\s+(\w+)<[^>]+>\s+for\s+(\w+)', 
        re.MULTILINE
    )
    
    for match in impl_pattern.finditer(content):
        trait_name = match.group(1)
        storage_type = match.group(2)
        
        key = f"{trait_name} for {storage_type}"
        implementations[key] = []
        
        # Find delegate calls within impl block
        delegate_pattern = re.compile(r'\b(add_sparse|sub_sparse|mul_sparse|div_sparse|matmul|spmv|spmm)\b')
        block_start = match.end()
        # Find next impl or end
        next_impl = content.find('impl<', block_start + 1)
        block_end = next_impl if next_impl > 0 else len(content)
        block = content[block_start:block_end]
        
        for delegate in delegate_pattern.finditer(block):
            implementations[key].append(delegate.group(1))
    
    return implementations


def verify_layer_separation(workspace: Path) -> List[str]:
    """Verify that operations are properly delegated through layers."""
    issues = []
    
    # Check tensor crate doesn't directly call backend kernels
    tensor_src = workspace / "tensor" / "src"
    backend_kernel_pattern = re.compile(r'\bcpu::(\w+_kernel|sparse_kernels::)')
    
    for rs_file in tensor_src.rglob("*.rs"):
        with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if backend_kernel_pattern.search(content):
            issues.append(f"Tensor crate directly references backend kernel in {rs_file.relative_to(workspace)}")
    
    # Check storage crate doesn't import tensor or autograd
    storage_src = workspace / "storage" / "src"
    forbidden_imports = re.compile(r'use (tensor|autograd)::')
    
    for rs_file in storage_src.rglob("*.rs"):
        with open(rs_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if forbidden_imports.search(content):
            issues.append(f"Storage crate has forbidden import in {rs_file.relative_to(workspace)}")
    
    return issues


def analyze_sparse_dispatch(workspace: Path) -> Dict[str, Dict[str, str]]:
    """Analyze sparse operation dispatch paths."""
    dispatch_paths = {}
    
    # Sparse arithmetic dispatch
    sparse_arith = workspace / "sparse" / "src" / "arithmetic"
    
    for op_file in ["add.rs", "sub.rs", "mul.rs", "div.rs", "matmul.rs"]:
        file_path = sparse_arith / op_file
        if not file_path.exists():
            continue
        
        op_name = op_file[:-3]  # Remove .rs
        dispatch_paths[op_name] = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for backend delegation
        if 'backend' in content.lower():
            dispatch_paths[op_name]['delegates_to_backend'] = 'yes'
        else:
            dispatch_paths[op_name]['delegates_to_backend'] = 'no (direct impl)'
        
        # Check which storage types are implemented
        for storage in ['CsrStorage', 'CscStorage', 'CooStorage']:
            if f'impl<' in content and storage in content:
                dispatch_paths[op_name][storage] = 'implemented'
            else:
                dispatch_paths[op_name][storage] = 'missing'
    
    return dispatch_paths


def generate_report(workspace: Path) -> str:
    """Generate comprehensive dispatch verification report."""
    lines = []
    lines.append("=" * 80)
    lines.append("Dispatch Chain Verification Report")
    lines.append("=" * 80)
    lines.append("")
    
    # Vector for checking dispatch architecture
    lines.append("## Tensor Dispatch Traits")
    lines.append("-" * 40)
    
    dispatch_file = workspace / "tensor" / "src" / "ops" / "arithmetic" / "dispatch.rs"
    dispatch_impls = check_dispatch_implementation(dispatch_file)
    
    for impl_name, delegates in sorted(dispatch_impls.items()):
        lines.append(f"\n{impl_name}:")
        if delegates:
            lines.append(f"  Delegates to: {', '.join(set(delegates))}")
        else:
            lines.append("  Direct implementation")
    
    lines.append("")
    
    # Check sparse dispatch
    lines.append("## Sparse Operation Dispatch Paths")
    lines.append("-" * 40)
    
    sparse_dispatch = analyze_sparse_dispatch(workspace)
    for op, info in sorted(sparse_dispatch.items()):
        lines.append(f"\n{op}:")
        for key, val in info.items():
            lines.append(f"  {key}: {val}")
    
    lines.append("")
    
    # Check layer separation
    lines.append("## Layer Separation Violations")
    lines.append("-" * 40)
    
    issues = verify_layer_separation(workspace)
    if issues:
        for issue in issues:
            lines.append(f"  ⚠ {issue}")
    else:
        lines.append("  ✓ No layer separation violations detected")
    
    lines.append("")
    
    # Backend trait methods
    lines.append("## Backend Trait Sparse Methods")
    lines.append("-" * 40)
    
    backend_lib = workspace / "backend" / "src" / "lib.rs"
    backend_traits = find_trait_methods(backend_lib)
    
    backend_trait = backend_traits.get("Backend", [])
    sparse_methods = [m for m in backend_trait if any(k in m for k in ['spmv', 'spmm', 'coo_', 'sparse'])]
    
    if sparse_methods:
        lines.append(f"\nSparse methods in Backend trait: {len(sparse_methods)}")
        for method in sparse_methods:
            lines.append(f"  - {method}")
    else:
        lines.append("\nNo sparse methods found in Backend trait (checking failed)")
        # Fallback - known methods
        lines.append("\nKnown sparse methods from manual inspection:")
        for m in ['spmv_csr', 'spmm_csr', 'coo_matmul_sparse', 'coo_matmul_dense', 'coo_add_sparse', 'coo_mul_sparse']:
            lines.append(f"  - {m}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    
    print("Verifying dispatch chain architecture...")
    print()
    
    report = generate_report(workspace)
    print(report)
    
    # Check for critical issues
    issues = verify_layer_separation(workspace)
    if issues:
        print("\n⚠ WARNING: Layer separation issues detected!")
        sys.exit(1)
    else:
        print("\n✓ Dispatch chain architecture verified successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
