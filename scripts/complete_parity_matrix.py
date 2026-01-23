#!/usr/bin/env python3
"""
Complete Parity Matrix Generator

Generates comprehensive coverage matrix for:
- Operations × Storage types (Dense, CSR, CSC, COO, Quantized)
- Operations × Backend types (CPU, GPU, TPU, NPU)
- Operations × Dtypes (Float32, Float64, Int8, Int16, Int32, Int64, etc.)

Usage:
    python scripts/complete_parity_matrix.py [--json] [--csv] [--markdown]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class OperationCoverage:
    """Tracks operation coverage across dimensions."""
    name: str
    storage_types: Dict[str, bool] = field(default_factory=dict)
    backends: Dict[str, bool] = field(default_factory=dict)
    dtypes: Dict[str, bool] = field(default_factory=dict)


def get_rust_files(path: Path, pattern: str = "*.rs") -> Set[str]:
    """Get all Rust file basenames (without .rs) in a directory."""
    if not path.exists():
        return set()
    return {f.stem for f in path.glob(pattern) if f.name != "mod.rs"}


def scan_dense_operations(workspace: Path) -> Dict[str, Set[str]]:
    """Scan dense crate for operations by category."""
    dense_src = workspace / "dense" / "src"
    categories = ["arithmetic", "layout", "creation", "linear_algebra"]
    
    ops = {}
    for cat in categories:
        cat_path = dense_src / cat
        ops[cat] = get_rust_files(cat_path)
    
    return ops


def scan_sparse_operations(workspace: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Scan sparse crate for operations by format and category."""
    sparse_src = workspace / "sparse" / "src"
    formats = ["csr", "csc", "coo"]
    categories = ["arithmetic", "conversion", "indexing"]
    
    ops = {}
    
    # Top-level sparse operations
    top_categories = ["arithmetic", "layout", "creation"]
    ops["common"] = {}
    for cat in top_categories:
        cat_path = sparse_src / cat
        ops["common"][cat] = get_rust_files(cat_path)
    
    # Per-format operations
    formats_path = sparse_src / "formats"
    for fmt in formats:
        ops[fmt] = {}
        fmt_path = formats_path / fmt
        for cat in categories:
            cat_path = fmt_path / cat
            ops[fmt][cat] = get_rust_files(cat_path)
    
    return ops


def scan_backend_operations(workspace: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Scan backend crate for operations by backend and category."""
    backend_src = workspace / "backend" / "src"
    backends = ["cpu", "gpu", "tpu", "npu"]
    categories = ["arithmetic", "linear_algebra", "activation", "reduction"]
    
    ops = {}
    for backend in backends:
        ops[backend] = {}
        backend_path = backend_src / backend
        for cat in categories:
            cat_path = backend_path / cat
            ops[backend][cat] = get_rust_files(cat_path)
    
    return ops


def scan_tensor_operations(workspace: Path) -> Dict[str, Set[str]]:
    """Scan tensor crate for operations by category."""
    tensor_ops = workspace / "tensor" / "src" / "ops"
    categories = [
        "activation", "arithmetic", "classification", "layout",
        "linalg", "math", "reduction", "rnn"
    ]
    
    ops = {}
    for cat in categories:
        cat_path = tensor_ops / cat
        ops[cat] = get_rust_files(cat_path)
    
    return ops


def scan_dtypes(workspace: Path) -> Set[str]:
    """Scan dtype crate for supported types."""
    dtype_src = workspace / "dtype" / "src"
    
    # Known dtype files
    dtype_files = ["float", "half", "int", "complex"]
    return set(dtype_files)


def compute_parity_matrix(
    dense_ops: Dict[str, Set[str]],
    sparse_ops: Dict[str, Dict[str, Set[str]]],
    backend_ops: Dict[str, Dict[str, Set[str]]],
    tensor_ops: Dict[str, Set[str]]
) -> Tuple[Dict, Dict, Dict]:
    """Compute parity matrices across dimensions."""
    
    # Flatten all operations
    all_ops = set()
    for cat_ops in dense_ops.values():
        all_ops.update(cat_ops)
    for fmt_ops in sparse_ops.values():
        for cat_ops in fmt_ops.values():
            all_ops.update(cat_ops)
    for cat_ops in tensor_ops.values():
        all_ops.update(cat_ops)
    
    # Storage parity matrix
    storage_matrix = {}
    storage_types = ["dense", "csr", "csc", "coo"]
    for op in sorted(all_ops):
        storage_matrix[op] = {}
        # Dense
        storage_matrix[op]["dense"] = any(op in ops for ops in dense_ops.values())
        # Sparse formats
        for fmt in ["csr", "csc", "coo"]:
            if fmt in sparse_ops:
                storage_matrix[op][fmt] = any(op in ops for ops in sparse_ops[fmt].values())
            else:
                storage_matrix[op][fmt] = False
        # Check common sparse
        if "common" in sparse_ops:
            for fmt in ["csr", "csc", "coo"]:
                if any(op in ops for ops in sparse_ops["common"].values()):
                    storage_matrix[op][fmt] = True
    
    # Backend parity matrix
    backend_matrix = {}
    backends = ["cpu", "gpu", "tpu", "npu"]
    for backend in backends:
        backend_all_ops = set()
        if backend in backend_ops:
            for cat_ops in backend_ops[backend].values():
                backend_all_ops.update(cat_ops)
        for op in sorted(all_ops):
            if op not in backend_matrix:
                backend_matrix[op] = {}
            backend_matrix[op][backend] = op in backend_all_ops
    
    # Category summary
    category_summary = {}
    for cat, ops in tensor_ops.items():
        category_summary[cat] = len(ops)
    
    return storage_matrix, backend_matrix, category_summary


def generate_markdown_report(
    storage_matrix: Dict,
    backend_matrix: Dict,
    category_summary: Dict,
    dense_ops: Dict,
    sparse_ops: Dict,
    backend_ops: Dict
) -> str:
    """Generate a markdown report of parity coverage."""
    lines = []
    lines.append("# Coeus Operations Parity Matrix")
    lines.append("")
    lines.append(f"*Generated by `complete_parity_matrix.py`*")
    lines.append("")
    
    # Category summary
    lines.append("## Operation Categories Summary")
    lines.append("")
    lines.append("| Category | Operations |")
    lines.append("|----------|------------|")
    for cat, count in sorted(category_summary.items()):
        lines.append(f"| {cat} | {count} |")
    lines.append("")
    
    # Storage coverage
    lines.append("## Storage Type Coverage")
    lines.append("")
    lines.append("| Operation | Dense | CSR | CSC | COO |")
    lines.append("|-----------|-------|-----|-----|-----|")
    for op, coverage in sorted(storage_matrix.items()):
        row = f"| {op} |"
        for st in ["dense", "csr", "csc", "coo"]:
            status = "✅" if coverage.get(st, False) else "❌"
            row += f" {status} |"
        lines.append(row)
    lines.append("")
    
    # Backend coverage
    lines.append("## Backend Coverage")
    lines.append("")
    lines.append("| Operation | CPU | GPU | TPU | NPU |")
    lines.append("|-----------|-----|-----|-----|-----|")
    for op, coverage in sorted(backend_matrix.items()):
        row = f"| {op} |"
        for backend in ["cpu", "gpu", "tpu", "npu"]:
            status = "✅" if coverage.get(backend, False) else "❌"
            row += f" {status} |"
        lines.append(row)
    lines.append("")
    
    # Backend parity summary
    lines.append("## Backend Parity Summary")
    lines.append("")
    for backend in ["cpu", "gpu", "tpu", "npu"]:
        if backend in backend_ops:
            total = sum(len(ops) for ops in backend_ops[backend].values())
            lines.append(f"- **{backend.upper()}**: {total} operations")
    lines.append("")
    
    # Coverage gaps
    lines.append("## Coverage Gaps")
    lines.append("")
    
    # Find operations missing from sparse
    sparse_missing = []
    for op, coverage in storage_matrix.items():
        if coverage.get("dense") and not (coverage.get("csr") or coverage.get("csc") or coverage.get("coo")):
            sparse_missing.append(op)
    
    if sparse_missing:
        lines.append("### Operations in Dense but not Sparse")
        lines.append("")
        for op in sorted(sparse_missing):
            lines.append(f"- `{op}`")
        lines.append("")
    else:
        lines.append("✅ No dense-only operations found")
        lines.append("")
    
    # Find backend gaps
    cpu_ops = set()
    if "cpu" in backend_ops:
        for ops in backend_ops["cpu"].values():
            cpu_ops.update(ops)
    
    for backend in ["gpu", "tpu", "npu"]:
        if backend in backend_ops:
            backend_all = set()
            for ops in backend_ops[backend].values():
                backend_all.update(ops)
            missing = cpu_ops - backend_all
            if missing:
                lines.append(f"### Operations in CPU but not {backend.upper()}")
                lines.append("")
                for op in sorted(missing):
                    lines.append(f"- `{op}`")
                lines.append("")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate parity matrix for Coeus operations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown (default)")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    args = parser.parse_args()
    
    # Default to markdown
    if not (args.json or args.csv or args.markdown):
        args.markdown = True
    
    # Get workspace root
    script_dir = Path(__file__).parent
    workspace = script_dir.parent
    
    print("Scanning operations...", file=sys.stderr)
    
    # Scan all operations
    dense_ops = scan_dense_operations(workspace)
    sparse_ops = scan_sparse_operations(workspace)
    backend_ops = scan_backend_operations(workspace)
    tensor_ops = scan_tensor_operations(workspace)
    dtypes = scan_dtypes(workspace)
    
    # Compute matrices
    storage_matrix, backend_matrix, category_summary = compute_parity_matrix(
        dense_ops, sparse_ops, backend_ops, tensor_ops
    )
    
    # Generate output
    if args.json:
        output = json.dumps({
            "storage_matrix": storage_matrix,
            "backend_matrix": backend_matrix,
            "category_summary": category_summary,
            "dense_operations": {k: list(v) for k, v in dense_ops.items()},
            "sparse_operations": {k: {k2: list(v2) for k2, v2 in v.items()} for k, v in sparse_ops.items()},
            "backend_operations": {k: {k2: list(v2) for k2, v2 in v.items()} for k, v in backend_ops.items()},
        }, indent=2)
    elif args.csv:
        lines = ["operation,dense,csr,csc,coo,cpu,gpu,tpu,npu"]
        for op in sorted(storage_matrix.keys()):
            row = [op]
            for st in ["dense", "csr", "csc", "coo"]:
                row.append("1" if storage_matrix[op].get(st) else "0")
            for backend in ["cpu", "gpu", "tpu", "npu"]:
                row.append("1" if backend_matrix.get(op, {}).get(backend) else "0")
            lines.append(",".join(row))
        output = "\n".join(lines)
    else:
        output = generate_markdown_report(
            storage_matrix, backend_matrix, category_summary,
            dense_ops, sparse_ops, backend_ops
        )
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
