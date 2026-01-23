#!/usr/bin/env python3
"""
Storage & Dtype Gap Audit Script

This script analyzes the Coeus codebase to identify:
1. Which dtypes have complete trait implementations
2. Which storage types support which operations
3. Which autograd functions exist and what they support
4. Sparse operation coverage

Run from the repository root:
    python scripts/audit_storage_dtype_gaps.py
"""

import os
import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from collections import defaultdict


@dataclass
class DtypeInfo:
    """Information about a dtype implementation"""
    name: str
    has_datatype_trait: bool = False
    has_float_ext: bool = False
    has_int_ext: bool = False
    has_complex_ext: bool = False
    feature_gated: Optional[str] = None


@dataclass
class StorageInfo:
    """Information about a storage type implementation"""
    name: str
    traits_implemented: Set[str] = field(default_factory=set)
    sparse_ops: Set[str] = field(default_factory=set)


@dataclass
class AutogradFunctionInfo:
    """Information about an autograd function"""
    name: str
    file: str
    line: int
    supports_sparse: bool = False
    converts_to_dense: bool = True


class CoeusAuditor:
    """Audits Coeus crates for storage/dtype/autograd coverage"""
    
    def __init__(self, root_path: str = "."):
        self.root = Path(root_path)
        self.dtypes: Dict[str, DtypeInfo] = {}
        self.storages: Dict[str, StorageInfo] = {}
        self.autograd_functions: List[AutogradFunctionInfo] = []
        self.sparse_traits: Dict[str, Set[str]] = defaultdict(set)
    
    def audit_all(self):
        """Run complete audit"""
        print("=" * 60)
        print("COEUS STORAGE & DTYPE GAP AUDIT")
        print("=" * 60)
        
        self.audit_dtypes()
        self.audit_storages()
        self.audit_autograd()
        self.audit_sparse_ops()
        self.generate_report()
    
    def audit_dtypes(self):
        """Audit dtype crate for type implementations"""
        print("\n[1/4] Auditing dtype crate...")
        
        # Expected dtypes from Dtype enum
        expected_dtypes = [
            "Half", "BFloat16", "Float32", "Float64",
            "Int8", "Int16", "Int32", "Int64",
            "UInt8", "UInt16", "UInt32", "UInt64",
            "Complex32", "Complex64",
            "QInt4", "QUInt4", "QInt8", "QUInt8"
        ]
        
        for name in expected_dtypes:
            self.dtypes[name] = DtypeInfo(name=name)
        
        # Check float.rs for Float32, Float64
        float_rs = self.root / "dtype" / "src" / "float.rs"
        if float_rs.exists():
            content = float_rs.read_text(encoding='utf-8', errors='ignore')
            
            if "pub struct Float32" in content:
                self.dtypes["Float32"].has_datatype_trait = True
                if "impl FloatExt for Float32" in content or "impl Float for Float32" in content:
                    self.dtypes["Float32"].has_float_ext = True
            
            if "pub struct Float64" in content:
                self.dtypes["Float64"].has_datatype_trait = True
                if "impl FloatExt for Float64" in content or "impl Float for Float64" in content:
                    self.dtypes["Float64"].has_float_ext = True
        
        # Check int.rs for integer types
        int_rs = self.root / "dtype" / "src" / "int.rs"
        if int_rs.exists():
            content = int_rs.read_text(encoding='utf-8', errors='ignore')
            
            for int_type in ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"]:
                if f"pub struct {int_type}" in content or f"impl_int_dtype!({int_type}" in content:
                    self.dtypes[int_type].has_datatype_trait = True
                    self.dtypes[int_type].has_int_ext = True
        
        # Check complex.rs
        complex_rs = self.root / "dtype" / "src" / "complex.rs"
        if complex_rs.exists():
            content = complex_rs.read_text(encoding='utf-8', errors='ignore')
            for complex_type in ["Complex32", "Complex64"]:
                if complex_type in content:
                    self.dtypes[complex_type].has_datatype_trait = True
                    self.dtypes[complex_type].has_complex_ext = True
                    self.dtypes[complex_type].feature_gated = "complex"
        
        # Check quantized.rs
        quantized_rs = self.root / "dtype" / "src" / "quantized.rs"
        if quantized_rs.exists():
            content = quantized_rs.read_text(encoding='utf-8', errors='ignore')
            for q_type in ["QInt4", "QUInt4", "QInt8", "QUInt8"]:
                if q_type in content:
                    self.dtypes[q_type].has_datatype_trait = True
                    self.dtypes[q_type].feature_gated = "quantized"
        
        # Half/BFloat16 - check if actual implementations exist
        half_rs = self.root / "dtype" / "src" / "half.rs"
        if half_rs.exists():
            content = half_rs.read_text(encoding='utf-8', errors='ignore')
            if "pub struct Half" in content:
                self.dtypes["Half"].has_datatype_trait = True
            if "pub struct BFloat16" in content:
                self.dtypes["BFloat16"].has_datatype_trait = True
    
    def audit_storages(self):
        """Audit storage crate for trait implementations"""
        print("[2/4] Auditing storage crate...")
        
        storage_types = ["DenseStorage", "CsrStorage", "CscStorage", "CooStorage"]
        traits_to_check = [
            "Storage", "StorageFromVec", "StorageToDense", 
            "StorageOps", "MatMulOps", "ArithmeticOps", "ReductionOps"
        ]
        
        for name in storage_types:
            self.storages[name] = StorageInfo(name=name)
        
        # Check storage source files
        for src_file in (self.root / "storage" / "src").glob("*.rs"):
            content = src_file.read_text(encoding='utf-8', errors='ignore')
            
            for storage in storage_types:
                for trait in traits_to_check:
                    # Look for impl Trait for Storage patterns
                    patterns = [
                        f"impl<T[^>]*> {trait}<T> for {storage}",
                        f"impl<T: DataType> {trait}<T> for {storage}",
                        f"impl {trait}<T> for {storage}",
                    ]
                    for pattern in patterns:
                        if re.search(pattern, content):
                            self.storages[storage].traits_implemented.add(trait)
                            break
    
    def audit_autograd(self):
        """Audit autograd crate for function implementations"""
        print("[3/4] Auditing autograd crate...")
        
        functions_rs = self.root / "autograd" / "src" / "functions.rs"
        if not functions_rs.exists():
            print("  WARNING: functions.rs not found")
            return
        
        content = functions_rs.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        # Find all struct definitions that end with "Function"
        struct_pattern = re.compile(r'pub struct (\w+Function)')
        
        for i, line in enumerate(lines):
            match = struct_pattern.search(line)
            if match:
                func_name = match.group(1)
                
                # Check if this function converts to dense
                # Look for StorageToDense in the trait bounds or to_dense calls
                converts_to_dense = True  # Assume true, check if native sparse
                
                # Look ahead for the impl block
                for j in range(i, min(i+100, len(lines))):
                    if "StorageToDense" in lines[j]:
                        converts_to_dense = True
                        break
                
                self.autograd_functions.append(AutogradFunctionInfo(
                    name=func_name,
                    file="autograd/src/functions.rs",
                    line=i+1,
                    converts_to_dense=converts_to_dense
                ))
    
    def audit_sparse_ops(self):
        """Audit sparse crate for operation implementations"""
        print("[4/4] Auditing sparse crate...")
        
        sparse_traits = [
            "SparseMatMul", "SparseAdd", "SparseSub", "SparseMul", "SparseDiv",
            "SparseElementWise", "SparseReduce", "SparseTranspose", "SparseReshape",
            "SparseOptimizerOps"
        ]
        
        storage_types = ["CsrStorage", "CscStorage", "CooStorage"]
        
        arithmetic_rs = self.root / "sparse" / "src" / "cpu" / "arithmetic.rs"
        if arithmetic_rs.exists():
            content = arithmetic_rs.read_text(encoding='utf-8', errors='ignore')
            
            for storage in storage_types:
                for trait in sparse_traits:
                    pattern = f"impl<[^>]*> {trait}<[^>]*> for {storage}"
                    if re.search(pattern, content):
                        self.sparse_traits[storage].add(trait)
    
    def generate_report(self):
        """Generate the audit report"""
        print("\n" + "=" * 60)
        print("AUDIT RESULTS")
        print("=" * 60)
        
        # Dtype summary
        print("\n## DTYPE COVERAGE")
        print("-" * 40)
        print(f"{'Dtype':<12} {'DataType':<10} {'FloatExt':<10} {'IntExt':<10} {'Feature':<10}")
        print("-" * 40)
        
        missing_dtypes = []
        for name, info in sorted(self.dtypes.items()):
            dt = "✓" if info.has_datatype_trait else "✗"
            fe = "✓" if info.has_float_ext else "-"
            ie = "✓" if info.has_int_ext else "-"
            feat = info.feature_gated or "-"
            print(f"{name:<12} {dt:<10} {fe:<10} {ie:<10} {feat:<10}")
            if not info.has_datatype_trait:
                missing_dtypes.append(name)
        
        if missing_dtypes:
            print(f"\n  MISSING: {', '.join(missing_dtypes)}")
        
        # Storage summary
        print("\n## STORAGE TRAIT COVERAGE")
        print("-" * 40)
        
        for name, info in sorted(self.storages.items()):
            traits = ", ".join(sorted(info.traits_implemented)) or "None"
            print(f"{name}: {traits}")
        
        # Autograd summary
        print("\n## AUTOGRAD FUNCTIONS")
        print("-" * 40)
        print(f"Total functions: {len(self.autograd_functions)}")
        
        dense_converts = sum(1 for f in self.autograd_functions if f.converts_to_dense)
        print(f"Convert to dense: {dense_converts}")
        print(f"Native sparse: {len(self.autograd_functions) - dense_converts}")
        
        print("\nFunctions found:")
        for func in sorted(self.autograd_functions, key=lambda x: x.name):
            sparse = "→dense" if func.converts_to_dense else "native"
            print(f"  - {func.name} ({sparse})")
        
        # Sparse ops summary
        print("\n## SPARSE OPERATION COVERAGE")
        print("-" * 40)
        
        for storage, traits in sorted(self.sparse_traits.items()):
            print(f"{storage}: {len(traits)} traits")
            for trait in sorted(traits):
                print(f"  - {trait}")
        
        # Generate JSON report
        report = {
            "dtypes": {
                name: {
                    "has_datatype_trait": info.has_datatype_trait,
                    "has_float_ext": info.has_float_ext,
                    "has_int_ext": info.has_int_ext,
                    "feature_gated": info.feature_gated
                }
                for name, info in self.dtypes.items()
            },
            "storages": {
                name: list(info.traits_implemented)
                for name, info in self.storages.items()
            },
            "autograd_functions": [
                {"name": f.name, "converts_to_dense": f.converts_to_dense}
                for f in self.autograd_functions
            ],
            "sparse_traits": {
                storage: list(traits)
                for storage, traits in self.sparse_traits.items()
            }
        }
        
        report_path = self.root / "storage_dtype_audit_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ JSON report saved: {report_path}")
        print("\n" + "=" * 60)


def main():
    # Find repo root (look for Cargo.toml)
    cwd = Path.cwd()
    root = cwd
    
    for parent in [cwd] + list(cwd.parents):
        if (parent / "Cargo.toml").exists() and (parent / "dtype").is_dir():
            root = parent
            break
    
    print(f"Auditing from: {root}")
    auditor = CoeusAuditor(root)
    auditor.audit_all()


if __name__ == "__main__":
    main()
