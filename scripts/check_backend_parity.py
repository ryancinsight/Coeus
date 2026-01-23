#!/usr/bin/env python3
"""
Backend Parity Checker

This script checks for parity across different backends (CPU, GPU, TPU, NPU)
by comparing their file structures and identifying missing implementations.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json

class BackendParityChecker:
    def __init__(self, base_path: str = "backend/src"):
        self.base_path = Path(base_path)
        self.backends = ["cpu", "gpu", "tpu", "npu"]
        self.results = {}
        
    def get_file_structure(self, backend: str) -> Set[str]:
        """Get the file structure for a specific backend."""
        backend_path = self.base_path / backend
        if not backend_path.exists():
            return set()
        
        files = set()
        for root, dirs, filenames in os.walk(backend_path):
            for filename in filenames:
                if filename.endswith('.rs') and filename != 'mod.rs':
                    # Get relative path from backend root
                    rel_path = Path(root).relative_to(backend_path) / filename
                    files.add(str(rel_path))
        
        return files
    
    def check_parity(self) -> Dict[str, Dict[str, Set[str]]]:
        """Check parity across all backends."""
        backend_files = {}
        
        # Get file structure for each backend
        for backend in self.backends:
            backend_files[backend] = self.get_file_structure(backend)
        
        # Use CPU as reference (most complete implementation)
        reference_backend = "cpu"
        reference_files = backend_files.get(reference_backend, set())
        
        results = {
            "reference": reference_backend,
            "reference_files": reference_files,
            "missing": {},
            "extra": {},
            "common": set(),
        }
        
        # Find common files (implemented in all backends)
        if backend_files:
            results["common"] = set.intersection(*backend_files.values())
        
        # Find missing and extra files for each backend
        for backend in self.backends:
            if backend == reference_backend:
                continue
                
            backend_file_set = backend_files.get(backend, set())
            results["missing"][backend] = reference_files - backend_file_set
            results["extra"][backend] = backend_file_set - reference_files
        
        self.results = results
        return results
    
    def generate_parity_matrix(self) -> Dict[str, Dict[str, bool]]:
        """Generate a matrix showing which operations are implemented in which backends."""
        matrix = defaultdict(dict)
        
        # Get all unique files across all backends
        all_files = set()
        for backend in self.backends:
            all_files.update(self.get_file_structure(backend))
        
        # Check implementation status for each file in each backend
        for file_path in sorted(all_files):
            for backend in self.backends:
                backend_files = self.get_file_structure(backend)
                matrix[file_path][backend] = file_path in backend_files
        
        return dict(matrix)
    
    def print_summary(self):
        """Print a summary of parity checking results."""
        if not self.results:
            self.check_parity()
        
        print("=== Backend Parity Summary ===")
        print(f"Reference backend: {self.results['reference']}")
        print(f"Total reference files: {len(self.results['reference_files'])}")
        print(f"Common files (all backends): {len(self.results['common'])}")
        print()
        
        for backend in self.backends:
            if backend == self.results['reference']:
                continue
            
            missing = len(self.results['missing'].get(backend, set()))
            extra = len(self.results['extra'].get(backend, set()))
            total_files = len(self.get_file_structure(backend))
            
            print(f"{backend.upper()} Backend:")
            print(f"  Total files: {total_files}")
            print(f"  Missing from reference: {missing}")
            print(f"  Extra files: {extra}")
            
            if missing > 0:
                coverage = (len(self.results['reference_files']) - missing) / len(self.results['reference_files']) * 100
                print(f"  Coverage: {coverage:.1f}%")
            else:
                print(f"  Coverage: 100.0%")
            print()
    
    def print_detailed_report(self):
        """Print detailed missing and extra files."""
        if not self.results:
            self.check_parity()
        
        print("=== Detailed Parity Report ===")
        
        for backend in self.backends:
            if backend == self.results['reference']:
                continue
            
            print(f"\n{backend.upper()} Backend Details:")
            
            missing = self.results['missing'].get(backend, set())
            if missing:
                print(f"  Missing files ({len(missing)}):")
                for file_path in sorted(missing):
                    print(f"    - {file_path}")
            
            extra = self.results['extra'].get(backend, set())
            if extra:
                print(f"  Extra files ({len(extra)}):")
                for file_path in sorted(extra):
                    print(f"    + {file_path}")
            
            if not missing and not extra:
                print("  ✅ Perfect parity with reference backend")
    
    def generate_markdown_report(self, output_file: str):
        """Generate a markdown report of parity status."""
        if not self.results:
            self.check_parity()
        
        matrix = self.generate_parity_matrix()
        
        with open(output_file, 'w') as f:
            f.write("# Backend Parity Report\n\n")
            f.write(f"Generated for Coeus framework backend implementations.\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- **Reference Backend**: {self.results['reference']}\n")
            f.write(f"- **Total Reference Files**: {len(self.results['reference_files'])}\n")
            f.write(f"- **Common Files**: {len(self.results['common'])}\n\n")
            
            # Coverage table
            f.write("## Coverage by Backend\n\n")
            f.write("| Backend | Total Files | Missing | Extra | Coverage |\n")
            f.write("|---------|-------------|---------|-------|----------|\n")
            
            for backend in self.backends:
                total_files = len(self.get_file_structure(backend))
                if backend == self.results['reference']:
                    f.write(f"| {backend.upper()} | {total_files} | 0 | 0 | 100.0% (reference) |\n")
                else:
                    missing = len(self.results['missing'].get(backend, set()))
                    extra = len(self.results['extra'].get(backend, set()))
                    if len(self.results['reference_files']) > 0:
                        coverage = (len(self.results['reference_files']) - missing) / len(self.results['reference_files']) * 100
                    else:
                        coverage = 100.0
                    f.write(f"| {backend.upper()} | {total_files} | {missing} | {extra} | {coverage:.1f}% |\n")
            
            f.write("\n")
            
            # Parity matrix
            f.write("## Implementation Matrix\n\n")
            f.write("| Operation | CPU | GPU | TPU | NPU |\n")
            f.write("|-----------|-----|-----|-----|-----|\n")
            
            for file_path in sorted(matrix.keys()):
                # Clean up file path for display
                display_name = file_path.replace('.rs', '').replace('/', ' → ')
                f.write(f"| {display_name}")
                
                for backend in self.backends:
                    implemented = matrix[file_path].get(backend, False)
                    symbol = "✅" if implemented else "❌"
                    f.write(f" | {symbol}")
                
                f.write(" |\n")
            
            f.write("\n")
            
            # Detailed missing implementations
            f.write("## Missing Implementations\n\n")
            for backend in self.backends:
                if backend == self.results['reference']:
                    continue
                
                missing = self.results['missing'].get(backend, set())
                if missing:
                    f.write(f"### {backend.upper()} Backend\n\n")
                    for file_path in sorted(missing):
                        f.write(f"- `{file_path}`\n")
                    f.write("\n")
            
            # Priority recommendations
            f.write("## Priority Recommendations\n\n")
            
            # Find operations missing in multiple backends
            missing_counts = defaultdict(int)
            for backend in self.backends:
                if backend == self.results['reference']:
                    continue
                for file_path in self.results['missing'].get(backend, set()):
                    missing_counts[file_path] += 1
            
            high_priority = [f for f, count in missing_counts.items() if count >= 2]
            medium_priority = [f for f, count in missing_counts.items() if count == 1]
            
            if high_priority:
                f.write("### High Priority (missing in multiple backends)\n\n")
                for file_path in sorted(high_priority):
                    f.write(f"- `{file_path}`\n")
                f.write("\n")
            
            if medium_priority:
                f.write("### Medium Priority (missing in one backend)\n\n")
                for file_path in sorted(medium_priority):
                    f.write(f"- `{file_path}`\n")
                f.write("\n")
    
    def generate_json_report(self, output_file: str):
        """Generate a JSON report for programmatic use."""
        if not self.results:
            self.check_parity()
        
        matrix = self.generate_parity_matrix()
        
        report = {
            "summary": {
                "reference_backend": self.results['reference'],
                "total_reference_files": len(self.results['reference_files']),
                "common_files": len(self.results['common']),
                "backends": self.backends,
            },
            "coverage": {},
            "matrix": matrix,
            "missing": {k: list(v) for k, v in self.results['missing'].items()},
            "extra": {k: list(v) for k, v in self.results['extra'].items()},
        }
        
        # Calculate coverage for each backend
        for backend in self.backends:
            total_files = len(self.get_file_structure(backend))
            if backend == self.results['reference']:
                report["coverage"][backend] = {
                    "total_files": total_files,
                    "missing": 0,
                    "extra": 0,
                    "coverage_percent": 100.0,
                }
            else:
                missing = len(self.results['missing'].get(backend, set()))
                extra = len(self.results['extra'].get(backend, set()))
                if len(self.results['reference_files']) > 0:
                    coverage = (len(self.results['reference_files']) - missing) / len(self.results['reference_files']) * 100
                else:
                    coverage = 100.0
                
                report["coverage"][backend] = {
                    "total_files": total_files,
                    "missing": missing,
                    "extra": extra,
                    "coverage_percent": coverage,
                }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Check backend parity across Coeus implementations")
    parser.add_argument("--base-path", default="backend/src", help="Base path to backend implementations")
    parser.add_argument("--output", help="Output file for markdown report")
    parser.add_argument("--json", help="Output file for JSON report")
    parser.add_argument("--detailed", action="store_true", help="Show detailed missing/extra files")
    parser.add_argument("--quiet", action="store_true", help="Only show summary")
    
    args = parser.parse_args()
    
    # Check if base path exists
    if not Path(args.base_path).exists():
        print(f"Error: Base path '{args.base_path}' does not exist")
        print("Make sure you're running this script from the Coeus project root")
        sys.exit(1)
    
    checker = BackendParityChecker(args.base_path)
    checker.check_parity()
    
    if not args.quiet:
        checker.print_summary()
        
        if args.detailed:
            checker.print_detailed_report()
    
    if args.output:
        checker.generate_markdown_report(args.output)
        print(f"Markdown report generated: {args.output}")
    
    if args.json:
        checker.generate_json_report(args.json)
        print(f"JSON report generated: {args.json}")
    
    # Exit with error code if there are missing implementations
    missing_count = sum(len(missing) for missing in checker.results['missing'].values())
    if missing_count > 0:
        print(f"\n⚠️  Found {missing_count} missing implementations across backends")
        sys.exit(1)
    else:
        print("\n✅ All backends have complete parity!")

if __name__ == "__main__":
    main()