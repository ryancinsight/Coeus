#!/usr/bin/env python3
"""
Parity Report Generator

This script generates comprehensive parity reports across all Coeus domains,
including backend implementations, operation coverage, and implementation status.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import json
from datetime import datetime

class ParityReportGenerator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.domains = {
            "backend": "backend/src",
            "nn": "nn/src",
            "dense": "dense/src", 
            "sparse": "sparse/src",
            "quantization": "quantization/src",
            "storage": "storage/src",
        }
        self.backends = ["cpu", "gpu", "tpu", "npu"]
        
    def get_domain_structure(self, domain: str) -> Dict[str, Set[str]]:
        """Get the file structure for a domain."""
        domain_path = self.project_root / self.domains[domain]
        if not domain_path.exists():
            return {}
        
        if domain == "backend":
            # Special handling for backend domain
            structure = {}
            for backend in self.backends:
                backend_path = domain_path / backend
                if backend_path.exists():
                    files = set()
                    for root, dirs, filenames in os.walk(backend_path):
                        for filename in filenames:
                            if filename.endswith('.rs') and filename != 'mod.rs':
                                rel_path = Path(root).relative_to(backend_path) / filename
                                files.add(str(rel_path))
                    structure[backend] = files
            return structure
        else:
            # Regular domain handling
            files = set()
            for root, dirs, filenames in os.walk(domain_path):
                for filename in filenames:
                    if filename.endswith('.rs') and filename != 'mod.rs':
                        rel_path = Path(root).relative_to(domain_path) / filename
                        files.add(str(rel_path))
            return {"main": files}
    
    def analyze_operation_coverage(self) -> Dict[str, Dict[str, bool]]:
        """Analyze which operations are implemented across domains."""
        operations = defaultdict(dict)
        
        # Get all unique operation names across domains
        all_operations = set()
        domain_operations = {}
        
        for domain in self.domains:
            domain_structure = self.get_domain_structure(domain)
            domain_operations[domain] = domain_structure
            
            if domain == "backend":
                # For backend, collect operations from CPU (reference)
                cpu_ops = domain_structure.get("cpu", set())
                for op_path in cpu_ops:
                    op_name = Path(op_path).stem
                    all_operations.add(op_name)
            else:
                # For other domains, collect all operations
                main_ops = domain_structure.get("main", set())
                for op_path in main_ops:
                    op_name = Path(op_path).stem
                    all_operations.add(op_name)
        
        # Check implementation status for each operation in each domain
        for op_name in sorted(all_operations):
            for domain in self.domains:
                domain_structure = domain_operations[domain]
                
                if domain == "backend":
                    # Check across all backends
                    for backend in self.backends:
                        backend_ops = domain_structure.get(backend, set())
                        has_op = any(Path(op_path).stem == op_name for op_path in backend_ops)
                        operations[op_name][f"{domain}_{backend}"] = has_op
                else:
                    # Check in main domain
                    main_ops = domain_structure.get("main", set())
                    has_op = any(Path(op_path).stem == op_name for op_path in main_ops)
                    operations[op_name][domain] = has_op
        
        return dict(operations)
    
    def calculate_domain_statistics(self) -> Dict[str, Dict]:
        """Calculate statistics for each domain."""
        stats = {}
        
        for domain in self.domains:
            domain_structure = self.get_domain_structure(domain)
            
            if domain == "backend":
                # Backend-specific statistics
                total_files = {}
                for backend in self.backends:
                    backend_files = domain_structure.get(backend, set())
                    total_files[backend] = len(backend_files)
                
                # Calculate parity
                cpu_files = domain_structure.get("cpu", set())
                parity = {}
                for backend in self.backends:
                    if backend == "cpu":
                        parity[backend] = 100.0
                    else:
                        backend_files = domain_structure.get(backend, set())
                        if len(cpu_files) > 0:
                            common = len(cpu_files.intersection(backend_files))
                            parity[backend] = (common / len(cpu_files)) * 100
                        else:
                            parity[backend] = 100.0
                
                stats[domain] = {
                    "type": "backend",
                    "total_files": total_files,
                    "parity": parity,
                    "reference": "cpu",
                }
            else:
                # Regular domain statistics
                main_files = domain_structure.get("main", set())
                
                # Categorize files by type
                categories = defaultdict(int)
                for file_path in main_files:
                    path_parts = Path(file_path).parts
                    if len(path_parts) > 1:
                        category = path_parts[0]
                        categories[category] += 1
                    else:
                        categories["root"] += 1
                
                stats[domain] = {
                    "type": "regular",
                    "total_files": len(main_files),
                    "categories": dict(categories),
                }
        
        return stats
    
    def generate_comprehensive_report(self, output_file: str):
        """Generate a comprehensive parity report."""
        operation_coverage = self.analyze_operation_coverage()
        domain_stats = self.calculate_domain_statistics()
        
        with open(output_file, 'w') as f:
            f.write("# Coeus Framework Parity Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            total_operations = len(operation_coverage)
            f.write(f"- **Total Operations Analyzed**: {total_operations}\n")
            
            # Backend parity summary
            backend_stats = domain_stats.get("backend", {})
            if backend_stats:
                f.write("- **Backend Implementation Status**:\n")
                for backend, parity in backend_stats.get("parity", {}).items():
                    f.write(f"  - {backend.upper()}: {parity:.1f}% parity with CPU\n")
            
            # Domain coverage summary
            f.write("- **Domain Coverage**:\n")
            for domain, stats in domain_stats.items():
                if stats["type"] == "regular":
                    f.write(f"  - {domain.capitalize()}: {stats['total_files']} implementations\n")
            
            f.write("\n")
            
            # Domain Statistics
            f.write("## Domain Statistics\n\n")
            
            for domain, stats in domain_stats.items():
                f.write(f"### {domain.capitalize()} Domain\n\n")
                
                if stats["type"] == "backend":
                    f.write("| Backend | Files | Parity with CPU |\n")
                    f.write("|---------|-------|----------------|\n")
                    
                    for backend in self.backends:
                        file_count = stats["total_files"].get(backend, 0)
                        parity = stats["parity"].get(backend, 0)
                        f.write(f"| {backend.upper()} | {file_count} | {parity:.1f}% |\n")
                    
                    f.write("\n")
                else:
                    f.write(f"- **Total Files**: {stats['total_files']}\n")
                    f.write("- **Categories**:\n")
                    for category, count in stats["categories"].items():
                        f.write(f"  - {category}: {count} files\n")
                    f.write("\n")
            
            # Operation Coverage Matrix
            f.write("## Operation Coverage Matrix\n\n")
            f.write("This matrix shows which operations are implemented across different domains.\n\n")
            
            # Create header
            domains_flat = []
            for domain in self.domains:
                if domain == "backend":
                    for backend in self.backends:
                        domains_flat.append(f"{domain}_{backend}")
                else:
                    domains_flat.append(domain)
            
            f.write("| Operation")
            for domain in domains_flat:
                f.write(f" | {domain.replace('_', ' ').title()}")
            f.write(" |\n")
            
            f.write("|-----------|")
            for _ in domains_flat:
                f.write("-----|")
            f.write("\n")
            
            # Add operation rows
            for op_name in sorted(operation_coverage.keys()):
                f.write(f"| {op_name}")
                for domain in domains_flat:
                    implemented = operation_coverage[op_name].get(domain, False)
                    symbol = "✅" if implemented else "❌"
                    f.write(f" | {symbol}")
                f.write(" |\n")
            
            f.write("\n")
            
            # Missing Implementations
            f.write("## Missing Implementations\n\n")
            
            missing_by_domain = defaultdict(list)
            for op_name, implementations in operation_coverage.items():
                for domain, implemented in implementations.items():
                    if not implemented:
                        missing_by_domain[domain].append(op_name)
            
            for domain in sorted(missing_by_domain.keys()):
                if missing_by_domain[domain]:
                    f.write(f"### {domain.replace('_', ' ').title()}\n\n")
                    for op_name in sorted(missing_by_domain[domain]):
                        f.write(f"- `{op_name}`\n")
                    f.write("\n")
            
            # Implementation Priorities
            f.write("## Implementation Priorities\n\n")
            
            # Count missing implementations per operation
            missing_counts = defaultdict(int)
            for op_name, implementations in operation_coverage.items():
                for domain, implemented in implementations.items():
                    if not implemented:
                        missing_counts[op_name] += 1
            
            # Categorize by priority
            high_priority = [(op, count) for op, count in missing_counts.items() if count >= 4]
            medium_priority = [(op, count) for op, count in missing_counts.items() if 2 <= count < 4]
            low_priority = [(op, count) for op, count in missing_counts.items() if count == 1]
            
            if high_priority:
                f.write("### High Priority (missing in 4+ domains)\n\n")
                for op_name, count in sorted(high_priority, key=lambda x: x[1], reverse=True):
                    f.write(f"- `{op_name}` (missing in {count} domains)\n")
                f.write("\n")
            
            if medium_priority:
                f.write("### Medium Priority (missing in 2-3 domains)\n\n")
                for op_name, count in sorted(medium_priority, key=lambda x: x[1], reverse=True):
                    f.write(f"- `{op_name}` (missing in {count} domains)\n")
                f.write("\n")
            
            if low_priority:
                f.write("### Low Priority (missing in 1 domain)\n\n")
                for op_name, count in sorted(low_priority):
                    f.write(f"- `{op_name}`\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Focus on High Priority Operations**: Implement operations missing in multiple domains first\n")
            f.write("2. **Backend Parity**: Prioritize GPU backend implementation to improve parity\n")
            f.write("3. **Domain Completeness**: Ensure each domain has comprehensive coverage of its core operations\n")
            f.write("4. **Testing**: Add property-based tests for newly implemented operations\n")
            f.write("5. **Documentation**: Update documentation as implementations are added\n\n")
            
            # Footer
            f.write("---\n\n")
            f.write("*This report was generated automatically by the Coeus parity tracking system.*\n")
    
    def generate_json_summary(self, output_file: str):
        """Generate a JSON summary for programmatic use."""
        operation_coverage = self.analyze_operation_coverage()
        domain_stats = self.calculate_domain_statistics()
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_operations": len(operation_coverage),
            "domains": list(self.domains.keys()),
            "backends": self.backends,
            "domain_statistics": domain_stats,
            "operation_coverage": operation_coverage,
        }
        
        # Calculate overall statistics
        total_implementations = 0
        total_possible = 0
        
        for op_name, implementations in operation_coverage.items():
            for domain, implemented in implementations.items():
                total_possible += 1
                if implemented:
                    total_implementations += 1
        
        summary["overall_coverage"] = {
            "implemented": total_implementations,
            "total_possible": total_possible,
            "percentage": (total_implementations / total_possible * 100) if total_possible > 0 else 0,
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive parity report for Coeus framework")
    parser.add_argument("--project-root", default=".", help="Root directory of Coeus project")
    parser.add_argument("--output", default="parity_report.md", help="Output file for markdown report")
    parser.add_argument("--json", help="Output file for JSON summary")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if project root exists
    if not Path(args.project_root).exists():
        print(f"Error: Project root '{args.project_root}' does not exist")
        sys.exit(1)
    
    generator = ParityReportGenerator(args.project_root)
    
    if args.verbose:
        print("Analyzing domain structures...")
    
    generator.generate_comprehensive_report(args.output)
    print(f"Comprehensive report generated: {args.output}")
    
    if args.json:
        generator.generate_json_summary(args.json)
        print(f"JSON summary generated: {args.json}")
    
    if args.verbose:
        print("Parity report generation completed!")

if __name__ == "__main__":
    main()