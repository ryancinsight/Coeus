"""
Audit what Coeus actually has implemented in its crates.
This helps identify what's already available but not exposed in PyCoeus.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set

def find_rust_crates() -> List[str]:
    """Find all Rust crates in the workspace."""
    crates = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and os.path.exists(os.path.join(item, 'Cargo.toml')):
            # Skip pycoeus and examples
            if item not in ['pycoeus', 'examples', 'target', 'xtask']:
                crates.append(item)
    return sorted(crates)


def parse_lib_rs(crate_name: str) -> Dict[str, List[str]]:
    """Parse lib.rs to find public modules and functions."""
    lib_path = Path(crate_name) / 'src' / 'lib.rs'
    if not lib_path.exists():
        return {}
    
    with open(lib_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find public modules
    pub_mods = re.findall(r'^pub\s+mod\s+(\w+);', content, re.MULTILINE)
    
    # Find public functions
    pub_fns = re.findall(r'^pub\s+fn\s+(\w+)', content, re.MULTILINE)
    
    # Find public structs
    pub_structs = re.findall(r'^pub\s+struct\s+(\w+)', content, re.MULTILINE)
    
    # Find public enums
    pub_enums = re.findall(r'^pub\s+enum\s+(\w+)', content, re.MULTILINE)
    
    # Find public traits
    pub_traits = re.findall(r'^pub\s+trait\s+(\w+)', content, re.MULTILINE)
    
    return {
        'modules': pub_mods,
        'functions': pub_fns,
        'structs': pub_structs,
        'enums': pub_enums,
        'traits': pub_traits,
    }


def check_pycoeus_exposure(crate_name: str) -> bool:
    """Check if a crate is exposed in PyCoeus."""
    pycoeus_lib = Path('pycoeus') / 'src' / 'lib.rs'
    if not pycoeus_lib.exists():
        return False
    
    with open(pycoeus_lib, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if crate is mentioned in PyCoeus
    return crate_name in content or f'pub mod {crate_name}' in content


def generate_audit_report():
    """Generate audit report of Coeus modules."""
    crates = find_rust_crates()
    
    report = []
    report.append("# Coeus Module Audit Report")
    report.append("")
    report.append("This report audits what Coeus has implemented in its crates and whether they're exposed in PyCoeus.")
    report.append("")
    
    # Categorize crates
    core_crates = ['tensor', 'storage', 'backend', 'dtype', 'autograd']
    nn_crates = ['nn', 'optim']
    specialized_crates = ['linalg', 'fft', 'signal', 'special', 'sparse']
    advanced_crates = ['distributed', 'distributions', 'jit', 'profiling']
    utility_crates = ['hub', 'tokenizer', 'vision', 'audio', 'utils']
    foundation_crates = ['foundation', 'coeus-error', 'coeus-semantic-api']
    
    categories = {
        'Core Infrastructure': core_crates,
        'Neural Networks': nn_crates,
        'Specialized Math': specialized_crates,
        'Advanced Features': advanced_crates,
        'Utilities': utility_crates,
        'Foundation': foundation_crates,
    }
    
    # Summary statistics
    total_crates = len(crates)
    exposed_count = 0
    not_exposed_count = 0
    
    report.append("## Summary")
    report.append("")
    report.append(f"**Total Crates:** {total_crates}")
    report.append("")
    
    # Detailed breakdown
    for category, crate_list in categories.items():
        report.append(f"## {category}")
        report.append("")
        
        for crate in crate_list:
            if crate not in crates:
                continue
            
            exposed = check_pycoeus_exposure(crate)
            if exposed:
                exposed_count += 1
            else:
                not_exposed_count += 1
            
            status = "✅ Exposed in PyCoeus" if exposed else "❌ Not exposed in PyCoeus"
            report.append(f"### {crate} - {status}")
            report.append("")
            
            # Parse lib.rs
            apis = parse_lib_rs(crate)
            
            if apis.get('modules'):
                report.append(f"**Public Modules ({len(apis['modules'])}):**")
                for mod in apis['modules'][:10]:
                    report.append(f"- `{mod}`")
                if len(apis['modules']) > 10:
                    report.append(f"- ... and {len(apis['modules']) - 10} more")
                report.append("")
            
            if apis.get('functions'):
                report.append(f"**Public Functions ({len(apis['functions'])}):**")
                for fn in apis['functions'][:10]:
                    report.append(f"- `{fn}`")
                if len(apis['functions']) > 10:
                    report.append(f"- ... and {len(apis['functions']) - 10} more")
                report.append("")
            
            if apis.get('structs'):
                report.append(f"**Public Structs ({len(apis['structs'])}):**")
                for struct in apis['structs'][:10]:
                    report.append(f"- `{struct}`")
                if len(apis['structs']) > 10:
                    report.append(f"- ... and {len(apis['structs']) - 10} more")
                report.append("")
            
            if apis.get('traits'):
                report.append(f"**Public Traits ({len(apis['traits'])}):**")
                for trait in apis['traits'][:5]:
                    report.append(f"- `{trait}`")
                if len(apis['traits']) > 5:
                    report.append(f"- ... and {len(apis['traits']) - 5} more")
                report.append("")
            
            report.append("---")
            report.append("")
    
    # Update summary
    report.insert(4, f"**Exposed in PyCoeus:** {exposed_count}")
    report.insert(5, f"**Not Exposed in PyCoeus:** {not_exposed_count}")
    report.insert(6, "")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("### High Priority: Expose Existing Crates")
    report.append("")
    report.append("The following crates are implemented but not exposed in PyCoeus:")
    report.append("")
    
    for crate in crates:
        if not check_pycoeus_exposure(crate):
            apis = parse_lib_rs(crate)
            total_apis = sum(len(v) for v in apis.values())
            if total_apis > 0:
                report.append(f"- **{crate}**: {total_apis} public APIs available")
    
    report.append("")
    report.append("### Action Items")
    report.append("")
    report.append("1. **Expose `linalg` crate**: Linear algebra operations (svd, qr, cholesky, etc.)")
    report.append("2. **Expose `signal` crate**: Signal processing (STFT, windows)")
    report.append("3. **Expose `special` crate**: Special functions (gamma, bessel, erf)")
    report.append("4. **Expose `sparse` crate**: Sparse tensor operations")
    report.append("5. **Expose `distributed` crate**: Distributed training (if implemented)")
    report.append("6. **Expose `distributions` crate**: Probability distributions")
    report.append("7. **Expose `vision` crate**: Vision transforms and utilities")
    report.append("8. **Expose `audio` crate**: Audio processing")
    report.append("9. **Expose `profiling` crate**: Performance profiling tools")
    report.append("")
    report.append("### Updated Parity Estimate")
    report.append("")
    report.append("If all existing crates are properly exposed in PyCoeus, the actual parity would be significantly higher than the current 3.5% module-level and 5.7% tensor method parity.")
    report.append("")
    report.append("**Estimated Impact:**")
    report.append("- `linalg` crate: +30 operations (svd, qr, cholesky, det, solve, etc.)")
    report.append("- `fft` crate: +8 operations (already partially exposed)")
    report.append("- `signal` crate: +10 operations (stft, windows)")
    report.append("- `special` crate: +15 operations (gamma, bessel, erf)")
    report.append("- `sparse` crate: +20 operations (sparse tensor ops)")
    report.append("- `vision` crate: +15 transforms")
    report.append("- `audio` crate: +10 operations")
    report.append("")
    report.append("**Total Potential Gain:** ~100+ operations just by exposing existing crates!")
    report.append("")
    
    # Write report
    with open('.kiro/specs/coeus-architecture-enhancement/COEUS_MODULE_AUDIT.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Total crates: {total_crates}")
    print(f"Exposed in PyCoeus: {exposed_count}")
    print(f"Not exposed: {not_exposed_count}")
    print(f"\nReport saved to: .kiro/specs/coeus-architecture-enhancement/COEUS_MODULE_AUDIT.md")


if __name__ == "__main__":
    generate_audit_report()
