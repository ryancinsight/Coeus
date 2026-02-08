#!/usr/bin/env python3
"""
Compare PyTorch and Pycoeus (Coeus Python bindings) API parity.

This script compares the exported symbols from torch and pycoeus
to identify missing components in the Coeus implementation.

Usage:
    python compare_coeus_torch.py
    
Requirements:
    pip install torch maturin
    cd D:\coeus && maturin develop -p pycoeus
"""

import sys
import difflib
from typing import Set, List, Dict, Tuple

def get_pytorch_exports() -> Set[str]:
    """Get all exported symbols from PyTorch."""
    try:
        import torch
        exports = set()
        for name in dir(torch):
            if not name.startswith('_'):
                exports.add(name)
        return exports
    except ImportError:
        print("Error: torch not installed. Run: pip install torch")
        return set()

def get_pycoeus_exports() -> Set[str]:
    """Get all exported symbols from Coeus."""
    try:
        import coeus
        exports = set()
        for name in dir(coeus):
            if not name.startswith('_'):
                exports.add(name)
        return exports
    except ImportError:
        print("Error: coeus not installed. Build with: maturin develop -p pycoeus")
        return set()

def categorize_symbol(name: str) -> str:
    """Categorize a symbol by its likely type."""
    name_lower = name.lower()
    
    categories = {
        'nn': ['linear', 'conv', 'batchnorm', 'layernorm', 'dropout', 'relu', 'sigmoid', 
               'tanh', 'gelu', 'softmax', 'embedding', 'rnn', 'lstm', 'gru', 'transformer',
               'mlp', 'attention', 'pool', 'unpool', 'upsample', 'module', 'sequential'],
        'optim': ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamw', 'optimizer',
                  'lr_scheduler', 'learning rate'],
        'tensor': ['tensor', 'zeros', 'ones', 'rand', 'randn', 'empty', 'full', 'eye',
                   'arange', 'linspace', 'logspace'],
        'ops': ['matmul', 'add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'exp', 'log',
                'sum', 'mean', 'max', 'min', 'argmax', 'argmin', 'transpose', 'permute',
                'reshape', 'view', 'squeeze', 'unsqueeze', 'cat', 'stack', 'split'],
        'linalg': ['det', 'inv', 'solve', 'cholesky', 'eig', 'svd', 'qr', 'lu',
                   'norm', 'matrix', 'vector'],
        'loss': ['loss', 'crossentropy', 'mse', 'bce', 'nll', 'kl_div'],
        'init': ['init', 'uniform', 'normal', 'xavier', 'kaiming', 'orthogonal'],
        'utils': ['data', 'dataloader', 'dataset', 'sampler', 'batch'],
        'cuda': ['cuda', 'gpu', 'device'],
        'jit': ['jit', 'script', 'trace'],
        'autograd': ['grad', 'backward', 'no_grad', 'enable_grad', 'requires_grad'],
    }
    
    for category, keywords in categories.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'other'

def compare_apis() -> Tuple[Set[str], Set[str], Set[str]]:
    """Compare PyTorch and Pycoeus APIs."""
    torch_exports = get_pytorch_exports()
    pycoeus_exports = get_pycoeus_exports()
    
    if not torch_exports or not pycoeus_exports:
        return set(), set(), set()
    
    common = torch_exports & pycoeus_exports
    torch_only = torch_exports - pycoeus_exports
    pycoeus_only = pycoeus_exports - torch_exports
    
    return common, torch_only, pycoeus_only

def print_comparison():
    """Print detailed API comparison."""
    print("=" * 80)
    print("PyTorch vs Pycoeus API Comparison")
    print("=" * 80)
    
    common, torch_only, pycoeus_only = compare_apis()
    
    if not common and not torch_only and not pycoeus_only:
        print("\nError: Could not load both libraries.")
        print("Make sure both torch and pycoeus are installed.")
        return
    
    print(f"\n📊 Summary:")
    print(f"   Common symbols:     {len(common)}")
    print(f"   PyTorch only:       {len(torch_only)}")
    print(f"   Pycoeus only:       {len(pycoeus_only)}")
    
    # Categorize missing symbols
    if torch_only:
        print(f"\n❌ Missing in Pycoeus (by category):")
        by_category: Dict[str, List[str]] = {}
        for name in sorted(torch_only):
            cat = categorize_symbol(name)
            by_category.setdefault(cat, []).append(name)
        
        for cat in sorted(by_category.keys()):
            symbols = by_category[cat]
            print(f"\n   [{cat.upper()}] ({len(symbols)} symbols)")
            for name in symbols[:10]:  # Show first 10
                print(f"      - {name}")
            if len(symbols) > 10:
                print(f"      ... and {len(symbols) - 10} more")
    
    if pycoeus_only:
        print(f"\n✅ Pycoeus-specific extensions ({len(pycoeus_only)} symbols):")
        for name in sorted(pycoeus_only)[:20]:
            print(f"   - {name}")
        if len(pycoeus_only) > 20:
            print(f"   ... and {len(pycoeus_only) - 20} more")
    
    # Calculate coverage
    total_torch = len(common) + len(torch_only)
    coverage = len(common) / total_torch * 100 if total_torch > 0 else 0
    print(f"\n📈 API Coverage: {coverage:.1f}% ({len(common)}/{total_torch})")
    
    # Write detailed report
    write_detailed_report(common, torch_only, pycoeus_only)

def write_detailed_report(common: Set[str], torch_only: Set[str], pycoeus_only: Set[str]):
    """Write detailed report to file."""
    with open('comparison_missing.txt', 'w') as f:
        f.write("Missing PyTorch Symbols in Pycoeus\n")
        f.write("=" * 80 + "\n\n")
        
        by_category: Dict[str, List[str]] = {}
        for name in sorted(torch_only):
            cat = categorize_symbol(name)
            by_category.setdefault(cat, []).append(name)
        
        for cat in sorted(by_category.keys()):
            symbols = by_category[cat]
            f.write(f"\n[{cat.upper()}]\n")
            f.write("-" * 40 + "\n")
            for name in symbols:
                f.write(f"  {name}\n")
    
    with open('comparison_common.txt', 'w') as f:
        f.write("Common Symbols (Both PyTorch and Pycoeus)\n")
        f.write("=" * 80 + "\n\n")
        for name in sorted(common):
            f.write(f"  {name}\n")
    
    print(f"\n📝 Detailed reports written to:")
    print(f"   - comparison_missing.txt ({len(torch_only)} missing symbols)")
    print(f"   - comparison_common.txt ({len(common)} common symbols)")

def find_similar_symbols(name: str, candidates: Set[str], cutoff: float = 0.6) -> List[str]:
    """Find similar symbol names using fuzzy matching."""
    matches = difflib.get_close_matches(name, candidates, n=3, cutoff=cutoff)
    return matches

def main():
    """Main entry point."""
    print_comparison()
    
    # Interactive mode: suggest alternatives for missing symbols
    _, torch_only, pycoeus_only = compare_apis()
    
    if torch_only and pycoeus_only:
        print("\n🔍 Potential name mappings (fuzzy match suggestions):")
        for name in sorted(torch_only)[:10]:  # Check first 10 missing
            similar = find_similar_symbols(name, pycoeus_only)
            if similar:
                print(f"   {name} -> {', '.join(similar)}")

if __name__ == "__main__":
    main()
