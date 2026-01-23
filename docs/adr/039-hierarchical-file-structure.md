# ADR-039: Hierarchical File Structure for Parity Tracking

**Status**: Accepted  
**Date**: 2026-01-16  
**Deciders**: Coeus Architecture Team  

## Context

The current file structure in Coeus uses monolithic files and shallow hierarchies, which creates several problems:

1. **Parity Tracking Difficulty**: Hard to identify missing implementations across backends
2. **Monolithic Files**: Large files with multiple operations are hard to maintain
3. **Inconsistent Organization**: Different crates use different organizational patterns
4. **Script-Based Analysis**: Difficult to automate parity checking and status tracking

We need a systematic approach to file organization that enables automated parity tracking and maintains clear separation of concerns.

## Decision

We will implement a deep vertical hierarchical file structure organized by domain and implementation type, with the following principles:

### Hierarchical Organization Principles

1. **Deep Vertical Hierarchies**: Mirror implementation domains (dense, sparse, quantized)
2. **Parallel File Structures**: Maintain identical structures across backends
3. **Domain-Specific Organization**: Each crate maintains its own hierarchy
4. **Script-Friendly Patterns**: Enable automated parity comparison
5. **No Monolithic Files**: Split operations by category and specific function

### File Structure Pattern

```
{crate}/src/
├── {domain}/
│   ├── {category}/
│   │   ├── {operation}.rs      # Single operation per file
│   │   ├── {operation2}.rs     # Another operation
│   │   └── mod.rs              # Module exports
│   ├── {category2}/
│   │   └── [similar structure]
│   └── mod.rs
└── lib.rs
```

### Backend Parity Structure

All backends must maintain identical file structures:

```
backend/src/
├── cpu/
│   ├── arithmetic/
│   │   ├── add.rs
│   │   ├── sub.rs
│   │   ├── mul.rs
│   │   └── div.rs
│   ├── linear_algebra/
│   │   ├── matmul.rs
│   │   ├── transpose.rs
│   │   └── decomposition.rs
│   └── activation/
│       ├── relu.rs
│       ├── sigmoid.rs
│       └── tanh.rs
├── gpu/
│   ├── arithmetic/          # Identical structure to CPU
│   │   ├── add.rs
│   │   ├── sub.rs
│   │   ├── mul.rs
│   │   └── div.rs
│   ├── linear_algebra/      # Identical structure to CPU
│   └── activation/          # Identical structure to CPU
├── tpu/
│   └── [identical structure to CPU/GPU]
└── npu/
    └── [identical structure to CPU/GPU]
```

## Rationale

### Benefits

1. **Automated Parity Tracking**: Scripts can compare file presence across backends
2. **Missing Implementation Detection**: Absent files indicate missing functionality
3. **Consistent Organization**: All crates follow the same organizational principles
4. **Maintainability**: Small, focused files are easier to maintain
5. **Clear Ownership**: Each file has a single, clear responsibility

### Parity Tracking Capabilities

The hierarchical structure enables:

```bash
# Check for missing implementations
find backend/src/cpu -name "*.rs" | sed 's/cpu/gpu/' | xargs -I {} test -f {} || echo "Missing: {}"

# Generate parity report
python scripts/check_backend_parity.py --output parity_report.md

# Identify implementation gaps
python scripts/generate_parity_matrix.py --backends cpu,gpu,tpu,npu
```

### Design Principles Satisfied

- **Single Responsibility**: Each file handles one operation
- **Domain Separation**: Clear boundaries between different domains
- **Consistency**: Uniform organization across all crates
- **Automation-Friendly**: Structure enables script-based analysis

## Consequences

### Positive

- **Better Parity Tracking**: Easy to identify missing implementations
- **Improved Maintainability**: Small, focused files
- **Consistent Organization**: Uniform structure across crates
- **Automated Analysis**: Scripts can analyze implementation status
- **Clear Navigation**: Easy to find specific operations

### Negative

- **More Files**: Increased number of files to manage
- **Migration Effort**: Existing monolithic files need to be split
- **Directory Depth**: Deeper directory structures

### File Count Impact

| Crate | Before | After | Change |
|-------|--------|-------|--------|
| backend | ~20 files | ~200 files | +180 files |
| nn | ~30 files | ~150 files | +120 files |
| storage | ~15 files | ~80 files | +65 files |
| sparse | ~10 files | ~60 files | +50 files |

## Implementation

### Phase 1: Backend Restructuring
- Create hierarchical structure for CPU backend
- Split monolithic files into operation-specific files
- Replicate structure for GPU, TPU, NPU backends

### Phase 2: Storage Restructuring
- Organize storage operations by format and category
- Split arithmetic, layout, and creation operations
- Maintain parallel structures for dense, sparse, quantized

### Phase 3: NN Restructuring
- Organize functional operations by category
- Split activation, loss, convolution, etc. into separate files
- Maintain parallel structure in modules directory

### Phase 4: Sparse Restructuring
- Organize by format (CSR, CSC, COO) and operation type
- Split arithmetic, conversion, and indexing operations
- Maintain consistent structure across formats

### Phase 5: Parity Scripts
- Create scripts to check backend parity
- Implement automated parity reporting
- Set up continuous integration checks

### Phase 6: Documentation
- Update navigation guides
- Document file organization principles
- Create developer guides for finding operations

## Directory Nesting Limits

To maintain readability, we limit directory nesting:

**Maximum 4 levels**:
```
crate/src/domain/category/operation.rs  # 4 levels maximum
```

**Examples**:
```
backend/src/cpu/arithmetic/add.rs       # 4 levels ✓
nn/src/functional/ops/activation/relu.rs # 5 levels ❌ (too deep)
nn/src/ops/activation/relu.rs           # 4 levels ✓ (better)
```

## File Naming Conventions

### Operation Files
- **Single operation per file**: `add.rs`, `relu.rs`, `conv2d.rs`
- **Descriptive names**: Use full operation names, not abbreviations
- **Consistent naming**: Same operation has same filename across backends

### Module Files
- **mod.rs**: Only for re-exports, keep under 10 lines
- **lib.rs**: Public API and top-level module declarations
- **error.rs**: Error types for the crate

### Directory Names
- **Lowercase with underscores**: `linear_algebra`, `activation`
- **Descriptive**: Clear indication of contained operations
- **Consistent**: Same category names across crates

## Parity Tracking Scripts

### Backend Parity Checker

```python
# scripts/check_backend_parity.py
def check_backend_parity():
    backends = ['cpu', 'gpu', 'tpu', 'npu']
    base_path = 'backend/src'
    
    # Get CPU file structure as reference
    cpu_files = get_file_structure(f'{base_path}/cpu')
    
    for backend in backends[1:]:  # Skip CPU (reference)
        backend_files = get_file_structure(f'{base_path}/{backend}')
        missing = cpu_files - backend_files
        extra = backend_files - cpu_files
        
        if missing:
            print(f"{backend} missing: {missing}")
        if extra:
            print(f"{backend} extra: {extra}")
```

### Parity Report Generator

```python
# scripts/generate_parity_report.py
def generate_parity_matrix():
    operations = discover_operations()
    backends = ['cpu', 'gpu', 'tpu', 'npu']
    
    matrix = {}
    for op in operations:
        matrix[op] = {}
        for backend in backends:
            matrix[op][backend] = file_exists(f'backend/src/{backend}/{op}.rs')
    
    return matrix
```

## Alternatives Considered

### Alternative 1: Keep monolithic files
**Rejected**: Makes parity tracking difficult and violates single responsibility.

### Alternative 2: Flat file structure with prefixes
**Rejected**: Doesn't provide clear organization and makes navigation difficult.

### Alternative 3: Very deep hierarchies (5+ levels)
**Rejected**: Makes navigation difficult and paths too long.

### Alternative 4: Different structures per crate
**Rejected**: Inconsistency makes the codebase harder to navigate.

## References

- [Requirements 8.1-8.7](../../.kiro/specs/coeus-architecture-enhancement/requirements.md#requirement-8-hierarchical-file-structure-for-parity-tracking)
- [Design Document: Hierarchical File Structure](../../.kiro/specs/coeus-architecture-enhancement/design.md#hierarchical-file-structure-for-parity-tracking)
- [Parity Tracking Methodology](../PARITY_TRACKING.md)

## Status

**Accepted** - This ADR has been approved and implementation is in progress.

## Notes

- Migration will be done incrementally to minimize disruption
- Parity tracking scripts will be developed alongside file restructuring
- Documentation will be updated to reflect new file organization
- Developer guides will help contributors navigate the new structure