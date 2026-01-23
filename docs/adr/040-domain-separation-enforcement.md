# ADR-040: Domain Separation Enforcement

**Status**: Accepted  
**Date**: 2026-01-16  
**Deciders**: Coeus Architecture Team  

## Context

The current Coeus architecture has some violations of domain separation principles:

1. **Cross-Domain Functionality**: Sparse operations exist in tensor crate, quantization logic in dtype crate
2. **Unclear Boundaries**: Some crates have functionality that belongs in other domains
3. **Maintenance Issues**: Changes in one domain affect unrelated domains
4. **Testing Complexity**: Domain violations make testing more complex

Clear domain boundaries are essential for maintainability, testability, and architectural clarity.

## Decision

We will enforce strict domain separation with clear boundaries between crates and their responsibilities:

### Domain Definitions

| Domain | Crate | Responsibility | Allowed Dependencies |
|--------|-------|----------------|---------------------|
| **Neural Networks** | `nn` | NN layers, operations, training | tensor, dense, sparse, quantization |
| **Multi-Dimensional** | `tensor` | Tensor API, autograd integration | dense, sparse, quantization, storage |
| **Dense Operations** | `dense` | Dense-specific algorithms | storage, dtype |
| **Sparse Operations** | `sparse` | Sparse-specific algorithms | storage, dtype |
| **Quantization** | `quantization` | Quantization algorithms | storage, dtype |
| **Memory Management** | `storage` | Memory layout, basic operations | backend, dtype |
| **Hardware Execution** | `backend` | Device-specific primitives | dtype |
| **Type System** | `dtype` | Type definitions, conversions | None |

### Domain Boundaries

#### What Each Domain MUST NOT Contain

**Neural Networks (`nn`)**:
- ❌ Sparse matrix algorithms (belongs in `sparse`)
- ❌ Quantization algorithms (belongs in `quantization`)
- ❌ Dense tensor algorithms (belongs in `dense`)
- ❌ Storage management (belongs in `storage`)
- ❌ Backend-specific code (belongs in `backend`)

**Tensor (`tensor`)**:
- ❌ Neural network layers (belongs in `nn`)
- ❌ Dense-specific algorithms (belongs in `dense`)
- ❌ Sparse-specific algorithms (belongs in `sparse`)
- ❌ Quantization algorithms (belongs in `quantization`)

**Dense (`dense`)**:
- ❌ Sparse operations (belongs in `sparse`)
- ❌ Quantization operations (belongs in `quantization`)
- ❌ Neural network operations (belongs in `nn`)
- ❌ Backend-specific code (belongs in `backend`)

**Sparse (`sparse`)**:
- ❌ Dense operations (belongs in `dense`)
- ❌ Quantization operations (belongs in `quantization`)
- ❌ Neural network operations (belongs in `nn`)
- ❌ Backend-specific code (belongs in `backend`)

**Quantization (`quantization`)**:
- ❌ Dense operations (belongs in `dense`)
- ❌ Sparse operations (belongs in `sparse`)
- ❌ Neural network layers (belongs in `nn`)
- ❌ Type definitions (belongs in `dtype`)

**Storage (`storage`)**:
- ❌ Complex algorithms (belongs in higher layers)
- ❌ Neural network operations (belongs in `nn`)
- ❌ Backend-specific implementations (belongs in `backend`)

**Backend (`backend`)**:
- ❌ High-level algorithms (belongs in higher layers)
- ❌ Storage management (belongs in `storage`)
- ❌ Type definitions (belongs in `dtype`)

**Dtype (`dtype`)**:
- ❌ Operations or algorithms (belongs in higher layers)
- ❌ Quantization logic (belongs in `quantization`)

## Rationale

### Benefits

1. **Clear Responsibilities**: Each crate has a single, well-defined responsibility
2. **Easier Maintenance**: Changes in one domain don't affect others
3. **Better Testing**: Domain-specific test suites
4. **Cleaner Dependencies**: Clear dependency hierarchy
5. **Improved Modularity**: Domains can be developed independently

### Design Principles Satisfied

- **Single Responsibility Principle**: Each crate has one clear purpose
- **Separation of Concerns**: Different concerns are in different crates
- **Dependency Inversion**: Higher-level crates depend on lower-level abstractions
- **Open/Closed Principle**: Domains are open for extension, closed for modification

### Domain Interaction Patterns

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural Networks (nn)                     │
│  - Layers, operations, training                             │
│  - Depends on: tensor, dense, sparse, quantization         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tensor (tensor)                        │
│  - Multi-dimensional API, autograd                          │
│  - Depends on: dense, sparse, quantization, storage        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│     Dense (dense)    │    Sparse (sparse)    │ Quantization │
│  - Dense algorithms  │  - Sparse algorithms  │ (quantization) │
│  - Depends on:       │  - Depends on:        │ - Depends on:  │
│    storage, dtype    │    storage, dtype     │   storage, dtype │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage (storage)                       │
│  - Memory management, basic operations                      │
│  - Depends on: backend, dtype                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Backend (backend)                      │
│  - Hardware primitives                                      │
│  - Depends on: dtype                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Dtype (dtype)                          │
│  - Type definitions, conversions                            │
│  - No dependencies                                           │
└─────────────────────────────────────────────────────────────┘
```

## Consequences

### Positive

- **Cleaner Architecture**: Clear separation of concerns
- **Better Maintainability**: Changes isolated to appropriate domains
- **Improved Testing**: Domain-specific test suites
- **Easier Development**: Developers can focus on specific domains
- **Better Documentation**: Clear domain-specific documentation

### Negative

- **Migration Effort**: Existing violations need to be fixed
- **Potential Duplication**: Some utilities might need to be duplicated
- **Coordination Overhead**: Changes affecting multiple domains need coordination

### Migration Impact

**High Impact Changes**:
- Moving sparse operations from tensor to sparse crate
- Moving quantization logic from dtype to quantization crate
- Moving complex operations from storage to appropriate crates

**Medium Impact Changes**:
- Updating imports and dependencies
- Reorganizing test suites by domain

**Low Impact Changes**:
- Documentation updates
- Example code updates

## Implementation

### Phase 1: Domain Audit
- Identify all cross-domain functionality violations
- Categorize violations by severity and impact
- Create migration plan for each violation

### Phase 2: Quantization Domain Separation
- Move quantization logic from dtype to quantization crate
- Move quantization operations from nn to quantization crate
- Update all imports and dependencies

### Phase 3: Dense/Sparse Domain Separation
- Move sparse operations from tensor to sparse crate
- Move dense operations from tensor to dense crate
- Ensure clear boundaries between dense and sparse

### Phase 4: Storage Domain Cleanup
- Move complex operations from storage to appropriate crates
- Ensure storage only contains basic operations
- Update storage dependencies

### Phase 5: Boundary Enforcement
- Create automated checks for domain violations
- Set up CI/CD checks to prevent future violations
- Document domain boundaries clearly

### Phase 6: Testing and Validation
- Reorganize test suites by domain
- Ensure comprehensive domain-specific testing
- Validate no functionality is lost in migration

## Domain Enforcement Mechanisms

### Automated Checks

```rust
// scripts/check_domain_boundaries.rs
fn check_domain_violations() -> Result<Vec<Violation>> {
    let mut violations = Vec::new();
    
    // Check for sparse operations in tensor crate
    if has_sparse_operations("tensor/src/") {
        violations.push(Violation::new(
            "tensor", 
            "sparse operations", 
            "should be in sparse crate"
        ));
    }
    
    // Check for quantization in dtype crate
    if has_quantization_logic("dtype/src/") {
        violations.push(Violation::new(
            "dtype", 
            "quantization logic", 
            "should be in quantization crate"
        ));
    }
    
    // Check for complex operations in storage crate
    if has_complex_operations("storage/src/") {
        violations.push(Violation::new(
            "storage", 
            "complex operations", 
            "should be in higher-level crates"
        ));
    }
    
    Ok(violations)
}
```

### CI/CD Integration

```yaml
# .github/workflows/domain-check.yml
name: Domain Boundary Check
on: [push, pull_request]

jobs:
  domain-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Check Domain Boundaries
        run: |
          cargo run --bin check_domain_boundaries
          if [ $? -ne 0 ]; then
            echo "Domain boundary violations detected!"
            exit 1
          fi
```

### Documentation Requirements

Each crate must document:
- **Domain Responsibility**: What the crate is responsible for
- **Allowed Dependencies**: Which crates it can depend on
- **Forbidden Functionality**: What it must NOT contain
- **Interface Contracts**: How it interacts with other domains

## Interface Contracts

### Clear Inter-Domain Interfaces

```rust
// Example: tensor crate using dense crate
use coeus_dense::ops::elementwise::add as dense_add;

impl<B, T> Tensor<B, DenseStorage<T>, T> {
    pub fn add(&self, other: &Self) -> Result<Self> {
        // Tensor crate delegates to dense crate
        let result_storage = dense_add(self.storage(), other.storage())?;
        Ok(Tensor::from_storage(result_storage, self.backend().clone()))
    }
}
```

### Dependency Injection

```rust
// Higher-level crates depend on abstractions, not implementations
pub trait DenseOperations<T: DataType> {
    fn add(&self, other: &Self) -> Result<Self>;
    fn mul(&self, other: &Self) -> Result<Self>;
}

// Implementation in dense crate
impl<T: DataType> DenseOperations<T> for DenseStorage<T> {
    fn add(&self, other: &Self) -> Result<Self> {
        // Dense-specific implementation
    }
}
```

## Alternatives Considered

### Alternative 1: Relaxed domain boundaries
**Rejected**: Would lead to continued architectural degradation and maintenance issues.

### Alternative 2: Merge related domains
**Rejected**: Would create overly large crates with multiple responsibilities.

### Alternative 3: Create more fine-grained domains
**Rejected**: Would create too much fragmentation and coordination overhead.

## References

- [Requirements 16.1-16.7](../../.kiro/specs/coeus-architecture-enhancement/requirements.md#requirement-16-domain-separation-and-crate-boundaries)
- [Design Document: Domain Separation](../../.kiro/specs/coeus-architecture-enhancement/design.md#domain-separation-enforcement)
- [Quantization Crate Extraction](./036-quantization-crate-extraction.md)
- [Dense Crate Creation](./037-dense-crate-creation.md)

## Status

**Accepted** - This ADR has been approved and implementation is in progress.

## Notes

- Domain boundaries will be enforced through automated checks
- Migration will be done incrementally to minimize disruption
- Clear documentation will help developers understand domain responsibilities
- Interface contracts will ensure clean inter-domain communication