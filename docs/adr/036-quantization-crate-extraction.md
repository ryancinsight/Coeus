# ADR-036: Quantization Crate Extraction

**Status**: Accepted  
**Date**: 2026-01-16  
**Deciders**: Coeus Architecture Team  

## Context

The Coeus framework currently has quantization logic scattered across multiple crates:
- Quantization algorithms in `nn/src/quantization/`
- Quantized data types in `dtype/src/quantized/`
- Fake quantization logic mixed with neural network operations

This violates the principle of domain separation and makes quantization functionality harder to maintain, test, and extend.

## Decision

We will extract all quantization-related functionality into a dedicated `quantization` crate with the following structure:

### New Quantization Crate Structure

```
quantization/
├── src/
│   ├── algorithms/
│   │   ├── symmetric.rs        # Symmetric quantization (zero_point = 0)
│   │   ├── asymmetric.rs       # Asymmetric quantization
│   │   ├── dynamic.rs          # Dynamic quantization
│   │   └── core.rs             # Common utilities
│   ├── calibration/
│   │   ├── entropy.rs          # KL divergence-based calibration
│   │   ├── percentile.rs       # Percentile-based calibration
│   │   ├── mse.rs              # MSE-based calibration
│   │   └── histogram.rs        # Histogram utilities
│   ├── fake_quantize/
│   │   ├── linear.rs           # Fake quantization for linear layers
│   │   ├── conv.rs             # Fake quantization for convolution
│   │   └── core.rs             # Common fake quantization utilities
│   ├── types/
│   │   ├── qint4.rs            # 4-bit quantized integer
│   │   ├── qint8.rs            # 8-bit quantized integer
│   │   ├── qint16.rs           # 16-bit quantized integer
│   │   └── scale_zero_point.rs # Scale and zero-point parameters
│   ├── kernels/
│   │   ├── quantize.rs         # Quantization kernels
│   │   ├── dequantize.rs       # Dequantization kernels
│   │   └── quantized_ops.rs    # Quantized arithmetic operations
│   ├── lib.rs
│   ├── error.rs
│   └── config.rs
└── Cargo.toml
```

### Migration Plan

1. **Move quantization algorithms** from `nn/src/quantization/` to `quantization/src/algorithms/`
2. **Move quantized types** from `dtype/src/quantized/` to `quantization/src/types/`
3. **Extract fake quantization** from nn crate to `quantization/src/fake_quantize/`
4. **Update all imports** across the codebase
5. **Ensure dtype crate** contains only pure type definitions

### Dependencies

The quantization crate will depend on:
- `coeus-dtype` (for base data types)
- `coeus-storage` (for storage operations)
- `coeus-backend` (for hardware execution)

## Rationale

### Benefits

1. **Domain Separation**: Clear separation of quantization concerns from neural networks and data types
2. **Maintainability**: Easier to maintain and extend quantization functionality
3. **Testability**: Dedicated test suite for quantization algorithms
4. **Reusability**: Quantization can be used independently of neural networks
5. **Clarity**: Clear ownership of quantization-related code

### Design Principles Satisfied

- **Single Responsibility**: Each crate has a clear, focused responsibility
- **Domain Separation**: Quantization logic contained within appropriate boundaries
- **Hierarchical Organization**: Clear file structure for different quantization aspects
- **B<S<T>> Architecture**: Maintains generic architecture pattern

## Consequences

### Positive

- **Cleaner Architecture**: Better separation of concerns
- **Easier Testing**: Dedicated quantization test suite
- **Better Documentation**: Focused documentation for quantization
- **Extensibility**: Easier to add new quantization algorithms
- **Independence**: Quantization can be used without neural networks

### Negative

- **Breaking Changes**: Existing code will need import updates
- **Additional Complexity**: One more crate to manage
- **Migration Effort**: Requires systematic migration of existing code

### Migration Impact

**Low Impact Changes**:
- Import statement updates
- Cargo.toml dependency additions

**Medium Impact Changes**:
- Code that directly uses quantized types
- Neural network layers with quantization

**High Impact Changes**:
- Custom quantization implementations
- Code that mixes quantization with other concerns

## Implementation

### Phase 1: Crate Creation
- Create new quantization crate structure
- Set up basic module organization
- Define public API

### Phase 2: Algorithm Migration
- Move quantization algorithms from nn crate
- Split algorithms into separate files
- Update internal imports

### Phase 3: Type Migration
- Move quantized types from dtype crate
- Update dtype crate to remove quantization logic
- Ensure clean separation

### Phase 4: Fake Quantization Migration
- Extract fake quantization from nn crate
- Organize by operation type (linear, conv, etc.)
- Update nn crate to use quantization crate

### Phase 5: Import Updates
- Update all imports across codebase
- Update PyCoeus imports
- Verify compilation

### Phase 6: Testing and Validation
- Comprehensive test suite for quantization crate
- Integration tests with nn crate
- Performance validation

## Alternatives Considered

### Alternative 1: Keep quantization in nn crate
**Rejected**: Violates domain separation principle and makes quantization harder to use independently.

### Alternative 2: Keep quantized types in dtype crate
**Rejected**: dtype should contain only pure type definitions, not quantization logic.

### Alternative 3: Split quantization across multiple specialized crates
**Rejected**: Would create too much fragmentation for a cohesive domain.

## References

- [Requirements 17.1-17.7](../../.kiro/specs/coeus-architecture-enhancement/requirements.md#requirement-17-quantization-crate-extraction)
- [Design Document: Quantization Crate](../../.kiro/specs/coeus-architecture-enhancement/design.md#quantization-crate-new)
- [Domain Separation Principles](./generic-architecture-commitment.md)

## Status

**Accepted** - This ADR has been approved and implementation is in progress.

## Notes

- Migration guide will be provided to help users update their code
- Compatibility layer may be provided temporarily during transition
- Performance benchmarks will validate no regression from extraction