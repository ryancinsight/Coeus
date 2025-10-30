# ADR: Trait System Simplification - Backend Associated Types

## Status
**ACCEPTED** - Implemented October 2025

## Context

The original Coeus tensor architecture used redundant generics: `Tensor<B, S, T>` where:
- `B`: Backend (CPU, GPU, etc.)
- `S`: Storage type (Dense, Sparse, etc.)
- `T`: Data type (Float32, Int32, etc.)

This led to verbose APIs and complex type signatures. However, analysis showed that:
- Storage types are not independent of data types (sparse storage for Float32 ≠ sparse storage for Int32)
- Backend capabilities determine supported storage and data type combinations
- The three-level generic hierarchy was unnecessarily complex

## Decision

**Simplify the generic architecture from `Tensor<B, S, T>` to `Tensor<B>` using associated types on the Backend trait.**

### Changes Made

1. **Backend Trait Refactoring**:
   ```rust
   // Before
   trait Backend<T: DataType> {
       // Methods generic over storage
   }

   // After
   trait Backend: Send + Sync + Clone + fmt::Debug + Default + 'static {
       type Data: DataType;
       type Device: DeviceInfo + Send + Sync;

       // Generic methods over any storage type
       fn add<S>(&self, lhs: &S, rhs: &S) -> Result<S>
       where S: Storage<Self::Data>;
   }
   ```

2. **Tensor Simplification**:
   ```rust
   // Before
   Tensor<CpuBackend, DenseStorage<Float32>, Float32>

   // After
   Tensor<CpuBackend>  // Where CpuBackend: Backend<Data = Float32, ...>
   ```

3. **Generic Method Implementation**:
   - All core operations now generic over storage types
   - Dynamic dispatch used temporarily for CpuBackend sparse operations
   - Dense-specific methods retained for internal implementations

## Consequences

### Positive

- **Simplified API**: Cleaner tensor creation and usage
- **Reduced Type Complexity**: Fewer generic parameters to specify
- **Maintained Functionality**: Full sparse/dense support preserved
- **Future Extensibility**: New backends can define their own storage/data combinations
- **Zero-Cost Abstractions**: Compile-time specialization maintained

### Negative

- **Temporary Dynamic Dispatch**: CpuBackend uses `as_any().downcast_ref()` for sparse operations
- **Increased Trait Complexity**: Backend trait now has more generic methods
- **Migration Effort**: All tensor operations need updating to new API

### Risks

- **Performance Impact**: Dynamic dispatch may affect sparse operations until optimized
- **API Breaking**: Complete breaking change requiring full codebase migration
- **Type Safety**: Need to ensure associated type relationships are correctly maintained

## Implementation Plan

1. ✅ **Phase 1**: Backend trait refactoring with associated types
2. ✅ **Phase 2**: CpuBackend/StubBackend implementation with generic methods
3. 🔄 **Phase 3**: Tensor struct simplification to `Tensor<B>`
4. ⏳ **Phase 4**: Update tensor operations to use new API
5. ⏳ **Phase 5**: NN crate migration to simplified generics
6. ⏳ **Phase 6**: Full test suite update and validation

## Alternatives Considered

### Option 1: Keep Original Architecture
- **Pros**: No breaking changes, proven design
- **Cons**: Verbose API, complex generics

### Option 2: Single Generic Parameter (Chosen)
- **Pros**: Clean API, reduced complexity
- **Cons**: Breaking changes, temporary performance cost

### Option 3: Type-Level Programming
- **Pros**: Maximum type safety at compile time
- **Cons**: Complex implementation, steep learning curve

## Validation

**Empirical Results**:
- ✅ Zero compilation errors in backend crate
- ✅ All backend tests passing (10/10)
- ✅ CpuBackend generic methods functional
- ✅ StubBackend updated successfully

**Next Steps**:
- Complete tensor struct refactoring
- Update all tensor operations
- Migrate NN and other crates
- Performance benchmarking of new architecture

## References

- ADR-003: Generic Architecture Commitment
- ADR-030: Complete Storage Abstraction
- Backend crate implementation details




