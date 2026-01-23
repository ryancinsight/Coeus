# Tensor Crate Operation Organization Audit (Task 13.1)

**Date:** January 14, 2026  
**Task:** 13.1 Audit tensor operation organization  
**Requirements:** 1.2, 1.4 (Single Source of Truth)

## Executive Summary

This audit examines the tensor crate's operation organization to identify duplicate implementations and violations of the Single Source of Truth (SSOT) principle. The tensor crate has significant organizational issues with operations scattered across multiple locations.

## Current Structure

### Directory Organization

```
tensor/src/
├── ops/                          # Stateless operations (should be SSOT)
│   ├── arithmetic.rs             # Element-wise arithmetic operations
│   ├── comparison.rs             # Comparison operations
│   ├── creation.rs               # Tensor creation operations
│   ├── matrix.rs                 # Matrix operations
│   ├── missing_math.rs           # Additional math operations
│   ├── reduction.rs              # Reduction operations
│   ├── sparse.rs                 # Sparse operations
│   └── tensor_ops.rs             # Concatenate, stack operations
├── implementations/              # Tensor method implementations
│   ├── autograd.rs               # Autograd-related methods
│   ├── creation.rs               # Creation methods (from_vec, zeros, ones, etc.)
│   ├── manipulation.rs           # Manipulation methods (shape, to_dense, etc.)
│   └── math.rs                   # Math methods (clamp, mul_scalar, etc.)
├── elementwise.rs                # Element-wise operations (DUPLICATE)
├── shape_ops.rs                  # Shape manipulation operations
├── functions.rs                  # Autograd function objects
├── tensor_core.rs                # Core Tensor struct definition
├── tensor_dense_ops.rs           # Dense-specific operations
├── tensor_dense_ops_ext.rs       # Extended dense operations
├── tensor_sparse_ops.rs          # Sparse-specific operations
└── simd_ops.rs                   # SIMD-optimized operations
```

## Identified Issues

### 1. DUPLICATE IMPLEMENTATIONS - Mathematical Operations

**Issue:** Mathematical operations are implemented in multiple locations.

#### ops/arithmetic.rs (Stateless Functions)
- `exp()`, `log()`, `sin()`, `cos()`, `sqrt()`, `pow()`
- `abs()`, `floor()`, `ceil()`, `round()`, `trunc()`, `sign()`
- `sinh()`, `cosh()`, `tanh()`, `asin()`, `acos()`, `atan()`
- `exp2()`, `log10()`, `log2()`, `rsqrt()`
- `erf()`, `maximum()`, `minimum()`

#### elementwise.rs (Tensor Methods)
- `exp()`, `log()`, `sin()`, `cos()`, `sqrt()`, `powf()`, `square()`
- **DUPLICATE IMPLEMENTATIONS** - These are reimplementations of ops/arithmetic.rs functions

**Violation:** Single Source of Truth (Requirements 1.2, 1.4)

**Recommendation:** 
- DELETE `elementwise.rs` entirely
- Tensor methods should delegate to `ops/arithmetic.rs` functions
- Example: `tensor.exp()` should call `ops::arithmetic::exp(&tensor)`

### 2. DUPLICATE IMPLEMENTATIONS - Creation Operations

**Issue:** Tensor creation is split between two locations.

#### ops/creation.rs (Stateless Functions)
- Contains creation logic but appears to be unused or incomplete

#### implementations/creation.rs (Tensor Methods)
- `from_vec()`, `from_vec_with_backend()`, `from_storage()`
- `zeros()`, `ones()`, `full()`, `eye()`
- `randn()`, `rand()`, `randint()`
- `arange()`, `linspace()`
- `zeros_like()`, `ones_like()`, `full_like()`

**Status:** Need to verify if ops/creation.rs is actually used or if it's dead code.

**Recommendation:**
- If ops/creation.rs is unused, DELETE it
- Keep implementations/creation.rs as the single source for creation methods
- Creation is inherently tied to Tensor construction, so methods are appropriate

### 3. UNCLEAR SEPARATION - Shape Operations

**Issue:** Shape operations are in a separate file but could be better organized.

#### shape_ops.rs (Tensor Methods)
- `transpose()`, `permute()`, `unsqueeze()`, `squeeze()`
- `reshape()`, `flatten()`

**Status:** These are appropriately implemented as methods since they're fundamental tensor operations.

**Recommendation:**
- Keep shape_ops.rs as-is
- These operations are core to tensor manipulation and appropriately placed

### 4. DUPLICATE IMPLEMENTATIONS - Clamp Operations

**Issue:** Clamp operations exist in multiple locations.

#### ops/arithmetic.rs
- `clamp()` - stateless function

#### implementations/math.rs
- `clamp()`, `clamp_()`, `clamp_min()`, `clamp_min_()`, `clamp_max()`, `clamp_max_()`
- More comprehensive with in-place variants

**Violation:** Single Source of Truth (Requirements 1.2, 1.4)

**Recommendation:**
- Keep implementations/math.rs versions (more complete)
- Remove or delegate ops/arithmetic.rs::clamp() to implementations/math.rs
- OR: Move all clamp logic to ops/arithmetic.rs and have methods delegate

### 5. UNCLEAR ORGANIZATION - Dense vs Sparse Operations

**Issue:** Dense and sparse operations are separated but organization is unclear.

#### tensor_dense_ops.rs & tensor_dense_ops_ext.rs
- Dense-specific implementations
- Unclear why split into two files

#### tensor_sparse_ops.rs
- Sparse-specific implementations

#### ops/sparse.rs
- Sparse operations

**Recommendation:**
- Consolidate tensor_dense_ops.rs and tensor_dense_ops_ext.rs into one file
- Clarify relationship between ops/sparse.rs and tensor_sparse_ops.rs
- Ensure SSOT for sparse operations

### 6. MISSING ORGANIZATION - Reduction Operations

**Issue:** Reduction operations are scattered.

#### ops/reduction.rs
- Should contain reduction operations but need to verify contents

#### implementations/manipulation.rs
- Contains `sum_simd()` which is a reduction

**Recommendation:**
- Consolidate all reduction operations in ops/reduction.rs
- Methods should delegate to ops/reduction.rs functions

## Single Source of Truth Violations Summary

| Operation Category | Current Locations | SSOT Violation | Recommended Action |
|-------------------|-------------------|----------------|-------------------|
| Element-wise math | ops/arithmetic.rs, elementwise.rs | **YES** | Delete elementwise.rs, delegate to ops |
| Clamp operations | ops/arithmetic.rs, implementations/math.rs | **YES** | Consolidate in one location |
| Creation operations | ops/creation.rs, implementations/creation.rs | **MAYBE** | Verify if ops/creation.rs is used |
| Dense operations | tensor_dense_ops.rs, tensor_dense_ops_ext.rs | **MAYBE** | Consolidate into one file |
| Reduction operations | ops/reduction.rs, implementations/manipulation.rs | **MAYBE** | Verify and consolidate |

## Architectural Recommendations

### Proposed Organization

```
tensor/src/
├── ops/                          # SINGLE SOURCE OF TRUTH for operations
│   ├── arithmetic.rs             # All element-wise arithmetic
│   ├── comparison.rs             # All comparison operations
│   ├── matrix.rs                 # All matrix operations
│   ├── reduction.rs              # All reduction operations
│   ├── shape.rs                  # All shape operations (merge from shape_ops.rs)
│   ├── sparse.rs                 # All sparse operations
│   └── tensor_ops.rs             # Concatenate, stack, etc.
├── implementations/              # Tensor methods that DELEGATE to ops/
│   ├── creation.rs               # Creation methods (inherently tied to Tensor)
│   ├── manipulation.rs           # Convenience methods that delegate
│   └── autograd.rs               # Autograd integration
├── functions.rs                  # Autograd function objects
├── tensor_core.rs                # Core Tensor struct
└── simd_ops.rs                   # SIMD optimizations (implementation detail)
```

### Key Principles

1. **ops/ is the Single Source of Truth** for all tensor operations
2. **Tensor methods delegate to ops/** - no duplicate logic
3. **Creation methods are exceptions** - they construct Tensors, so they're inherently methods
4. **Shape operations can be methods** - they're fundamental to tensor manipulation
5. **Delete elementwise.rs** - clear SSOT violation

## Next Steps (Task 13.2)

1. Delete `elementwise.rs`
2. Update Tensor methods to delegate to `ops/arithmetic.rs`
3. Consolidate clamp operations
4. Verify and clean up creation operations
5. Consolidate dense operations files
6. Update all imports throughout the crate

## Compliance Check

- **Requirement 1.2:** Single Source of Truth - **VIOLATED** (multiple locations for same operations)
- **Requirement 1.4:** Eliminate duplicate implementations - **VIOLATED** (elementwise.rs duplicates ops/arithmetic.rs)

## Files to Modify/Delete

### Delete
- `tensor/src/elementwise.rs` - Complete duplicate of ops/arithmetic.rs

### Consolidate
- `tensor/src/tensor_dense_ops.rs` + `tensor/src/tensor_dense_ops_ext.rs` → single file
- Clamp operations: choose one location (ops/arithmetic.rs OR implementations/math.rs)

### Verify
- `tensor/src/ops/creation.rs` - Is this used? If not, delete.

### Update
- All files that import from elementwise.rs
- All files that use duplicate clamp implementations
