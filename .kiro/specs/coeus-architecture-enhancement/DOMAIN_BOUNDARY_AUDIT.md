# Domain Boundary Audit Report

**Date:** January 16, 2026  
**Task:** 5.1 Audit cross-domain functionality  
**Requirements:** 16.1, 16.2, 16.3, 16.4

## Executive Summary

This audit identifies violations of domain boundaries across the Coeus framework crates. The analysis focuses on three key areas:
1. Sparse operations in tensor crate (should be in sparse crate)
2. Quantization logic in dtype crate (should be in quantization crate)
3. Backend-specific code in storage crate (should be in backend crate)

## Findings

### 1. Sparse Operations in Tensor Crate ✅ ACCEPTABLE

**Status:** ACCEPTABLE - Proper delegation pattern

**Location:** `tensor/src/ops/sparse.rs`

**Analysis:**
The tensor crate contains sparse tensor operations, but these are **thin wrappers** that properly delegate to the `coeus-sparse` crate. This follows the correct architectural pattern:

```rust
// tensor/src/ops/sparse.rs
use coeus_sparse::{SparseMatMul, SparseAdd}; // Import traits from sparse crate

impl<B, T> Tensor<B, CsrStorage<T>, T> {
    pub fn sparse_matmul(&self, other: &Self) -> Result<Tensor<B, CooStorage<T>, T>> {
        // Delegates to sparse crate implementation
        let result_storage = self.storage
            .matmul_sparse(&other.storage, SparseFormat::Csr, &self.backend)
            .map_err(TensorError::StorageError)?;
        Ok(Tensor::from_storage(result_storage, self.backend.clone()))
    }
}
```

**Verdict:** ✅ NO VIOLATION - Tensor provides high-level API, delegates to sparse crate for implementation

**Recommendation:** No action needed. This is the correct pattern.

---

### 2. Quantization Logic in Dtype Crate ✅ CLEAN

**Status:** CLEAN - Quantization properly extracted

**Location:** `dtype/src/lib.rs`, `dtype/src/traits.rs`

**Analysis:**
The dtype crate has been properly cleaned of quantization logic. Only minimal type-checking methods remain:

```rust
// dtype/src/lib.rs
pub const fn is_quantized(self) -> bool {
    false // No quantized types in dtype crate anymore
}

// dtype/src/traits.rs
fn is_quantized() -> bool {
    Self::dtype().is_quantized()
}
```

These are **pure type query methods** with no quantization algorithms or logic. The quantization crate has been successfully extracted (Phase 1 completed).

**Verdict:** ✅ NO VIOLATION - Dtype contains only pure type definitions

**Recommendation:** No action needed. Quantization extraction is complete.

---

### 3. Backend-Specific Code in Storage Crate ⚠️ NEEDS REVIEW

**Status:** NEEDS REVIEW - Storage contains complex operations

**Location:** `storage/src/traits.rs`, `storage/src/sparse.rs`, `storage/src/dense.rs`

**Analysis:**

#### 3.1 MatMul Operations in Storage Traits

The storage crate defines `MatMulOps` trait with matrix multiplication:

```rust
// storage/src/traits.rs
pub trait MatMulOps<T: DataType>: StorageOps<T> {
    fn matmul(&self, other: &Self, m: usize, n: usize, k: usize) -> Result<Self>;
}
```

**Issue:** Matrix multiplication is a **complex operation** that should potentially be in a higher layer (dense/sparse crates) rather than storage foundation.

**Counter-argument:** MatMul is a fundamental linear algebra operation that storage formats need to support. It's not backend-specific, but format-specific (CSR matmul differs from dense matmul).

#### 3.2 Format Conversion Operations

Storage contains extensive format conversion logic:

```rust
// storage/src/sparse.rs
impl CooStorage {
    pub fn to_csr(&self) -> CsrStorage<T> { ... }
    pub fn to_csc(&self) -> CscStorage<T> { ... }
}

impl CsrStorage {
    pub fn to_coo(&self) -> CooStorage<T> { ... }
    pub fn to_csc(&self) -> CscStorage<T> { ... }
}
```

**Issue:** These are **complex algorithms** for format conversion. Should they be in storage or sparse crate?

**Counter-argument:** Format conversions are storage-level operations that don't involve computation, just data reorganization.

#### 3.3 No Backend-Specific Code Found ✅

**Good news:** No direct backend-specific code (SIMD, CUDA, OpenCL, Metal) found in storage crate.

**Verification:**
- No `CpuBackend`, `GpuBackend`, `TpuBackend`, `NpuBackend` references
- No SIMD intrinsics
- No GPU kernel code
- No hardware-specific optimizations

**Verdict:** ✅ Storage properly abstracts away backend details

---

## Detailed Violations Summary

| Domain | Location | Severity | Status | Action Required |
|--------|----------|----------|--------|-----------------|
| Sparse in Tensor | `tensor/src/ops/sparse.rs` | None | ✅ Clean | None - proper delegation |
| Quantization in Dtype | `dtype/src/lib.rs` | None | ✅ Clean | None - already extracted |
| Backend in Storage | `storage/src/**/*.rs` | None | ✅ Clean | None - no backend code found |
| Complex Ops in Storage | `storage/src/traits.rs` | ⚠️ Low | 🔍 Review | Evaluate if MatMul should move |

---

## Architectural Questions for Review

### Question 1: Should MatMul be in Storage?

**Current State:** `MatMulOps` trait defined in storage

**Arguments FOR keeping in storage:**
- MatMul is format-specific (CSR matmul ≠ dense matmul)
- Storage needs to provide complete linear algebra primitives
- Higher layers shouldn't need to know format details

**Arguments AGAINST keeping in storage:**
- MatMul is a complex operation, not a "basic" operation
- Requirements 18.1-18.4 specify storage should have only basic operations
- Design document states storage should provide only: add, sub, mul, div, reshape, transpose, stride

**Recommendation:** ⚠️ **MOVE MatMul to dense/sparse crates** per requirements

### Question 2: Should Format Conversions be in Storage?

**Current State:** `to_csr()`, `to_csc()`, `to_coo()` methods in storage

**Arguments FOR keeping in storage:**
- Format conversions are storage-level concerns
- No computation involved, just data reorganization
- Sparse crate would need to depend on storage anyway

**Arguments AGAINST keeping in storage:**
- These are complex algorithms with non-trivial logic
- Requirements specify storage should be "basic operations only"

**Recommendation:** ✅ **KEEP in storage** - these are storage format concerns, not computation

---

## Compliance with Requirements

### Requirement 16.1: Sparse operations exclusively in sparse crate
**Status:** ✅ COMPLIANT
- Tensor crate properly delegates to sparse crate
- No sparse algorithms implemented in tensor

### Requirement 16.2: Dense operations exclusively in tensor crate
**Status:** ✅ COMPLIANT
- Dense operations properly separated
- Dense crate extracted in Phase 1

### Requirement 16.3: Backend-specific implementations in backend crate
**Status:** ✅ COMPLIANT
- No backend-specific code found in storage
- Storage properly abstracts backend details

### Requirement 16.4: Prevent cross-domain functionality leakage
**Status:** ⚠️ PARTIAL COMPLIANCE
- Most boundaries are clean
- MatMul in storage may violate "basic operations only" requirement

---

## Recommended Actions

### Priority 1: MatMul Placement Decision (Task 5.2) - USER CONSULTATION REQUIRED

**Action:** Evaluate whether `MatMulOps` should remain in storage or move to dense/sparse crates

**Rationale:** Requirements 18.1-18.4 specify storage should provide only basic operations (add, sub, mul, div, reshape, transpose, stride). MatMul is not listed as a basic operation.

**Current State:**
- Storage defines `MatMulOps` trait with `matmul()` and `matvec()` methods
- Sparse crate already has its own matmul implementations (properly separated)
- Dense crate does NOT have matmul implementations yet
- Removing MatMul from storage would be a breaking change

**Options:**
1. **Move to dense/sparse crates** (Recommended per requirements)
   - ✅ Aligns with Requirement 18.4: "SHALL NOT provide complex operations like linear transformations"
   - ✅ MatMul is a linear transformation, not a basic operation
   - ✅ Sparse crate already has matmul (no work needed)
   - ⚠️ Dense crate needs matmul implementation added
   - ⚠️ Breaking change for existing code using storage matmul
   
2. **Keep in storage** (Practical but violates requirements)
   - ❌ Violates Requirement 18.4
   - ✅ No breaking changes
   - ✅ Simpler for format-specific implementations
   
3. **Clarify requirements** (Update requirements to include MatMul)
   - ⚠️ Changes the architectural vision
   - ✅ No code changes needed

**Recommendation:** **Option 1 - Move to dense/sparse crates**
- This aligns with the stated requirements
- Sparse crate already has proper matmul implementations
- Dense crate needs matmul added (straightforward to implement)
- Breaking changes can be managed with deprecation warnings

**Implementation Plan if Option 1 chosen:**
1. Add matmul to dense crate (delegate to backend BLAS operations)
2. Deprecate MatMulOps trait in storage with warning
3. Update tensor crate to use dense/sparse matmul directly
4. Remove MatMulOps from storage in next major version

**USER DECISION NEEDED:** Which option should be pursued?

### Priority 2: Document Current Architecture (Task 5.3)

**Action:** Create clear documentation of current domain boundaries and interfaces

**Rationale:** Current architecture is mostly clean, but needs explicit documentation

### Priority 3: Create Boundary Tests (Task 5.4)

**Action:** Implement tests to prevent future violations

**Rationale:** Automated enforcement prevents regression

---

## Conclusion

The Coeus framework demonstrates **strong domain separation** with only minor questions about operation placement. The main architectural question is whether matrix multiplication should be considered a "basic" storage operation or a higher-level operation.

**Overall Assessment:** 🟢 **GOOD** - Most boundaries are clean and well-enforced

**Next Steps:**
1. Consult user on MatMul placement decision
2. Proceed with Task 5.2 based on decision
3. Document interfaces (Task 5.3)
4. Implement boundary tests (Task 5.4)
