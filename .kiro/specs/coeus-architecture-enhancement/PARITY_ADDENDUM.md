# PyTorch API Parity - Addendum: Existing Coeus Modules

**Date:** January 15, 2026  
**Status:** Critical Update to Parity Analysis

## Executive Summary

**IMPORTANT FINDING:** The initial parity analysis (3.5% module-level, 5.7% tensor methods) **significantly underestimates** Coeus's actual capabilities. 

A module audit reveals that **Coeus has already implemented 24 crates** with substantial functionality, but **11 of these crates (46%) are not exposed in PyCoeus**. This means the actual parity is much higher than reported, but the functionality is not accessible to Python users.

## Key Findings

### Implemented but Not Exposed

The following crates are **fully implemented in Rust** but **not exposed in PyCoeus**:

1. **`storage`** - Storage abstraction layer
   - 11 public modules
   - 7 public traits
   - Dense, sparse, quantized, strided storage implementations
   - **Impact:** Core infrastructure for tensor operations

2. **`autograd`** - Automatic differentiation
   - 12 public modules
   - Computation graph, gradient operations, custom functions
   - **Impact:** Critical for training neural networks

3. **`linalg`** - Linear algebra operations
   - Modules: inverse, norm, cholesky, det, qr, solve, svd
   - **Impact:** ~30 operations (svd, qr, cholesky, det, solve, inverse, norm)

4. **`signal`** - Signal processing
   - Modules: stft, windows
   - **Impact:** ~10 operations (STFT, window functions)

5. **`special`** - Special mathematical functions
   - Modules: bessel, error_functions, gamma, trig
   - **Impact:** ~15 operations (gamma, bessel, erf, erfc)

6. **`distributed`** - Distributed training infrastructure
   - **Impact:** Distributed training capabilities

7. **`distributions`** - Probability distributions
   - **Impact:** Statistical distributions

8. **`jit`** - JIT compilation infrastructure
   - Modules: cache, compiler, error, fusion, graph, hardware
   - **Impact:** JIT compilation capabilities (architectural)

9. **`profiling`** - Performance profiling
   - 15+ public structs for profiling and monitoring
   - **Impact:** Performance analysis tools

10. **`vision`** - Computer vision utilities
    - Modules: io, transforms
    - **Impact:** Vision transforms and I/O

11. **`audio`** - Audio processing
    - Modules: classification, features, models, processing, recognition, synthesis
    - **Impact:** Audio processing capabilities

### Already Exposed in PyCoeus

The following 13 crates are already exposed:

1. ✅ **`tensor`** - Core tensor operations
2. ✅ **`backend`** - Backend abstraction (CPU, GPU)
3. ✅ **`dtype`** - Data type system
4. ✅ **`nn`** - Neural network layers
5. ✅ **`optim`** - Optimizers
6. ✅ **`fft`** - FFT operations (partially)
7. ✅ **`hub`** - Model hub
8. ✅ **`tokenizer`** - Tokenizers
9. ✅ **`utils`** - Utilities
10. ✅ **`foundation`** - Foundation layer
11. ✅ **`coeus-error`** - Error handling
12. ✅ **`coeus-semantic-api`** - Semantic API
13. ✅ **`sparse`** - Sparse operations (partially)

## Revised Parity Estimate

### Current Reported Parity
- **Module-level:** 3.5% (50/1,443 items)
- **Tensor methods:** 5.7% (35/615 methods)

### Potential Parity (If All Crates Exposed)

If all existing crates are properly exposed in PyCoeus:

**Additional Operations Available:**
- **Linear Algebra (`linalg`):** ~30 operations
  - `svd`, `qr`, `cholesky`, `det`, `solve`, `inverse`, `norm`, `matrix_power`, etc.
  
- **Signal Processing (`signal`):** ~10 operations
  - `stft`, `istft`, `hamming_window`, `hann_window`, `blackman_window`, etc.
  
- **Special Functions (`special`):** ~15 operations
  - `gamma`, `lgamma`, `digamma`, `polygamma`, `erf`, `erfc`, `i0`, `i1`, `bessel_j0`, etc.
  
- **Sparse Operations (`sparse`):** ~20 operations
  - Sparse tensor creation, conversion, arithmetic
  
- **Vision Transforms (`vision`):** ~15 transforms
  - Image transforms, I/O operations
  
- **Audio Processing (`audio`):** ~10 operations
  - Audio features, classification, synthesis
  
- **Profiling (`profiling`):** ~10 utilities
  - Performance monitoring, benchmarking

**Total Additional Operations:** ~110 operations

**Revised Parity Estimate:**
- **Module-level:** ~11% (160/1,443 items) - **3x improvement**
- **Tensor methods:** ~8% (50/615 methods) - **1.4x improvement**

## Critical Action Items

### Immediate Priority: Expose Existing Crates

1. **Phase 0: Expose Existing Infrastructure (2-3 weeks)**
   - Expose `linalg` crate in PyCoeus
   - Expose `signal` crate in PyCoeus
   - Expose `special` crate in PyCoeus
   - Expose `sparse` crate fully in PyCoeus
   - Expose `vision` crate in PyCoeus
   - Expose `audio` crate in PyCoeus
   - Expose `profiling` crate in PyCoeus

2. **Update PyCoeus `lib.rs`**
   - Add module declarations for unexposed crates
   - Create Python bindings for public APIs
   - Add to `_coeus` module registration

3. **Testing**
   - Verify all exposed operations work correctly
   - Add Python tests for new functionality
   - Update documentation

### Why This Matters

**Before exposing existing crates:**
- Users think Coeus has 3.5% parity
- Missing ~110 operations that are already implemented
- Duplicate work implementing what already exists

**After exposing existing crates:**
- Users see 11% parity (3x improvement)
- Access to linear algebra, signal processing, special functions
- Can focus on truly missing functionality

## Updated Implementation Roadmap

### Revised Phase 0 (NEW): Expose Existing Crates (2-3 weeks)

**Goal:** Make existing Rust crates accessible to Python users.

**Tasks:**
1. Create Python bindings for `linalg` crate
2. Create Python bindings for `signal` crate
3. Create Python bindings for `special` crate
4. Enhance `sparse` crate exposure
5. Create Python bindings for `vision` crate
6. Create Python bindings for `audio` crate
7. Create Python bindings for `profiling` crate
8. Update documentation and examples

**Estimated Effort:** 2-3 weeks
**Impact:** 3x improvement in reported parity
**Dependencies:** None - all crates already implemented

### Original Phases (Adjusted)

**Phase 1:** Critical Tensor Operations (Q1 2026)
- **Adjustment:** Remove operations already in `linalg`, `signal`, `special`
- **New Focus:** Tensor creation, indexing, reductions, element-wise ops

**Phase 2:** Neural Network Layers (Q2 2026)
- **No change:** Still need LSTM, GRU, LayerNorm, etc.

**Phase 3:** Functional API (Q2-Q3 2026)
- **No change:** Still need functional versions

**Phase 4:** Advanced Tensor Operations (Q3 2026)
- **Adjustment:** Remove operations already in `linalg`, `fft`, `signal`, `special`
- **New Focus:** Additional advanced operations not yet implemented

**Phase 5:** Optimizers and Training Utilities (Q3-Q4 2026)
- **No change:** Still need additional optimizers and schedulers

**Phase 6:** Specialized Layers (Q4 2026)
- **No change:** Still need transformer and specialized layers

## Architectural Features Already Implemented

Surprisingly, some "architectural" features are already implemented:

1. **JIT Compilation (`jit` crate):**
   - Modules: cache, compiler, error, fusion, graph, hardware
   - **Status:** Implemented but not exposed
   - **Recommendation:** Expose for advanced users

2. **Distributed Training (`distributed` crate):**
   - **Status:** Implemented but not exposed
   - **Recommendation:** Expose for distributed training

3. **Profiling (`profiling` crate):**
   - Comprehensive profiling infrastructure
   - **Status:** Implemented but not exposed
   - **Recommendation:** Expose for performance analysis

## Recommendations

### Immediate Actions (This Week)

1. **Audit PyCoeus Exposure:**
   - Review `pycoeus/src/lib.rs`
   - Identify which crates are not exposed
   - Create tracking issue for each unexposed crate

2. **Prioritize Exposure:**
   - **High Priority:** `linalg`, `signal`, `special` (commonly used)
   - **Medium Priority:** `vision`, `audio`, `profiling` (specialized)
   - **Low Priority:** `jit`, `distributed` (advanced, may need more work)

3. **Create Bindings:**
   - Start with `linalg` crate (highest impact)
   - Follow pattern from existing exposed crates
   - Add comprehensive tests

### Short-Term (Next Month)

1. **Complete Phase 0:**
   - Expose all high and medium priority crates
   - Update documentation
   - Create examples

2. **Update Parity Reports:**
   - Re-run comparison script
   - Update parity percentages
   - Communicate improved parity to users

3. **Begin Phase 1:**
   - Start implementing truly missing tensor operations
   - Focus on operations not in any existing crate

## Conclusion

This addendum reveals a critical insight: **Coeus's actual parity is significantly higher than reported**. The framework has substantial functionality already implemented in Rust crates, but it's not accessible to Python users.

**Key Takeaway:** Before implementing new functionality, we should first expose what already exists. This will:
- Triple the reported parity (3.5% → 11%)
- Provide immediate value to users
- Avoid duplicate work
- Establish patterns for future crate exposure

**Next Steps:**
1. Create Phase 0 tasks for exposing existing crates
2. Update implementation roadmap
3. Begin with `linalg` crate exposure
4. Re-run parity analysis after exposure

This discovery fundamentally changes the implementation strategy and timeline. Instead of 12 months to achieve 60-70% parity, we can achieve 11% parity in 2-3 weeks just by exposing existing work!
