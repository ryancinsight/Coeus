# Coeus Module Audit Report

This report audits what Coeus has implemented in its crates and whether they're exposed in PyCoeus.

**Exposed in PyCoeus:** 13
**Not Exposed in PyCoeus:** 11

## Summary

**Total Crates:** 24

## Core Infrastructure

### tensor - ✅ Exposed in PyCoeus

**Public Modules (11):**
- `ops`
- `functions`
- `minimal_tensor`
- `tensor_backend_dispatch`
- `tensor_core`
- `implementations`
- `elementwise`
- `error`
- `indexing`
- `shape_ops`
- ... and 1 more

**Public Functions (1):**
- `grad_rwlock`

---

### storage - ❌ Not exposed in PyCoeus

**Public Modules (11):**
- `error`
- `iter`
- `shape`
- `traits`
- `dense`
- `quantized`
- `strided`
- `sparse`
- `sparse_indexing`
- `broadcast`
- ... and 1 more

**Public Traits (7):**
- `AsAny`
- `ActivationOps`
- `StorageFromVec`
- `StorageToDense`
- `MatMulStorage`
- ... and 2 more

---

### backend - ✅ Exposed in PyCoeus

**Public Modules (5):**
- `cpu`
- `device`
- `gpu`
- `distributed`
- `memory_integration`

**Public Structs (10):**
- `WorkloadCharacteristics`
- `PerformanceMetrics`
- `BackendSelector`
- `BackendDispatchStats`
- `PerformanceMonitor`
- `PerformanceSummary`
- `StubBackend`
- `MemoryManager`
- `MemoryAnalysisResult`
- `StubDevice`

**Public Traits (2):**
- `AdaptiveBackendDispatch`
- `Backend`

---

### dtype - ✅ Exposed in PyCoeus

**Public Modules (8):**
- `error`
- `traits`
- `complex`
- `float`
- `int`
- `quantized`
- `promotion`
- `quantization`

---

### autograd - ❌ Not exposed in PyCoeus

**Public Modules (12):**
- `checkpointing`
- `computation_graph`
- `custom`
- `functions`
- `graph_node`
- `loss`
- `nn`
- `numerical`
- `ops`
- `sparse_gradients`
- ... and 2 more

---

## Neural Networks

### nn - ✅ Exposed in PyCoeus

**Public Modules (14):**
- `containers`
- `functional`
- `io`
- `modules`
- `research`
- `training`
- `clip`
- `datasets`
- `evaluation`
- `multimodal`
- ... and 4 more

---

### optim - ✅ Exposed in PyCoeus

**Public Modules (15):**
- `adadelta`
- `adagrad`
- `adam`
- `adamax`
- `adamw`
- `error`
- `gpu_backend`
- `nadam`
- `optimizer`
- `optimizer_core`
- ... and 5 more

---

## Specialized Math

### linalg - ✅ Exposed in PyCoeus

**Public Modules (8):**
- `error`
- `inverse`
- `norm`
- `cholesky`
- `det`
- `qr`
- `solve`
- `svd`

---

### fft - ✅ Exposed in PyCoeus

**Public Modules (2):**
- `cpu`
- `gpu`

---

### signal - ✅ Exposed in PyCoeus

**Public Modules (2):**
- `stft`
- `windows`

---

### special - ✅ Exposed in PyCoeus

**Public Modules (4):**
- `bessel`
- `error_functions`
- `gamma`
- `trig`

---

### sparse - ✅ Exposed in PyCoeus

**Public Modules (2):**
- `cpu`
- `gpu`

---

## Advanced Features

### distributed - ❌ Not exposed in PyCoeus

**Public Modules (6):**
- `communication`
- `data_parallel`
- `error`
- `optimizer`
- `process_group`
- `reducer`

---

### distributions - ❌ Not exposed in PyCoeus

**Public Modules (2):**
- `error`
- `parameter`

---

### jit - ❌ Not exposed in PyCoeus

**Public Modules (12):**
- `cache`
- `compiler`
- `error`
- `fusion`
- `graph`
- `hardware`
- `memory`
- `optimizer`
- `shapes`
- `simd`
- ... and 2 more

---

### profiling - ❌ Not exposed in PyCoeus

**Public Modules (1):**
- `training_monitor`

**Public Structs (25):**
- `Timer`
- `ProfileStats`
- `MemoryStats`
- `PerformanceProfile`
- `MemoryDelta`
- `Profiler`
- `ScopedTimer`
- `PerformanceEvent`
- `PerformanceSubscriber`
- `PerformanceReport`
- ... and 15 more

---

## Utilities

### hub - ✅ Exposed in PyCoeus

**Public Modules (6):**
- `cache`
- `error`
- `loader`
- `models`
- `registry`
- `validator`

**Public Structs (1):**
- `Hub`

---

### tokenizer - ✅ Exposed in PyCoeus

**Public Modules (8):**
- `encoding`
- `error`
- `post_processor`
- `pre_tokenizer`
- `vocabulary`
- `bpe`
- `sentencepiece`
- `wordpiece`

**Public Structs (1):**
- `PyTorchBatchEncoding`

**Public Traits (3):**
- `Tokenizer`
- `BatchTokenizer`
- `PyTorchTokenizer`

---

### vision - ❌ Not exposed in PyCoeus

**Public Modules (2):**
- `io`
- `transforms`

---

### audio - ❌ Not exposed in PyCoeus

**Public Modules (7):**
- `classification`
- `error`
- `features`
- `models`
- `processing`
- `recognition`
- `synthesis`

---

### utils - ✅ Exposed in PyCoeus

**Public Modules (5):**
- `dataloader`
- `dataset`
- `error`
- `sampler`
- `transforms`

---

## Foundation

### foundation - ❌ Not exposed in PyCoeus

**Public Modules (9):**
- `error`
- `trainer`
- `training`
- `transformers`
- `distributed`
- `memory`
- `optimization`
- `monitoring`
- `data`

---

### coeus-error - ❌ Not exposed in PyCoeus

---

### coeus-semantic-api - ❌ Not exposed in PyCoeus

**Public Modules (6):**
- `clip_service`
- `custom_middleware`
- `errors`
- `handlers`
- `state`
- `types`

**Public Functions (3):**
- `create_router`
- `init_tracing`
- `init_metrics`

---

## Recommendations

### High Priority: Expose Existing Crates

The following crates are implemented but not exposed in PyCoeus:

- **audio**: 7 public APIs available
- **autograd**: 12 public APIs available
- **coeus-error**: 11 public APIs available
- **coeus-semantic-api**: 9 public APIs available
- **distributed**: 6 public APIs available
- **distributions**: 2 public APIs available
- **foundation**: 9 public APIs available
- **jit**: 12 public APIs available
- **profiling**: 26 public APIs available
- **storage**: 18 public APIs available
- **vision**: 2 public APIs available

### Action Items

1. **Expose `linalg` crate**: Linear algebra operations (svd, qr, cholesky, etc.)
2. **Expose `signal` crate**: Signal processing (STFT, windows)
3. **Expose `special` crate**: Special functions (gamma, bessel, erf)
4. **Expose `sparse` crate**: Sparse tensor operations
5. **Expose `distributed` crate**: Distributed training (if implemented)
6. **Expose `distributions` crate**: Probability distributions
7. **Expose `vision` crate**: Vision transforms and utilities
8. **Expose `audio` crate**: Audio processing
9. **Expose `profiling` crate**: Performance profiling tools

### Updated Parity Estimate

If all existing crates are properly exposed in PyCoeus, the actual parity would be significantly higher than the current 3.5% module-level and 5.7% tensor method parity.

**Estimated Impact:**
- `linalg` crate: +30 operations (svd, qr, cholesky, det, solve, etc.)
- `fft` crate: +8 operations (already partially exposed)
- `signal` crate: +10 operations (stft, windows)
- `special` crate: +15 operations (gamma, bessel, erf)
- `sparse` crate: +20 operations (sparse tensor ops)
- `vision` crate: +15 transforms
- `audio` crate: +10 operations

**Total Potential Gain:** ~100+ operations just by exposing existing crates!
