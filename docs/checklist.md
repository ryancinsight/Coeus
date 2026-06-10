# Global Progress Checklist: Coeus

## Active Epic: Heterogeneous GPU Backends (wgpu & cuda-oxide)

### Current Sprint: Sprint MS-55 (Heterogeneous GPU Backends) [PLANNED]
**Objective**: Overhaul the backend trait with associated types, implement `coeus-wgpu` and `coeus-cuda` workspace crates, and resolve remaining CPU operation compilation blockers.

> **Roadmap (docs/backlog.md MS-60+)**: the Atlas burn-replacement program now stages
> (A2) routing the CPU backend's `BackendOps` through `coeus-leto` and deleting the
> duplicated CPU traversal — keeping `Tensor<T,B>` and the `ComputeBackend` seam; and
> (D) the GPU program: ADR to migrate `coeus-cuda` from cutile to **cuda-oxide**, finish
> wgpu op parity, consume mnemosyne device pools / melinoe device-buffer ownership.
> burn is eliminated end-to-end in Stage E.

---

### Workspace Crate Status Matrix

| Crate Name | Path | Primary Responsibilities | Compilation Status | Notes / Blockers |
| :--- | :--- | :--- | :--- | :--- |
| **coeus-core** | [coeus-core](file:///d:/coeus/coeus-core) | Scalar types, layouts, storage traits, backend traits, CPU backends | ✅ Compiles | Clean compilation |
| **coeus-tensor** | [coeus-tensor](file:///d:/coeus/coeus-tensor) | N-dimensional strided tensor representation (`Tensor<T, B, S>`) | ✅ Compiles | Clean compilation |
| **coeus-ops** | [coeus-ops](file:///d:/coeus/coeus-ops) | Element-wise math, matrix operations, reductions, Apollo FFT | ✅ Compiles | Zero-copy layout traversal and thread-safe |
| **coeus-autograd** | [coeus-autograd](file:///d:/coeus/coeus-autograd) | Tape-based automatic differentiation (`Var`, `Tape`) | ✅ Compiles | Clean compilation |
| **coeus-optim** | [coeus-optim](file:///d:/coeus/coeus-optim) | Optimizers (SGD, Adam, RMSProp) | ✅ Compiles | Borrow checker conflicts resolved |
| **coeus-nn** | [coeus-nn](file:///d:/coeus/coeus-nn) | Neural network modules (Linear, activations, losses, normalization) | ✅ Compiles | Clean compilation |
| **coeus-sparse** | [coeus-sparse](file:///d:/coeus/coeus-sparse) | Sparse format storage primitives (COO, CSR) | ✅ Compiles | Clean compilation |
| **coeus-python** | [coeus-python](file:///d:/coeus/coeus-python) | Thin PyO3 bindings | ✅ Compiles | Clean compilation |

---

## Action Items Checklist

### Phase 1: Core CPU Workspace Stabilization [COMPLETE]
Resolve the remaining compiler and thread-safety blockers in `coeus-ops` and `coeus-optim`.

- [x] **coeus-optim Fixes**:
  - [x] Resolve `E0502` borrow conflicts in SGD step loop by caching `.numel()` before borrowing `param.tensor` mutably.
  - [x] Resolve `E0502` borrow conflicts in Adam step loop.
  - [x] Resolve `E0502` borrow conflicts in RMSProp step loop.
- [x] **coeus-ops Thread-Safety Fixes**:
  - [x] Implement `SendPtr<T>` and `SendPtrMut<T>` wrapper types in `coeus-ops` implementing `Send + Sync` to wrap raw pointers (`*const T` / `*mut T`).
  - [x] Refactor `coeus-ops/src/matmul/kernel.rs` to pass `SendPtr`/`SendPtrMut` into the `parallel_for` closure.
  - [x] Refactor `coeus-ops/src/reduction/sum.rs` to pass `SendPtr`/`SendPtrMut` into the `parallel_for` closure.
- [x] **Workspace Verification**:
  - [x] Run `cargo check --workspace` to verify zero compilation errors.
  - [x] Run `cargo test --workspace` to verify that all existing CPU tests pass.

### Phase 2: Parity Testing & Performance Benchmarks [COMPLETE]
Establish numerical validation, autograd equivalence, and performance measurements against baseline libraries.

- [x] **Numerical Parity Validation**:
  - [x] Implement `coeus-tensor/tests/parity_tests.rs` comparing Coeus tensor operations against `ndarray` outputs.
  - [x] Verify exact parity across various strides, shapes, and layouts.
- [x] **Autograd Parity & Design Equivalence**:
  - [x] Implement autograd validation tests and verify gradient correctness.
- [x] **Performance Benchmarks**:
  - [x] Configure `criterion` benchmarks in `coeus-tensor/benches/tensor_bench.rs` comparing Sequential/Moirai backends against `ndarray`.

### Phase 3: GPU Integration Abstractions (Sprint MS-55 Phase 1) [COMPLETE]
Introduce generic, zero-cost associated-type abstractions to support device-specific execution.

- [x] **Associated-Type Backend Overhaul**:
  - [x] Define the `ComputeBackend` trait in `coeus-core::backend` to replace/extend `Backend`.
  - [x] Define associated types:
    - `type DeviceBuffer<T>` (device memory allocation type representing device-allocated storage).
    - `type KernelDescriptor` (information needed to run kernels on the backend).
    - `type DispatchFuture<T>` (async execution handle for non-blocking queue operations).
- [x] **Unified Storage Design**:
  - [x] Update `Storage<T>` and `StorageMut<T>` traits to support non-CPU-accessible buffers (preventing direct CPU slicing assumptions).
- [x] **Device Transfer API**:
  - [x] Implement `Tensor::to_backend::<NewB>(&self, backend: &NewB)` on `Tensor` to handle copying between host/device memory boundaries (CPU-to-GPU, GPU-to-CPU, GPU-to-GPU).

### Phase 4: WebGPU Backend Crate (coeus-wgpu) [COMPLETE]
Implement a cross-platform GPU backend powered by `wgpu`.

- [x] **Crate Structure**:
  - [x] Create the new `coeus-wgpu` crate and add it to the workspace members.
  - [x] Add `wgpu` and `bytemuck` dependencies in `coeus-wgpu/Cargo.toml`.
- [x] **Storage & Memory Management**:
  - [x] Implement `WgpuStorage<T>` wrapping a `wgpu::Buffer` handle with unified allocation and staging logic.
- [x] **Compute Pipelines**:
  - [x] Write WGSL (WebGPU Shading Language) compute shaders for element-wise unary operations, binary operations, matrix multiplication, sum reductions, and Conv1D/Conv2D forward and backward passes.
  - [x] Implement compilation cache for `wgpu::ComputePipeline` instances.
- [x] **Backend Trait implementation**:
  - [x] Implement `ComputeBackend` for `WgpuBackend` mapping operations to GPU shader dispatch.
  - [x] Implement `BackendOps` for `WgpuBackend` to route math operations.

### Phase 5: CUDA Backend Crate (coeus-cuda) [COMPLETE]
Implement an optimized native NVIDIA GPU backend dynamically loading the CUDA driver.

- [x] **Crate Structure**:
  - [x] Create the new `coeus-cuda` crate in the workspace.
  - [x] Add dynamic CUDA driver bindings (`cuInit`, `cuMemcpy`, etc.) using `libloading` for robust cross-host compilation.
- [x] **Storage & Context Management**:
  - [x] Implement `CudaStorage<T>` wrapping CUDA `CUdeviceptr` raw handles.
  - [x] Integrate thread-safe CUDA context management and memory allocation.
- [x] **Kernel Management**:
  - [x] Implement embedded PTX source containing element-wise, tiled matrix multiplication, sum reduction, Conv1D/Conv2D forward and backward passes, and strided/broadcasted element-wise operations.
  - [x] Implement direct driver dispatch launchers and remove CPU staging fallbacks for device execution.


---

## Success Criteria & Definition of Done

- [x] **Clean Build**: `cargo check --workspace` passes with zero compiler errors.
- [x] **Zero-Cost Dispatch Verified**: Monomorphization checked; calling operations on specific backends resolves to direct static execution paths with zero abstraction overhead.
- [x] **Numerical Parity**: Parity tests verify absolute numerical equivalence between CPU, `wgpu`, and `cuda` execution (absolute tolerance $\le 10^{-5}$).
- [x] **Test Coverage**: 100% of tensor operations have verification tests covering contiguous, non-contiguous, broadcasted, and sliced tensor views.
- [x] **Memory Safety**: No memory leaks or data races on GPU backends under parallel operations (RAII wrapper verification).
