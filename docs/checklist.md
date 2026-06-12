# Global Progress Checklist: Coeus

## Active Epic: Heterogeneous GPU Backends (wgpu & cuda-oxide)

### Current Sprint: Sprint MS-55 (Heterogeneous GPU Backends) [PLANNED]
**Objective**: Overhaul the backend trait with associated types, implement `coeus-wgpu` and `coeus-cuda` workspace crates, and resolve remaining CPU operation compilation blockers.
**Target version**: 0.2.0.

> **Roadmap (docs/backlog.md MS-60+)**: the Atlas burn-replacement program now stages
> (A2) routing the CPU backend's `BackendOps` through `coeus-leto` and deleting the
> duplicated CPU traversal — keeping `Tensor<T,B>` and the `ComputeBackend` seam; and
> (D) the GPU program: ADR to migrate `coeus-cuda` from cutile to **cuda-oxide**, finish
> wgpu op parity, consume mnemosyne device pools / melinoe device-buffer ownership.
> burn is eliminated end-to-end in Stage E.

### Current Verification Note (2026-06-12)

- [x] [patch] Added committed nextest timeout config at `.config/nextest.toml`.
- [x] [patch] Synced README verification commands to `cargo nextest run`,
  doctests, and clippy with `-D warnings`.
- [x] [patch] `coeus-cuda` now defaults to a CPU-backed no-CUDA provider so the
  full workspace can check on hosts without `CUDA_TOOLKIT_PATH`; real cutile
  CUDA integration is retained behind the explicit `cuda` feature.
- [x] [patch] The default no-CUDA `CudaBackend` implements the full
  `BackendOps` surface by delegating to the existing CPU fallback path, with
  value-semantic coverage in `coeus-cuda/tests/no_cuda_fallback.rs`.
- [x] [patch] Replaced high-arity `coeus-wgpu` attention and convolution helper
  calls with typed request structs; verified by clippy and wgpu nextest.
- [x] [patch] `cargo clippy --workspace --exclude coeus-cuda --all-targets
  -- -D warnings` passes after the `coeus-wgpu` request-struct refactor.
- [x] [patch] `cargo fmt --check` passes after workspace formatting.
- [x] [patch] `cargo check --workspace` passes without excluding `coeus-cuda`.
- [x] [patch] `cargo clippy --workspace --all-targets -- -D warnings` passes
  without excluding `coeus-cuda`.
- [x] [patch] `cargo nextest run --workspace` passes: 255 tests passed, 0
  skipped. CUDA integration tests are feature-gated under `cuda` because they
  require `CUDA_TOOLKIT_PATH` and a working CUDA driver.
- [x] [patch] `cargo test --doc --workspace` passes; four doctests are
  intentionally ignored.
- [x] [patch] Added `coeus-core/tests/dependency_policy.rs` to enforce the
  Moirai parallel/async SSOT: production sources and production manifest
  dependency sections may not import or depend on `rayon` or `tokio`. Evidence:
  `cargo test -p coeus-core --test dependency_policy` passes; normal dependency
  tree checks show no production `rayon` edge and no resolved `tokio` package.
- [x] [patch] Removed Coeus' direct `pollster` dependency from `coeus-wgpu` and
  extended `coeus-core/tests/dependency_policy.rs` to reject Coeus production
  `pollster` imports/dependencies. Evidence: `cargo test -p coeus-core --test
  dependency_policy` and `cargo tree -p coeus-wgpu --edges normal -i pollster`
  pass; the remaining resolved `pollster` edge is isolated inside
  `hephaestus-wgpu`.
- [x] [patch] Extended the dependency policy to reject direct production imports
  and direct production manifest dependencies on replacement libraries (`burn`,
  `nalgebra`, `ndarray`, `tch`) while preserving benchmark/dev-only comparisons.
  Evidence: `cargo test -p coeus-core --test dependency_policy` passes.
- [x] [patch] Expanded `coeus-leto` contract coverage for the CPU consolidation
  seam: binary dispatch covers `Sub`/`Mul`/`Div`, unary dispatch covers
  `Relu`/`Abs`/`Neg`, and keep-dim reductions cover `Sum`/`Max`/`Min`. Evidence:
  `cargo test -p coeus-leto` passes; the current contract suite contains 14 tests.
- [x] [patch] Added `coeus-ops/tests/unary_leto_diff.rs` to prove
  `SequentialBackend` and `MoiraiBackend` unary `BackendOps` dispatch matches
  direct scalar `CpuUnaryDispatch::eval_unary` for the full `CpuUnaryOp` surface.
  Evidence: `cargo test -p coeus-ops --test unary_leto_diff` passes.
- [x] [patch] Added `coeus-ops/tests/matmul_leto_diff.rs` to prove
  `SequentialBackend` and `MoiraiBackend` `BackendOps::matmul` dispatch matches
  an independent row-major triple-loop reference for contiguous and strided
  transposed input layouts. Evidence: `cargo test -p coeus-ops --test
  matmul_leto_diff` passes.
- [x] [patch] Added `coeus-ops/tests/batched_matmul_leto_diff.rs` to prove the
  public `coeus_ops::matmul` batching layer matches an independent reference on
  `SequentialBackend` and `MoiraiBackend` for equal batch counts and RHS 2-D
  broadcast. Evidence: `cargo test -p coeus-ops --test
  batched_matmul_leto_diff` passes.
- [x] [patch] Routed `coeus_ops::cumsum` and `suffix_sum` through
  `coeus-leto` scan dispatch and added value-semantic coverage in
  `coeus-leto/tests/contract.rs` plus `coeus-ops/tests/scan_leto_diff.rs`.
  Evidence: focused scan tests pass.
- [x] [patch] Added public CPU reduction differential coverage for
  `sum`/`mean`/`sum_axis`/`mean_axis`/`max_axis`/`min_axis` on
  `SequentialBackend` and `MoiraiBackend`, including transposed input views.
  Evidence: `cargo test -p coeus-ops --test public_reduction_leto_diff`
  passes.
- [x] [patch] Routed public scalar `mean` through backend
  `ReductionOp::Mean`, so CPU scalar mean now uses the `coeus-leto` mean
  reducer instead of local `sum / count` division. Evidence: `cargo test -p
  coeus-ops --test public_reduction_leto_diff` passes.
- [x] [patch] Promoted mean to `ReductionOp::Mean` and routed public
  `mean_axis` through backend reduction dispatch. CPU dispatch uses Leto
  `MeanAxis`; WGPU/CUDA generated reducers and CPU fused reductions cover the
  same variant. Evidence: focused CPU, Leto, WGPU fused, and CUDA fallback tests
  pass.
- [x] [patch] Routed public `argmax` and `argmin` through `coeus-leto`
  keep-dim arg-reduction dispatch for CPU-addressable tensors and added
  transposed-view coverage for `SequentialBackend` and `MoiraiBackend`.
  Evidence: `cargo test -p coeus-leto
  arg_reduction_dispatch_covers_keepdim_axis_ops` and `cargo test -p coeus-ops
  --test arg_reduction_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::pad` through `coeus-leto` structural
  pad dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo test -p coeus-leto pad_dispatch_covers_strided_input_view` and
  `cargo test -p coeus-ops --test pad_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::cat` through `coeus-leto` structural
  concat dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo test -p coeus-leto concat_dispatch_covers_strided_input_views` and
  `cargo test -p coeus-ops --test concat_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::split` through `coeus-leto` structural
  split dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo test -p coeus-leto split_dispatch_covers_strided_input_view` and
  `cargo test -p coeus-ops --test split_leto_diff` pass.
- [x] [patch] Routed `coeus_nn::init::{uniform_with_seed, normal_with_seed}`
  through `coeus-leto` seeded random dispatch, deleting the local Xorshift
  initializer implementation. Constructor-only `RandomScalar` bounds preserve
  forward/module surfaces for existing `Float` APIs. Evidence: `cargo test -p
  coeus-leto random_dispatch_matches_leto_seeded_constructors` and `cargo test
  -p coeus-nn --test init_leto_diff` pass.
- [x] [patch] Routed `Tensor::to_contiguous_on` for CPU-addressable storage
  through `coeus-leto` view materialization, deleting the local strided
  materialization loop from that path. Evidence: `cargo test -p coeus-leto
  contiguous_dispatch_matches_leto_view_materialization` and `cargo test -p
  coeus-tensor --test contiguous_leto_diff` pass.
- [x] [patch] Routed `Tensor::{reshape, permute}` plus `t`/`t_nd` through
  `coeus-leto` layout validation, preserving zero-copy storage sharing while
  deleting the local reshape/permute metadata duplication from the public tensor
  path. Evidence: `cargo test -p coeus-leto layout_dispatch` and `cargo test -p
  coeus-tensor --test shape_view_leto_diff` pass.
- [x] [patch] Routed non-contiguous cross-backend `Tensor::to_backend_on`
  materialization through `coeus-leto`, deleting the remaining local strided
  transfer loops from the public tensor transfer path. Evidence: `cargo test -p
  coeus-tensor --test backend_transfer_leto_diff` passes.
- [x] [patch] Routed `Tensor::from_fn_on` coordinate generation through
  `coeus-leto`, deleting the local row-major dynamic-index generation loop from
  the public tensor constructor path. Evidence: `cargo test -p coeus-leto
  shape_function_dispatch_matches_leto_coordinate_order` and `cargo test -p
  coeus-tensor --test from_fn_leto_diff` pass.
- [x] [patch] Routed `Tensor::eye_on` identity value generation through
  `coeus-leto`, deleting the local diagonal mutation loop from the public tensor
  constructor path. Fixed zero-length `CpuStorage` to expose non-null aligned
  zero-length slices for empty tensors. Evidence: `cargo test -p coeus-core
  --test cow_storage_tests` and `cargo test -p coeus-tensor --test
  identity_leto_diff` pass.
- [x] [minor] Added `Scalar::from_usize` as the native index-conversion seam
  and routed `Tensor::arange_on` through `coeus-leto`, deleting the local
  mutation loop and the constructor's f64 index conversion. Evidence: `cargo
  test -p coeus-core --test scalar_index_conversion` and `cargo test -p
  coeus-tensor --test arange_leto_diff` pass.
- [x] [patch] Routed `Tensor::linspace_on` coordinate traversal through
  `coeus-leto`, deleting the local mutable fill loop while preserving the
  existing `Scalar::from_f64` value contract. Evidence: `cargo test -p
  coeus-tensor --test linspace_leto_diff` passes.
- [x] [patch] Routed tensor broadcast shape and zero-copy broadcast layout
  validation through `coeus-leto`, deleting local dynamic broadcast metadata
  construction from `Tensor::broadcast` while preserving scalar rank-0
  broadcasts. Evidence: `cargo test -p coeus-leto
  broadcast_layout_dispatch_matches_leto_validation` and `cargo test -p
  coeus-tensor --test broadcast_leto_diff` pass.
- [x] [minor] Added public `coeus_ops::stack` through dynamic-rank
  `coeus-leto` stack dispatch, covering equal-shaped strided input views on
  `SequentialBackend` and `MoiraiBackend`. Evidence: `cargo test -p coeus-leto
  stack_dispatch_covers_strided_input_views` and `cargo test -p coeus-ops
  --test stack_leto_diff` pass.
- [x] [minor] Added `BackendOps::batched_matmul` as the batched matmul seam,
  routed public batched `coeus_ops::matmul` through it, and overrode the CPU
  `SequentialBackend`/`MoiraiBackend` path with `coeus-leto` rank-3 batched
  dispatch. Evidence: `cargo test -p coeus-leto
  batched_matmul_dispatch_covers_rhs_batch_broadcast`, `cargo test -p coeus-ops
  --test batched_matmul_leto_diff`, and `cargo test -p coeus-wgpu
  wgpu::transfers_and_matmul::test_wgpu_backend_ops_unified` pass.
- [x] [patch] Added `Scalar::{dot_slice, scale_slice}` Hermes SIMD seams and
  routed CPU forward attention contiguous Q/K row dot products plus softmax row
  scaling through them. Evidence: `cargo test -p coeus-core --test
  scalar_dot_scale` and `cargo test -p coeus-nn --test nn_attention_tests`
  pass.
- [x] [patch] Routed CPU attention backward contiguous `dO @ V^T` rows and
  softmax row products through `Scalar::dot_slice`. Evidence: `cargo test -p
  coeus-ops --test attention_backward_hermes_diff` passes.
- [x] [patch] Current full gate after Coeus direct `pollster` removal:
  `cargo fmt --check`,
  `git diff --check`, `cargo check --workspace`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo nextest run --workspace` (295 passed,
  0 skipped), and `cargo test --doc --workspace` pass.
- [x] [minor] Added Criterion baselines in `coeus-tensor/benches/tensor_bench.rs`
  for direct Leto, Coeus-Leto dispatch, `ndarray`, `nalgebra`, and Rayon slice
  elementwise add alongside Coeus Sequential and Moirai.
- [x] [patch] Added `[profile.bench]` thin LTO with one codegen unit so
  cross-crate generic kernels are benchmarked after production-grade
  monomorphization. Evidence tier: empirical Criterion measurement.
- [x] [minor] Ran a short empirical benchmark pass:
  `cargo bench -p coeus-tensor --bench tensor_bench -- --warm-up-time 1
  --measurement-time 2 --sample-size 10`. Evidence tier: empirical Criterion
  measurement. Median estimates before the bench-profile fix: 1024x1024 add,
  Coeus Sequential 1.2061 ms, Coeus Moirai 1.2963 ms, ndarray 1.0895 ms,
  nalgebra 954.33 us, Rayon slice 1.0532 ms; 256x256 matmul, Coeus Sequential
  6.8640 ms, Coeus Moirai 6.8874 ms, ndarray 595.62 us, nalgebra 585.70 us.
  Focused post-profile 256x256 matmul measurement: Coeus Sequential 1.0006 ms,
  Coeus Moirai 1.1146 ms, direct Leto 1.1012 ms, Coeus-Leto dispatch 1.0905 ms,
  ndarray 557.02 us, nalgebra 557.99 us. Rejected upstream Hermes tiled-GEMM
  route: Leto 256x256 f64 regressed to 3.6848 ms and Coeus f32 direct Leto
  regressed to 8.7577 ms; source change was removed. Dense matmul remains a
  measured optimization target with an approximate 2x gap to ndarray/nalgebra.

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
