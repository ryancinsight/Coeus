# Coeus Development Roadmap Checklist

## Backend-generic host extraction [minor]
- [x] Materialize tensor views through the selected backend's host-copy seam.
- [x] Preserve logical row-major values for offset and strided layouts.
- [x] Verify 58/58 tensor tests and warning-denied Clippy.

## Coeus-ops hierarchical integration harness [patch]
- [x] Move the 36 flat `coeus-ops` integration-test files into ten
      operation-family directories under `coeus-ops/tests/ops/`.
- [x] Add one `coeus-ops/tests/ops.rs` integration target with explicit module
      manifests; preserve every leaf test body and assertion.
- [x] Verify the target census reduced from 36 integration binaries to 1,
      the harness exposes 87 integration tests, and package Nextest passes
      196/196 with warning-denied Clippy.

Evidence: `cargo metadata --locked --no-deps` reports one `coeus-ops` test
target (`ops`); `cargo nextest list --locked -p coeus-ops --all-features`
reports 87 `ops` tests; the exact package run passes 196/196. This is a test
topology and build-artifact change only; production operation code is
unchanged.

## Coeus-NN hierarchical integration harness [patch]
- [x] Move the 33 flat `coeus-nn/tests/*.rs` leaf files into ten
      operation-family directories under `coeus-nn/tests/nn_ops/`.
- [x] Use one `coeus-nn/tests/nn_ops.rs` integration target for the established
      `tests/nn/` module tree and the operation-family modules.
- [x] Verify the target census reduced from 2 to 1 and the exact package
      Nextest run passes 268/268 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `nn_ops` integration target and the
`nn_bench` benchmark target; the exact package run reports 268/268 tests with
0 skipped in 4.463 seconds. Package check, warning-denied Clippy, format, and
diff checks pass. The established NN module tree and all operation-family test
bodies remain unchanged. This is a test topology and build-artifact change
only; it does not claim a production NN speedup, memory reduction, or
whole-workspace debug-tree size delta.

## Coeus-autograd hierarchical integration harness [patch]
- [x] Move `grid_sample_3d.rs`, `linear_interpolation.rs`, and
      `selective_scan.rs` into three operation-family directories under
      `coeus-autograd/tests/autograd_ops/`.
- [x] Use one `coeus-autograd/tests/autograd_ops.rs` target for the established
      `tests/autograd/` tree and the standalone operation-family modules.
- [x] Verify the target census reduced from 2 to 1 and the exact package
      Nextest run passes 94/94 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `autograd_ops` integration target;
the exact package run reports 94/94 tests with 0 skipped in 1.535 seconds.
Package check, warning-denied Clippy, format, and diff checks pass. The existing
autograd module tree and all standalone operation-family test bodies are
unchanged. This is a test topology and build-artifact change only; it does not
claim a production autograd speedup, memory reduction, or whole-workspace
debug-tree size delta.

## Coeus-tensor hierarchical integration harness [patch]
- [x] Move the 13 flat `coeus-tensor/tests/*.rs` leaf files into six
      operation-family directories under `coeus-tensor/tests/tensor_ops/`.
- [x] Add one `coeus-tensor/tests/tensor_ops.rs` integration target with
      explicit backend, checkpoint, constructor, layout, operation, and
      property manifests; preserve every leaf test body and assertion.
- [x] Verify the target census reduced from 13 to 1 and the exact package
      Nextest run passes 58/58 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `tensor_ops` integration target;
the source census remains 53 annotated integration tests and the exact package
run reports 58/58 tests with 0 skipped, including five library unit tests.
Production tensor code is unchanged. This is a test topology and build-artifact
change only; it does not claim a whole-workspace debug-tree size reduction.

## Coeus-sparse hierarchical integration harness [patch]
- [x] Move the three flat `coeus-sparse/tests/*.rs` leaf files into conversion,
      differential, and invariant directories under `tests/sparse_ops/`.
- [x] Add one `coeus-sparse/tests/sparse_ops.rs` integration target with
      explicit operation-family manifests; preserve every leaf test body and
      value-semantic assertion.
- [x] Verify the target census reduced from 3 to 1 and the exact package
      Nextest run passes 19/19 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `sparse_ops` integration target;
the exact package run reports 19/19 tests with 0 skipped in 0.713 seconds.
Production sparse code is unchanged. This is a test topology and build-artifact
change only; it does not claim a whole-workspace debug-tree size reduction.

## Coeus-core hierarchical integration harness [patch]
- [x] Move the four flat `coeus-core/tests/*.rs` leaf files into storage,
      dependency-policy, and scalar directories under `tests/core_ops/`.
- [x] Add one `coeus-core/tests/core_ops.rs` integration target with explicit
      operation-family manifests; preserve every leaf assertion and retain the
      seven existing library unit tests in `src`.
- [x] Verify the target census reduced from 4 to 1 and the exact package
      Nextest run passes 21/21 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `core_ops` integration target; the
exact package run reports 21/21 tests with 0 skipped, comprising 14 integration
cases and seven unchanged library unit tests. Production core code is unchanged.
This is a test topology and build-artifact change only; it does not claim a
whole-workspace debug-tree size reduction.

## Coeus-CUDA feature-aware integration harness [patch]
- [x] Move the three flat CUDA test files into device and fallback directories
      under `coeus-cuda/tests/cuda_ops/` and add one `cuda_ops.rs` harness.
- [x] Preserve the `cuda` and `not(feature = "cuda")` gates, including the
      existing nested `tests/cuda/` module tree used by `cuda_tests.rs`.
- [x] Verify metadata reports one integration target; default Nextest passes
      3/3, and all-features check plus warning-denied Clippy pass.

Evidence: exact default package Nextest passes 3/3 with 0 skipped in 0.053
seconds. All-features execution is not claimable on this host: the linker
fails to find `/usr/local/cuda-11.3/lib64/libcuda`; this is an environment
linker dependency, not a test assertion failure.

## Coeus-Python hierarchical integration harness [patch]
- [x] Move the six flat `coeus-python/tests/*.rs` leaf files into activation,
      distributed, NN, operation, optimizer, and autodiff directories under
      `tests/binding_ops/`.
- [x] Add one `coeus-python/tests/binding_ops.rs` target with a single shared
      `tests/common` lock module; preserve every binding assertion and Python
      parity file.
- [x] Verify the target census reduced from 6 to 1 and the exact package
      Nextest run passes 75/75 with warning-denied Clippy and package check.

Evidence: locked Cargo metadata reports one `binding_ops` integration target;
the exact all-features package run reports 75/75 tests with 0 skipped in
6.585 seconds. Production PyO3 and Python parity code is unchanged. This is a
test topology and build-artifact change only; it does not claim Python wheel or
external-interpreter coverage.

## Coeus-WGPU hierarchical integration harness [patch]
- [x] Move the two flat `coeus-wgpu/tests/*.rs` integration targets under one
      `tests/wgpu_ops.rs` harness with `fusion` and `backend/wgpu` ownership.
- [x] Preserve the existing backend operation modules and every WGPU/fused
      assertion; no production kernel, fixture, or tolerance changed.
- [x] Verify locked metadata reports one integration target and the exact
      package Nextest run remains 85/85 with zero skipped.

Evidence: the moved source files are content-identical renames; locked Cargo
metadata reports one `wgpu_ops` integration target instead of two. The exact
package Nextest run passes 85/85 with zero skipped in 84.155 seconds, including
the three unchanged library tests. Package check, warning-denied Clippy, format,
and diff checks pass. The remaining `backend/wgpu/parity.rs` file is 808 lines
and spans multiple operation families; its family split is a separate claimed
follow-up, not part of this target-count relocation.

## Coeus-WGPU parity-family split [patch]
- [x] Split the 808-line parity leaf into a shared oracle manifest plus
      elementwise, reduction, matmul, convolution/pooling, optimizer, and
      strided operation-family modules.
- [x] Preserve every parity test name, generated unary test, tolerance, and
      CPU/GPU differential assertion; production kernels and fixtures remain
      unchanged.
- [x] Keep each parity leaf below 500 lines and rerun format, check, Clippy,
      and package Nextest.

Evidence: the pre/post source-name census remains 47 unique parity identifiers;
locked package Nextest passes 85/85 with zero skipped in 80.113 seconds. The
largest new parity leaf is `elementwise.rs` at 287 lines; all seven leaves are
below 500 lines. Package check and warning-denied Clippy pass after removing two
unused imports exposed by the split.

## Coeus-Leto hierarchical integration harness [patch]
- [x] Move `contract.rs` and `sparse_dispatch.rs` under one
      `coeus-leto/tests/leto_ops.rs` harness with contract and sparse-dispatch
      operation-family modules.
- [x] Preserve all 28 listed integration tests and their cross-provider
      contract assertions; production APIs, fixtures, and tolerances remain
      unchanged.
- [x] Verify locked metadata reports one integration target and exact package
      Nextest preserves the full test count with warning-denied Clippy.

Evidence: locked metadata reports one `leto_ops` integration target instead of
two. The exact package Nextest run passes 28/28 with zero skipped in 1.064
seconds. Package check, warning-denied Clippy, format, and diff checks pass.
The live census corrected the prior 26-test tracking claim: `contract` contains
26 tests and `sparse_dispatch` contains 2. This is a test topology and
maintainability change only; it does not claim a production-kernel speedup,
memory reduction, or whole-workspace debug-tree delta.

This document tracks the high-level roadmap and feature validation checklist for the Coeus tensor library. For detailed task lists, sprint archives, and progress tracking, see:
- [docs/checklist.md](file:///d:/coeus/docs/checklist.md)
- [docs/backlog.md](file:///d:/coeus/docs/backlog.md)

---

## Named Optimizer Ownership [major]
- [x] Move the canonical `Parameter` carrier below NN and optimizer consumers.
- [x] Make every optimizer retain stable names through steps and gradient clipping.
- [x] Validate complete path inventories before loading updated values into modules.
- [x] Require explicit `(name, tensor)` pairs at the PyO3 boundary.
- [x] Verify optimizer nextest 20/20, cross-boundary nextest 21/21,
      affected NN parity 144/144, Clippy, Rustdoc, and doctests.

## RITK Stable Named Parameters [minor]
- [x] Add canonical named parameter collection to the `Module` seam.
- [x] Preserve optimizer ordering while assigning semantic leaf names.
- [x] Prefix dynamic/static containers, recurrent modules, and transformer trees hierarchically.
- [x] Verify exact decoder paths, full-transformer uniqueness, shared gradient identity, nextest 410/410, and warning-denied Clippy.

## RITK Dimension-Complete Interpolation [minor]
- [x] Replace dimension-specific entry points with one const-dimension linear interpolation family.
- [x] Encode replicated-border behavior as a sealed zero-sized policy.
- [x] Provide shared 2-D/3-D forward and reverse-mode kernels without hot-loop allocation.
- [x] Verify analytical values/gradients, every coordinate by central difference,
      malformed contracts, Sequential/Moirai agreement, and affected nextest 282/282.

## RITK Bounded Archived State Support [minor]
- [x] Replace the eager bespoke `StateDict` format with validated rkyv archives.
- [x] Expose zero-copy borrowed tensor names, shapes, and payload bytes before materialization.
- [x] Enforce archive/tensor/name/rank/payload limits, scalar identity, byte order, duplicate-name rejection, and deterministic ordering.
- [x] Verify package nextest 56/56, warning-denied Clippy, Rustdoc, and doctests.

## RITK VMamba Depthwise Convolution Support [minor]
- [x] Add a canonical Coeus depthwise 3-D convolution module with one learned kernel per channel.
- [x] Preserve input, kernel, and bias reverse-mode paths without a consumer-owned grouped-convolution adapter.
- [x] Verify exact two-channel values and analytical input gradients under nextest.

## RITK Attention Matmul Support [patch]
- [x] Preserve rank-N logical batch axes across batched matmul.
- [x] Dispatch backward accumulation through an explicit rank-3 kernel layout.
- [x] Verify exact rank-4 values and both operand gradients, affected nextest
      689/689, and warning-denied Clippy.

## RITK TransMorph Provider Support [minor]
- [x] Generalized `coeus_nn::Linear` to project the final feature axis of
      rank-2 and higher-rank inputs through one autograd-preserving path.
- [x] Verified exact rank-3/rank-5 forward values and rank-3
      input/weight/bias gradients, all 409 `coeus-nn` tests,
      warning-denied Clippy, and rustdoc.

## Default Parallel Memory Features [patch]
- [x] Restored Moirai default features for workspace consumers so parallel execution, Mnemosyne-backed memory surfaces, and Mellinoe branding are active by default.
- [x] Restored Leto and Leto Ops defaults so Coeus consumes Mnemosyne-backed Leto storage and default parallel ops without local feature suppression.
- [x] Verification: `cargo check --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo nextest run --workspace`, and `cargo test --doc --workspace` pass in the default no-CUDA-provider configuration. Real cutile CUDA tests remain under `coeus-cuda --features cuda`.

## 🟩 Phase 1: CPU Workspace Stabilization (100% Complete)
Established a clean, warning-free compiler baseline and resolved lifetime and borrow checker conflicts across all workspace crates.

- [x] **Zero-Copy Layout Traversal**: Refactored `coeus-ops` kernels (unary, binary, matmul, sum/mean reductions, SpMV, SpMM) to perform direct strided index math without calling `to_contiguous()`.
- [x] **Thread-Safe Pointers**: Created `SendPtr` and `SendPtrMut` raw pointer wrappers in `coeus-ops/src/ptr.rs` to allow safe, thread-safe capture of raw pointers in `Moirai` parallel closures.
- [x] **Apollo FFT Integration**: Decoupled FFT operations from coeus crates, supporting them directly inside the `apollo-fft` crate in the `apollo` workspace.
- [x] **Autograd & Optimizers**: Resolved lifetime and borrow checker errors (SGD, Adam, RMSProp step loops, and LayerNorm/BatchNorm backward closures).
- [x] **Numerical Parity Validation**: Verified mathematical outputs (relu, matmul, reductions, FFT, sparse operations) against `ndarray` and PyTorch references inside `coeus-tensor/tests/parity_tests.rs`.
- [x] **Autodiff PyTorch Comparison**: Implemented integration benchmarks in `coeus-python/tests/autodiff_comparison.rs` verifying 100% mathematical gradient parity (X, weight, and bias gradients) and measuring step time comparison.

---

## 🟩 Phase 2: Associated-Type Backend Overhaul (GPU Abstractions) (100% Complete)
Evolve the backend trait to support heterogeneous device compilation (CPU and GPU backends) with zero abstraction overhead.

- [x] **ComputeBackend Trait**: Define a new generalized backend trait containing associated types:
  - `type DeviceBuffer<T>`: Handle for device-allocated memory buffers.
  - `type KernelDescriptor`: Configuration and dispatch params for compute pipelines.
  - `type DispatchFuture<T>`: Non-blocking future wrapping GPU execution handles.
- [x] **Unified Memory Management**: Update the `Storage` and `StorageMut` traits to support non-CPU-addressable memory buffers.
- [x] **Explicit Memory Transfers**: Add explicit copy APIs (`Tensor::to_backend::<NewB>(&self, backend: &NewB)`) to handle CPU-to-GPU, GPU-to-CPU, and GPU-to-GPU memory boundaries.

---

## 🟩 Phase 3: WebGPU Backend Crate (coeus-wgpu) (100% Complete)
Implement a cross-platform GPU backend powered by `wgpu`.

- [x] **Crate Setup**: Create `coeus-wgpu` and add it to the workspace members.
- [x] **Device Memory Wrapper**: Implement `WgpuStorage<T>` managing raw GPU buffer allocations.
- [x] **WGSL Compute Kernels**: Write WebGPU Shading Language compute shaders for element-wise ops, matrix multiplication, sum reductions, and Conv1D/Conv2D forward and backward passes.
- [x] **Compilation Caching**: Implement automated compute pipeline caching using `wgpu::ComputePipeline`.

---

## 🟪 Phase 4: CUDA Backend Crate (coeus-cuda) (100% Complete)
Implemented a native NVIDIA GPU backend dynamically loading the CUDA driver.

- [x] **Crate Setup**: Created `coeus-cuda` and added it to the workspace members.
- [x] **Context Management**: Dynamically binds to the native CUDA Driver API (`cuInit`, `cuCtxCreate`, etc.) via `libloading` at runtime.
- [x] **Staging Memory Transfers**: Implemented FFI copies between host CPU and device GPU memory (`cuMemcpyHtoD`, `cuMemcpyDtoH`).
- [x] **Thread-Safe Context**: Implemented a thread-safe static wrapper for the loaded context pointer.
- [x] **Hephaestus CUDA Substrate**: Allocations and shared primitive
  contiguous elementwise kernels route through `hephaestus-cuda`, matching the
  WGPU provider boundary; Coeus-local CUDA kernels remain for aliasing,
  strided/dynamic layout coverage, and NN-specific convolution, pooling,
  optimizer, and activation formulas.

