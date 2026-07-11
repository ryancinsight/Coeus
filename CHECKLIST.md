# Coeus Development Roadmap Checklist

This document tracks the high-level roadmap and feature validation checklist for the Coeus tensor library. For detailed task lists, sprint archives, and progress tracking, see:
- [docs/checklist.md](file:///d:/coeus/docs/checklist.md)
- [docs/backlog.md](file:///d:/coeus/docs/backlog.md)

---

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

