# Coeus Development Roadmap Checklist

This document tracks the high-level roadmap and feature validation checklist for the Coeus tensor library. For detailed task lists, sprint archives, and progress tracking, see:
- [docs/checklist.md](file:///d:/coeus/docs/checklist.md)
- [docs/backlog.md](file:///d:/coeus/docs/backlog.md)

---

## Default Parallel Memory Features [patch]
- [x] Restored Moirai default features for workspace consumers so parallel execution, Mnemosyne-backed memory surfaces, and Mellinoe branding are active by default.
- [x] Restored Leto and Leto Ops defaults so Coeus consumes Mnemosyne-backed Leto storage and default parallel ops without local feature suppression.
- [ ] Verification: pending.

## 🟩 Phase 1: CPU Workspace Stabilization (100% Complete)
Established a clean, warning-free compiler baseline and resolved lifetime and borrow checker conflicts across all workspace crates.

- [x] **Zero-Copy Layout Traversal**: Refactored `coeus-ops` kernels (unary, binary, matmul, sum/mean reductions, SpMV, SpMM) to perform direct strided index math without calling `to_contiguous()`.
- [x] **Thread-Safe Pointers**: Created `SendPtr` and `SendPtrMut` raw pointer wrappers in `coeus-ops/src/ptr.rs` to allow safe, thread-safe capture of raw pointers in `Moirai` parallel closures.
- [x] **Apollo FFT Integration**: Replaced DFT placeholder with high-performance typed routing to the real `apollo-fft` library.
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
- [x] **Embedded PTX Kernels**: Implemented tiled matrix multiplication, sum reduction, Conv1D/Conv2D forward and backward passes, and strided/broadcasted element-wise operations entirely on-device.

