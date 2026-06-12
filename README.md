# Coeus: High-Performance Strided Tensor Library in Rust

Coeus is an experimental, high-performance, N-dimensional strided tensor library written in Rust from first principles. It is engineered with a strict emphasis on monomorphization, zero-copy layout traversal, and zero-cost abstractions, designed to serve as a high-signal alternative to `ndarray`, PyTorch, and JIT-compiled runtimes.

---

## Core System Architecture & Design Invariants

Coeus is partitioned into a deep vertical hierarchy of workspace crates, enforcing absolute Separation of Concerns (SoC) and the Dependency Inversion Principle (DIP).

### Crate Structure
- **[coeus-core](file:///d:/coeus/coeus-core)**: Core primitives. Defines `Scalar` and `Float` traits, strided `Layout`, the `Storage` and `StorageMut` allocation interface, and the `Backend` execution trait.
- **[coeus-tensor](file:///d:/coeus/coeus-tensor)**: The fundamental `Tensor<T, B, S>` type with Copy-on-Write (COW) semantics and layout views (slicing, transposing, broadcasting).
- **[coeus-ops](file:///d:/coeus/coeus-ops)**: Optimized mathematical kernels (unary, binary, matmul, reductions) and signal processing routines.
- **[coeus-autograd](file:///d:/coeus/coeus-autograd)**: A tape-based automatic differentiation engine supporting reverse-mode autodiff.
- **[coeus-nn](file:///d:/coeus/coeus-nn)**: Neural network modules (Linear, Conv2d, LayerNorm, RMSNorm, BatchNorm2d) and activation functions.
- **[coeus-optim](file:///d:/coeus/coeus-optim)**: Parameter optimizers (SGD, Adam, RMSProp).
- **[coeus-sparse](file:///d:/coeus/coeus-sparse)**: Sparse storage representations (COO, CSR).
- **[coeus-python](file:///d:/coeus/coeus-python)**: Lightweight PyO3 bindings exposing the Rust core to Python.

### Architectural Invariants

1. **Zero-Copy Layout Traversal**
   All kernels in `coeus-ops` traverse tensors natively using strides and physical offsets. Memory allocation and copying (e.g. `to_contiguous()`) are completely eliminated during layout manipulation, slicing, broadcasting, or transposition.
2. **Monomorphized Kernel Dispatch**
   Operations are parameterized by generic parameters `<T: Scalar, B: Backend, S: Storage<T>>`. Downstream compiles evaluate to direct, fully inlined specialization blocks with zero virtual dispatch or heap overhead.
3. **Moirai Parallel Execution**
   Multithreading is driven by the [Moirai Threading Engine](https://github.com/ryancinsight/Moirai.git), utilizing a work-stealing parallel execution queue via `MoiraiBackend::parallel_for`.
4. **Mnemosyne Memory Allocator**
   Low-level heap allocations are managed via the custom [Mnemosyne Allocator](https://github.com/ryancinsight/Mnemosyne), implementing aligned memory blocks (`RawBlock`) directly from the allocator instance.
5. **Apollo FFT Integration**
   1D forward and inverse Fast Fourier Transforms are routed to the [Apollo FFT Library](https://github.com/ryancinsight/apollo.git) using compile-time trait routing and dynamic `TypeId` mapping.

---

## Workspace Backlog & Roadmap

The current development focus is transitioning Coeus to a heterogeneous execution model supporting GPU acceleration as first-class targets.

Refer to the global checklist and backlog documents:
- **Checklist**: [docs/checklist.md](file:///d:/coeus/docs/checklist.md)
- **Backlog**: [docs/backlog.md](file:///d:/coeus/docs/backlog.md)

---

## Building and Running Tests

Ensure you have the latest Rust toolchain installed.

### Compile Workspace
```bash
cargo build --workspace
```

### Run Test Suite
Runs all numerical validation, sparse format, autograd, and FFT parity tests:
```bash
cargo nextest run --workspace
```

Run doctests separately because nextest does not execute them:
```bash
cargo test --doc --workspace
```

### Run Clippy Lints
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

The default `coeus-cuda` build compiles a CPU-backed fallback provider so the
full workspace can be checked on hosts without CUDA. Real cutile-backed CUDA
integration is explicit:
```bash
cargo test -p coeus-cuda --features cuda
```
That CUDA feature requires `CUDA_TOOLKIT_PATH` and a working CUDA driver.

### Run Benchmarks
`coeus-tensor` contains Criterion baselines for Coeus Sequential, Coeus Moirai,
direct Leto, the Coeus-Leto dispatch shim, `ndarray`, `nalgebra`, and a Rayon
slice elementwise baseline. The workspace bench profile uses thin LTO and one
codegen unit so generic cross-crate kernels are measured with production-grade
monomorphization:
```bash
cargo bench -p coeus-tensor --bench tensor_bench
```
