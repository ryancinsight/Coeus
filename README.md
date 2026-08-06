# Coeus: High-Performance Strided Tensor Library in Rust

Coeus is an experimental, high-performance, N-dimensional strided tensor library written in Rust from first principles. It is engineered with a strict emphasis on monomorphization, zero-copy layout traversal, and zero-cost abstractions, designed to serve as a high-signal alternative to `ndarray`, PyTorch, and JIT-compiled runtimes.

---

## Core System Architecture & Design Invariants

Coeus is partitioned into a deep vertical hierarchy of workspace crates, enforcing absolute Separation of Concerns (SoC) and the Dependency Inversion Principle (DIP).

### Crate Structure
- **[coeus-core](crates/coeus-core)**: Core primitives. Defines `Scalar` and `Float` traits, strided `Layout`, the `Storage` and `StorageMut` allocation interface, and the `Backend` execution trait.
- **[coeus-tensor](crates/coeus-tensor)**: The fundamental `Tensor<T, B>` type with Copy-on-Write (COW) semantics and layout views (slicing, transposing, broadcasting).
- **[coeus-ops](crates/coeus-ops)**: Optimized mathematical kernels (unary, binary, matmul, reductions) and signal processing routines.
- **[coeus-autograd](crates/coeus-autograd)**: A tape-based automatic differentiation engine supporting reverse-mode autodiff.
- **[coeus-nn](crates/coeus-nn)**: Neural network modules (Linear, Conv2d, LayerNorm, RMSNorm, BatchNorm2d) and activation functions.
- **[coeus-optim](crates/coeus-optim)**: Parameter optimizers (SGD, Adam, RMSProp).
- **[coeus-sparse](crates/coeus-sparse)**: Sparse storage representations (COO, CSR).
- **[coeus-python](crates/coeus-python)**: Lightweight PyO3 bindings exposing the Rust core to Python.

### Architectural Invariants

1. **Zero-Copy Layout Traversal**
   All kernels in `coeus-ops` traverse tensors natively using strides and physical offsets. Memory allocation and copying (e.g. `to_contiguous()`) are completely eliminated during layout manipulation, slicing, broadcasting, or transposition.
2. **Monomorphized Kernel Dispatch**
   Operations are parameterized by generic parameters such as `<T: Scalar, B: ComputeBackend>`. Downstream compiles evaluate to direct specialization blocks with zero virtual dispatch on the provider path.
3. **Moirai Parallel Execution**
   Multithreading is driven by the [Moirai Threading Engine](https://github.com/ryancinsight/Moirai.git), utilizing a work-stealing parallel execution queue via `MoiraiBackend::parallel_for`.
4. **Mnemosyne Memory Allocator**
   Low-level heap allocations are managed via the custom [Mnemosyne Allocator](https://github.com/ryancinsight/Mnemosyne), implementing aligned memory blocks (`RawBlock`) directly from the allocator instance.
5. **Apollo FFT Integration**
   FFT operations on `coeus` Tensors and differentiable Vars are supported via the integration layer in the [Apollo FFT Library](https://github.com/ryancinsight/apollo.git), keeping `coeus` clean of compile-time dependencies on Apollo.


---

## Workspace Backlog & Roadmap

The current development focus is transitioning Coeus to a heterogeneous execution model supporting GPU acceleration as first-class targets.

Refer to the global checklist and backlog documents:
- **Checklist**: [CHECKLIST.md](CHECKLIST.md)
- **Backlog**: [docs/backlog.md](docs/backlog.md)

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

## Python Releases

GitHub Releases tagged `coeus-python-v<version>` build locked CPython 3.9–3.13
wheels for Linux, Windows, and macOS. The workflow installs and imports each
wheel as `pycoeus`, verifies that its `coeus-python` metadata version matches
the release tag, attests and attaches the exact wheel set to the GitHub
Release, then publishes those same artifacts to PyPI through OIDC Trusted
Publishing. The tag version must equal the workspace Cargo version.

## Rust Crate Releases

The `Crates.io Release` workflow validates a named workspace package on manual
dispatch. After that package's required first release is published locally and
its crates.io Trusted Publisher is registered, a GitHub Release tagged
`crate-<package>-v<version>` packages, verifies, and publishes the matching
Cargo version with a short-lived OIDC token. Validation runs in a separate
read-only job. The publish job is bound to the GitHub `crates-io` environment;
register each package's Trusted Publisher with that environment. The PyO3
package remains a wheel-only artifact and is marked `publish = false` for
crates.io.

### Run Clippy Lints
```bash
cargo clippy --all-targets --all-features -- -D warnings
```

The default `coeus-cuda` build exposes storage and capability types but no
mathematical CPU fallback. Native Hephaestus/Cutile CUDA execution is explicit:
```bash
cargo test -p coeus-cuda --features cuda
```
That CUDA feature requires `CUDA_TOOLKIT_PATH` and a working CUDA driver.
Provider failures are surfaced to callers; a present CUDA provider never
silently downgrades execution to Leto.

### Run Benchmarks
`coeus-tensor` contains Criterion baselines for Coeus Sequential, Coeus Moirai,
direct Leto, and the Coeus-Leto dispatch seam. Native analytical and
backend-conformance tests own correctness evidence; benchmarks measure the
implemented provider paths. The workspace bench profile
uses thin LTO and one codegen unit so generic cross-crate kernels are measured
with production-grade monomorphization:
```bash
cargo bench -p coeus-tensor --bench tensor_bench
```
