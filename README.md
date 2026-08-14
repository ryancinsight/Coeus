# Coeus: High-Performance Strided Tensor Library in Rust

Coeus is an experimental, high-performance, N-dimensional strided tensor library written in Rust from first principles. It is engineered with a strict emphasis on monomorphization, zero-copy layout traversal, and zero-cost abstractions, designed to serve as a high-signal alternative to `ndarray`, PyTorch, and JIT-compiled runtimes.

---

## Core System Architecture & Design Invariants

Coeus is partitioned into a deep vertical hierarchy of workspace crates, enforcing absolute Separation of Concerns (SoC) and the Dependency Inversion Principle (DIP).

### Crate Structure

Foundation and CPU stack:

- **[coeus-core](crates/coeus-core/README.md)**: Core primitives. Defines `Scalar` and `Float` traits, strided `Layout`, the `Storage`/`StorageMut` allocation interface, and the `ComputeBackend` device abstraction with its `SequentialBackend` and `MoiraiBackend` CPU implementations.
- **[coeus-tensor](crates/coeus-tensor/README.md)**: The fundamental `Tensor<T, B>` type with Copy-on-Write (COW) semantics, layout views (slicing, transposing, broadcasting), and rkyv-backed checkpointing. Holds no math kernels.
- **[coeus-ops](crates/coeus-ops/README.md)**: The kernel library and `BackendOps` dispatch surface — elementwise, matmul, reductions, scans, convolution, pooling, embedding, interpolation, attention, and a lazy fused-expression DAG.
- **[coeus-autograd](crates/coeus-autograd/README.md)**: A tape-based reverse-mode automatic differentiation engine over `Var<T, B>`.
- **[coeus-nn](crates/coeus-nn/README.md)**: Neural network modules (Linear, Conv1/2/3d, the normalization family, pooling, attention, transformer blocks, RNN) and activations.
- **[coeus-optim](crates/coeus-optim/README.md)**: Parameter optimizers (SGD, Adam, AdamW, RMSProp, AdaGrad), gradient clipping, and LR schedulers.
- **[coeus-sparse](crates/coeus-sparse/README.md)**: Sparse storage representations (`CooTensor`, `CsrTensor`). Data structures only.
- **[coeus-leto](crates/coeus-leto/README.md)**: Const-rank dispatch shim adapting Coeus dynamic-rank layouts onto Leto array kernels (leto ADR 0002).

Accelerator and integration crates — the current development focus:

- **[coeus-hephaestus](crates/coeus-hephaestus/README.md)**: Vendor-neutral bridge owning device storage, host/device transfer, layout validation, and dispatch once, against the `HephaestusProvider` trait family.
- **[coeus-wgpu](crates/coeus-wgpu/README.md)**: WebGPU backend with WGSL compute kernels for unary, binary, matmul, reduction, pooling, unfold/fold, and fused ops.
- **[coeus-cuda](crates/coeus-cuda/README.md)**: CUDA backend with CUDA C kernels, gated behind the `cuda` feature.
- **[coeus-rocm](crates/coeus-rocm/README.md)**: Provider wiring binding Coeus operations to `hephaestus-rocm`. Contains no kernels of its own.
- **[coeus-metal](crates/coeus-metal/README.md)**: Provider wiring binding Coeus operations to `hephaestus-metal`. Contains no kernels of its own.
- **[coeus-dist](crates/coeus-dist/README.md)**: Collective communication over a `Communicator` trait, with thread-local and TCP implementations.
- **[coeus-fft](crates/coeus-fft/README.md)**: Tensor- and autograd-level FFT wrappers over Apollo.

Bindings:

- **[coeus-python](crates/coeus-python/README.md)**: PyO3 bindings exposing the Rust stack to Python as `pycoeus`. Not published to crates.io.

### Architectural Invariants

1. **Zero-Copy Layout Manipulation**
   Slicing, transposition, broadcasting, and reshaping produce new layouts over
   shared storage: they allocate and copy nothing. Kernels in `coeus-ops`
   traverse tensors natively using strides and physical offsets.

   What this does *not* claim: copying is not eliminated everywhere.
   `to_contiguous()` returns the receiver unchanged only when it is already
   contiguous at offset 0; on a strided input it materializes a compacted copy.
   The `iter()` / `iter_mut()` element iterators assert contiguity rather than
   walking strides, so iterating a strided tensor requires materializing first.
2. **Monomorphized Kernel Dispatch**
   Operations are parameterized by generic parameters such as
   `<T: Scalar, B: ComputeBackend>`. Downstream compiles evaluate to direct
   specialization blocks with zero virtual dispatch on the provider path.

   Scope of that claim: it covers the backend/provider dispatch path only. The
   autograd tape is deliberately dynamic — `Var::creator` is an
   `Option<Arc<dyn BackwardNode<T, B>>>`, one heap allocation and one virtual
   call per differentiable operation. Forward kernels stay monomorphized; graph
   construction does not.
3. **Moirai Parallel Execution**
   Multithreading is driven by the [Moirai Threading Engine](https://github.com/ryancinsight/Moirai.git), utilizing a work-stealing parallel execution queue via `MoiraiBackend::parallel_for`.
4. **Mnemosyne Memory Allocator**
   Low-level heap allocations are managed via the custom [Mnemosyne Allocator](https://github.com/ryancinsight/Mnemosyne), implementing aligned memory blocks (`RawBlock`) directly from the allocator instance.
5. **Apollo FFT Integration**
   FFT operations on `coeus` Tensors and differentiable Vars route to the
   [Apollo FFT Library](https://github.com/ryancinsight/apollo.git); Coeus owns
   no FFT math. `apollo-fft` is a hard, non-optional dependency of both
   `coeus-fft` and `coeus-autograd`.


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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
