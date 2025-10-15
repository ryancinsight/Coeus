# Coeus Agent Playbook

## Architecture Snapshot
- Multi-crate Rust workspace with crates `dtype`, `storage`, `backend`, `tensor`, `autograd`, `nn`, `optim`, `utils`, `tokenizer`; Python bridge lives in `pycoeus`.
- Tensors follow `Tensor<Backend, DenseStorage<T>, T>`; CPU dense tensors dominate, so default to `Tensor<CpuBackend, DenseStorage<T>, T>` unless explicitly targeting sparse or GPU backends.
- `dtype` defines numeric traits (`DataType`, `FloatExt`) consumed everywhere; import these when writing math or gradient code instead of hard-coding `f32`.
- `storage` and `backend` layer abstractions for memory layout and execution; prefer reusing helpers in `tensor/src/shape.rs` and `storage/src/dense.rs` to keep shape math consistent.
- `tensor` crate owns arithmetic and shape ops; new tensor functionality must return `Result` and obey ADR-002 (no panics on user input).

## Autograd & Tensor Patterns
- Variables wrap tensors via `coeus_autograd::Variable` (Arc + RwLock); cloning a `Variable` shares gradient state, so avoid interior mutation outside the provided APIs.
- Each differentiable op has an `Operation<T>` variant (`autograd/src/operation.rs`); extending autograd means adding a variant plus `backward` and `accumulate_gradients` logic.
- When constructing new Variables, always call `result.set_creator(Arc::new(Operation::...))` so `backward(&[&var], &[])` can traverse the graph (`autograd/src/variable.rs`).
- Gradient accumulation happens through `Variable::accumulate_grad`; keep shapes aligned by using `Tensor::from_slice`/`from_vec` with the original dims to avoid silent broadcasting.
- The graph utilities in `autograd/src/graph.rs` assume CPU dense tensors; if you introduce other backends, ensure gradients convert back to `Tensor<CpuBackend, DenseStorage<T>, T>`.

## Neural Network Modules
- Modules implement `nn::module::Module`; `parameters()` should return only this module's learnables, while `modules()` exposes children so `state_dict` avoids duplicates.
- `nn::parameter::Parameter` wraps `Variable`; create parameters with `Parameter::new(Variable::new(tensor), true, "weight".into())` and call `zero_grad()` after optimizer steps.
- Weight init lives in `nn::init`; prefer helpers like `xavier_uniform_` instead of ad-hoc randomness.
- RNN/LSTM/GRU layers mirror PyTorch gate layout (`nn/src/rnn.rs`) and expect pre-split weight tensors; preserve tensor ordering when mutating or reinitializing.
- Serialization uses `ModuleSerialize`; ensure new modules fill `child_module_names()` when order matters so `state_dict` names stay stable.

## Dev Workflow
- Use targeted tests to stay fast: `cargo test -p coeus-autograd` for gradient logic, `cargo test -p coeus-nn rnn::` for sequence modules, `cargo test -p coeus-tensor tensor::reshape` for shape math.
- Python bindings require `maturin develop` in `pycoeus/` before running `pytest tests/test_pycoeus.py`; the Rust side has sanity tests in `pycoeus/tests/python_integration.rs`.
- Enforce hygiene with `cargo fmt`, `cargo clippy -p coeus-autograd -D warnings`, and run `cargo miri test -p coeus-autograd` before touching unsafe (nearly all code stays safe).
- Benchmarks live under `benches/`; replicate performance numbers with `cargo bench --bench conditional_unsafe` after enabling `--release`.
- Docs and ADRs live in `docs/`; consult `docs/adr.md` or crate-specific ADR files before changing architecture decisions.

## Integration & Observability
- Logging uses `tracing`; prefer the `instrument` macro (see `autograd/src/graph.rs`) when instrumenting new hot paths.
- Distributed, JIT, and Hub crates are present but less mature; check their READMEs before relying on them for production features.
- Quantization, AMP, and optimizer utilities live in `nn/quantization.rs` and `nn/amp.rs`; follow existing enums and config structs to keep Python parity.
- Sparse tensor support is centralized in `tensor/tensor/sparse_*` helpers; reuse format converters instead of reinventing CSR/COO bookkeeping.
- Avoid panics in public APIs; return crate-specific `Result` types (`autograd::Result`, `nn::Result`) with `thiserror` variants from the corresponding `error.rs`.

## Reference Spots
- Canonical tensor math: `tensor/src/lib.rs` and `tensor/src/ops/*.rs`.
- Autograd entry points: `autograd/src/lib.rs`, `variable.rs`, `operation.rs`, `graph.rs`.
- Neural modules: `nn/src/linear.rs`, `nn/src/conv.rs`, `nn/src/transformer.rs`, with functional mirrors in `nn/src/functional.rs`.
- Optimizers: `optim/src/*`; they expect parameters with gradients already set (tests demonstrate calling `set_grad` before `step()`).
- Python surface: `pycoeus/src/lib.rs` (PyO3 bindings) and `pycoeus/python/coeus/__init__.py` for exported API surface.
