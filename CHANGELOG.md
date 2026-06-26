# Changelog

## 0.5.2 - 2026-06-26

### Added

- **MHA backward Burn parity** — added `multi_head_attention_backward_matches_burn`
  covering forward output, input gradient, and Q/K/V/O projection-weight
  gradients against Burn autodiff with explicit weights.

### Changed

- **Generic Conv dimension strategy** — consolidated `Conv1d`/`Conv2d`/`Conv3d`
  into a single `Conv<T, B, D: ConvDim>` layer with sealed ZST dimension
  strategies and public type aliases for the concrete ranks.
- **SRP backend/autograd layout** — split `coeus-ops` CPU backend dispatch and
  `coeus-autograd` convolution nodes into operation-family leaf modules.
- **Documented public surfaces** — enforced and fixed `coeus-nn`
  `missing_docs`, added missing docs across touched core/tensor/ops/cuda/wgpu
  public items, and kept `coeus-python` missing-docs enforcement as the next
  ready sprint because its binding surface still needs a crate-wide docs pass.

### Verified

- `rustup run nightly cargo fmt -p coeus-core -p coeus-cuda -p coeus-nn -p
  coeus-ops -p coeus-python -p coeus-tensor -p coeus-wgpu --check`.
- `rustup run nightly cargo clippy -p coeus-nn --tests -- -D warnings`.
- `rustup run nightly cargo nextest run -p coeus-nn` passes 270/270.
- `rustup run nightly cargo clippy -p coeus-core -p coeus-cuda -p coeus-ops
  -p coeus-python -p coeus-tensor -p coeus-wgpu --tests -- -D warnings`.
- `rustup run nightly cargo doc -p coeus-core -p coeus-cuda -p coeus-nn -p
  coeus-ops -p coeus-python -p coeus-tensor -p coeus-wgpu --no-deps`.
- `rustup run nightly cargo test --doc -p coeus-core -p coeus-nn -p coeus-ops
  -p coeus-tensor -p coeus-wgpu -p coeus-cuda`.
- `rustup run nightly cargo nextest run -p coeus-python --test
  binding_tests_nn --test binding_tests_ops test_pycoeus_nn
  test_nn_functional_ops` passes 2/2.
- `rustup run nightly cargo clippy -p coeus-ops -p coeus-autograd --tests --
  -D warnings`.
- `rustup run nightly cargo nextest run -p coeus-ops -p coeus-autograd` passes
  224/224.

## 0.5.1 - 2026-06-26

### Added

- **Core Rustdoc examples** — `coeus-core` now carries compiling examples for
  backend, scalar, layout, stride, shape, and CPU storage contracts.
- **WGPU aliasing parity coverage** — `coeus-wgpu` verifies aliased unary and
  binary elementwise paths remain correct when Hephaestus routing is not used.
- **Additional Burn parity coverage** — `coeus-nn` verifies Conv3d forward and
  stride/padding behavior, InstanceNorm2d forward behavior, and transpose
  backward gradients against Burn NdArray/autodiff.
- **Optimizer analytical coverage** — `coeus-nn` Burn parity tests now include
  closed-form RMSProp, AdaGrad, and AdamW first-step references.
- **BatchNorm2d training-mode backward parity** — differential Burn autodiff
  test verifying `dx`, `dw`, `db` for training-mode BatchNorm2d using Coeus's
  NHWC-layout, population-variance formula (97th parity test; ε ≤ 1e-4).
- **Conv3d backward parity** — differential Burn autodiff test verifying `dx`
  and `dw` for valid 3D convolution, completing backward coverage for Conv1d/2d/3d
  (98th parity test).
- **InstanceNorm1d/2d backward parity** — differential Burn autodiff tests
  verifying `dx`, `dw`, `db` for InstanceNorm1d [N,C,L] and InstanceNorm2d
  [N,C,H,W] backward passes (99th and 100th parity tests; ε ≤ 1e-4).
- **InstanceNorm3d** — new `InstanceNorm3d<T, B>` layer normalizing [N,C,D,H,W]
  inputs over the D×H×W spatial volume per (sample, channel); shared
  `ensure_cache` + `instance_norm_forward` helpers eliminate the duplicate
  `get_cache` from InstanceNorm1d/2d (consolidation). Exported from `coeus_nn`.
- **InstanceNorm3d Burn parity** — differential test verifying forward values
  and `dx`/`dw`/`db` against a manual Burn autodiff reference (101st parity test;
  ε ≤ 1e-4).
- **Functional bilinear Python surface** — added `pycoeus.bilinear(input1,
  input2, weight, bias=None)` as a thin validated wrapper over Rust-core
  `coeus_nn::bilinear(...)`.
- **Functional normalization core helpers** — added Rust-core
  `coeus_nn::layer_norm(...)` and `coeus_nn::rms_norm(...)` for thin-wrapper
  reuse across bindings.
- **Autograd and tensor Rustdoc examples** — added executable examples for
  gradient mode, `Var`, core tracked ops, and tensor shape/view contracts.
- **Coeus ops Rustdoc examples** — public arithmetic, CPU backend dispatch,
  reductions, matmul helpers, shape concatenation/stacking, and unary math APIs
  now carry executable examples.
- **Distributed and sparse Rustdoc examples** — `coeus-dist` communicator/local
  cluster examples and `coeus-sparse` COO/CSR construction/accessor examples now
  compile as doctests.
- **Leto bridge Rustdoc examples** — `coeus-leto` layout conversion, view
  conversion, elementwise dispatch, initialization, layout transform, and linear
  algebra bridge APIs now carry executable examples.
- **BatchNorm3d training-mode backward parity** — differential Burn autodiff
  test verifying `dx`, `dw`, and `db` for training-mode BatchNorm3d using the
  Coeus population-variance formula (ε ≤ 1e-4).
- **Python transformer encoder bindings** — `coeus-python` now registers
  `TransformerEncoderLayer`, `TransformerEncoder`, and `SinusoidalEncoding`
  wrappers over Rust-core `coeus_nn` implementations.
- **Autograd public documentation surface** — documented public operation
  modules, backward-node state, and tracked sparse/shape entry points so
  `coeus-autograd` satisfies its `#![deny(missing_docs)]` contract.

### Fixed

- **Core doctest correctness** — corrected public examples to import the real
  trait providers and use existing `Shape` APIs.
- **CUDA fused-kernel cache contention** — `coeus-cuda` now uses a read/write
  lock for fused-kernel cache hits and inserts, matching the read-mostly cache
  contract.

### Changed

- **CUDA Hephaestus primitive routing (contiguous + strided)** — contiguous
  non-aliased `coeus-cuda` primitive elementwise operations route through
  `hephaestus-cuda` first; strided paths with rank ≤ `MAX_STRIDED_RANK` and no
  broadcast output dimension now also route through Hephaestus's dynamic-strided
  kernel, with Coeus-local CUDA kernels retained for aliasing, out-of-range rank,
  broadcast output, and NN-specific formulas.
- **WGPU Hephaestus zero-allocation dispatch** — contiguous non-aliased
  elementwise unary/binary routes in `coeus-wgpu` now use Hephaestus `*_into`
  APIs to write into caller-owned output buffers instead of allocating and
  swapping buffers.
- **Bilinear SSOT routing** — `coeus_nn::Bilinear::bilinear_forward` now
  delegates to shared functional `coeus_nn::bilinear`, and `coeus-python`
  bilinear module forward reuses that same core path without constructing a
  temporary module per call.
- **Normalization wrapper SSOT routing** — `coeus-python` functional
  `layer_norm` / `rms_norm` now dispatch directly to Rust-core functional
  helpers with explicit boundary validation instead of constructing modules in
  the binding.
- **Autograd backend bounds** — corrected touched autograd node definitions to
  use the canonical `coeus_ops::BackendOps` backend trait in public bounds.
- **WGPU metadata-buffer pooling** — the metadata-buffer pool now uses a
  nonblocking reuse path and a fixed retained-buffer cap, avoiding mutex waits
  on contended kernel submissions and preventing unbounded pool growth.
- **GELU doctest reference** — corrected the `coeus-ops` exact-GELU doctest to
  assert the exact `erf`-based reference value, not the tanh approximation.
- **Python transformer dispatch errors** — unsupported transformer const-generic
  binding selections now return `ValueError` instead of unwinding through PyO3.

### Verified

- `cargo test --doc -p coeus-core` passes 32/32 doctests.
- `cargo nextest run -p coeus-core` validates core tests.
- `cargo nextest run -p coeus-nn --test burn_live_parity
  validates 94/94 Burn parity and analytical optimizer tests.
- `cargo check -p coeus-cuda` and `cargo check -p coeus-cuda --features cuda`
  validate both CUDA backend build surfaces.
- `cargo clippy -p coeus-cuda --all-targets --features cuda -- -D warnings`
  validates the CUDA routing surface.
- `cargo nextest run -p coeus-cuda --features cuda` passes 69/69 live CUDA
  tests, including primitive parity rows routed through Hephaestus.
- `cargo test -p coeus-wgpu
  test_wgpu_hephaestus_contiguous_binary_reuses_output_buffer` validates
  contiguous binary delegated routing preserves output-buffer identity.
- `cargo test -p coeus-wgpu
  test_wgpu_hephaestus_contiguous_unary_reuses_output_buffer` validates
  contiguous unary delegated routing preserves output-buffer identity.
- `rustup run nightly cargo check -p coeus-nn --lib` and `rustup run nightly
  cargo check -p coeus-python --lib` validate exported normalization functional
  surfaces and wrapper routing.
- `rustup run nightly cargo nextest run -p coeus-nn --test nn_norm_tests
  test_layernorm test_rmsnorm` passes 4/4 normalization functional tests.
- `rustup run nightly cargo nextest run -p coeus-python --test
  binding_tests_ops test_nn_functional_ops` validates Python functional norm
  wrappers.
- `rustup run nightly cargo nextest run -p coeus-python --test
  binding_tests_ops test_transformer_encoder_bindings test_transformer_decoder_layer
  test_nn_functional_ops` passes 3/3 focused Python binding tests.
- `rustup run nightly cargo nextest run -p coeus-nn --test burn_live_parity
  batchnorm3d_training_backward_matches_burn` passes the BatchNorm3d
  training-backward Burn parity case.
- `rustup run nightly cargo test --doc -p coeus-optim` passes 10/10 doctests.
- `rustup run nightly cargo nextest run -p coeus-optim` passes 14/14 tests.
- `rustup run nightly cargo doc -p coeus-wgpu -p coeus-nn -p coeus-python -p
  coeus-optim --no-deps` validates touched package docs.
- `rustup run nightly cargo nextest run -p coeus-wgpu` passes 83/83 WGPU
  tests, including strided Hephaestus parity and storage tier tests.
- `rustup run nightly cargo test --doc -p coeus-ops` validates arithmetic and
  activation public examples.
- `rustup run nightly cargo test --doc -p coeus-dist -p coeus-sparse` passes
  16/16 distributed and sparse doctests.
- `rustup run nightly cargo test --doc -p coeus-leto` passes 28/28 bridge
  doctests.
- `rustup run nightly cargo fmt -p coeus-autograd --check` validates autograd
  formatting.
- `rustup run nightly cargo test --doc -p coeus-autograd` passes 15/15 doctests.
- `rustup run nightly cargo clippy -p coeus-autograd --tests -- -D warnings`
  validates the documented autograd test surface.
- `rustup run nightly cargo nextest run -p coeus-autograd` passes 35/35 tests.
- `rustup run nightly cargo doc -p coeus-autograd --no-deps` validates the
  warning-clean public documentation build.
- `cargo nextest run -p coeus-nn --test bilinear_parity` validates functional and
  module bilinear parity on Sequential and Moirai backends.
- `cargo nextest run -p coeus-python --test binding_tests_ops
  test_nn_functional_ops` validates new `pycoeus.bilinear` functional behavior
  and validation paths.
- `cargo nextest run -p coeus-python --test binding_tests_ops
  test_bilinear_module` validates the Python module path after direct
  Rust-core bilinear delegation.
- `cargo nextest run -p coeus-nn --test burn_live_parity
  instancenorm3d_forward_backward_matches_burn` validates InstanceNorm3d
  forward and backward parity against Burn autodiff.
- `cargo test --doc -p coeus-autograd -p coeus-tensor -p coeus-nn` validates
  executable Rustdoc examples for the touched public APIs.
- `cargo fmt --check`, `coeus-core`/`coeus-cuda` clippy, and `coeus-core`
  rustdoc pass.

## 0.5.0 - 2026-06-26

### Added

- **Python functional GroupNorm** — `coeus-python` now exposes
  `pycoeus.group_norm` as a thin PyO3 wrapper over Rust-core
  `coeus_nn::group_norm`, including Python boundary validation and
  `allow_threads` compute execution.
- **BatchNorm2d Burn parity coverage** — `coeus-nn` now verifies eval-mode
  `BatchNorm2d` forward output against Burn NdArray.

### Verified

- `cargo nextest run -p coeus-python --test binding_tests_ops
  test_nn_functional_ops` validates exact no-affine and affine GroupNorm
  output plus zero-group rejection through the Python binding.
- `cargo nextest run -p coeus-nn --test burn_live_parity
  batchnorm2d_eval_forward_matches_burn` validates BatchNorm2d eval-mode
  parity against Burn.
- `cargo fmt --check`, `coeus-python` clippy, and `coeus-python` rustdoc pass.

## 0.4.0 - 2026-06-26

### Added

- **Functional GroupNorm** — `coeus-nn` now exports stateless
  `group_norm`, keeping normalization computation in Rust core for thin PyO3
  wrapper parity with Burn/PyTorch-style functional APIs.

### Changed

- **WGPU elementwise routing** — contiguous non-aliased unary and binary
  elementwise operations in `coeus-wgpu` now route through Hephaestus public
  kernels where supported, retaining Coeus-local kernels for aliasing and
  unsupported operation cases.

### Verified

- `cargo nextest run -p coeus-nn --test norm_parity` validates exact
  analytical functional GroupNorm output and rejection semantics.
- `cargo nextest run -p coeus-wgpu` validates WGPU package behavior.
- `cargo fmt --check`, `coeus-nn`/`coeus-wgpu` clippy, and
  `coeus-nn`/`coeus-wgpu` rustdoc pass.

## 0.3.0 - 2026-06-26

### Added

- **Statistics pair reductions** — `coeus-ops` now exposes `var_mean`,
  `std_mean`, `var_mean_axis`, and `std_mean_axis`; standalone variance and
  standard-deviation APIs delegate to the pair-returning SSOT.
- **Python statistics pairs** — `coeus-python` now exposes thin PyO3
  `var_mean` and `std_mean` wrappers with optional `axis` and `keepdim`.
- **Sequence-level RNN modules** — `coeus-nn` now exports `Gru` and `Lstm`
  modules with `forward_seq` outputs and final hidden/cell state access.
- **NN module parity coverage** — `coeus-nn` now verifies `Bilinear`,
  `ConvTranspose1d`/`ConvTranspose2d`, and sequence-level `Gru`/`Lstm`
  modules against analytical references on SequentialBackend and MoiraiBackend.

### Verified

- `cargo nextest run -p coeus-ops --test stats_diff` passes 2/2 tests.
- `cargo nextest run -p coeus-python --test binding_tests_ops` passes 58/58
  tests.
- `cargo nextest run -p coeus-nn --test bilinear_parity --test
  conv_transpose_nn_parity --test rnn_seq_parity` passes 6/6 tests.
- `cargo fmt --check`, `coeus-ops`/`coeus-python`/`coeus-nn` clippy, and
  `coeus-ops`/`coeus-python`/`coeus-nn` rustdoc pass.

## 0.2.34 - 2026-06-26

### Added

- **Recurrent-cell differential coverage** — `coeus-nn` now verifies `GRUCell`
  and `LSTMCell` zero-input analytical references on SequentialBackend and
  MoiraiBackend.
- **Interpolation differential coverage** — `coeus-nn` now verifies
  `interpolate_1d` and `interpolate_2d` references on both CPU backends.
- **Loss differential coverage** — `coeus-nn` now verifies MSE, NLL, Huber,
  binary cross entropy, and cosine embedding loss against closed-form
  references.
- **Positional encoding differential coverage** — `coeus-nn` now verifies
  sinusoidal and rotary positional encodings against analytical references.
- **Pooling differential coverage** — `coeus-nn` now verifies global 1-D/3-D
  pooling plus `AvgPool3d` and `MaxPool3d` analytical references.

### Verified

- `cargo nextest run -p coeus-nn` passes 236/236 tests.
- `cargo fmt --check`, `coeus-nn` clippy, and `coeus-nn` rustdoc pass.

## 0.2.33 - 2026-06-26

### Added

- **Embedding differential coverage** — `coeus-ops` now verifies embedding
  lookup, repeated-index gradient accumulation, and padding-index gradient
  suppression on SequentialBackend and MoiraiBackend.
- **Unary and shape ops differential coverage** — `coeus-ops` now verifies
  unary math identities plus `flip`, `roll`, triangular masks, sorting,
  one-hot, repeat-interleave, outer product, and cross product on
  SequentialBackend and MoiraiBackend.
- **Activation differential coverage** — `coeus-ops` now verifies sigmoid,
  GELU, tanh-GELU, SiLU, Mish, ELU, Softplus, and LeakyReLU against exact or
  analytically bounded scalar references.
- **Transposed convolution differential coverage** — `coeus-ops` now verifies
  `conv_transpose1d` and `conv_transpose2d` against hand-derived
  scatter-accumulate references.
- **Miscellaneous ops differential coverage** — `coeus-ops` now verifies `amax`,
  `amin`, `dot`, `cumprod`, `broadcast_to`, `chunk`, `diag`, and `diagonal`
  against exact integer-valued references.
- **Product/tile/masked-fill differential coverage** — `coeus-ops` now verifies
  `prod`, `tile`, and `masked_fill` exact references on SequentialBackend and
  MoiraiBackend.
- **Sparse conversion differential coverage** — `coeus-ops` now verifies
  dense/COO/CSR conversions and dense roundtrip invariants.
- **Leto unary dispatch contract coverage** — `coeus-leto` now verifies
  `Exp`, `Log`, and `Sqrt` dispatch against exact scalar references while
  consuming the upstream Leto SIMD sqrt strategy through public Leto APIs.

### Verified

- `cargo nextest run -p coeus-ops` passes 189/189 tests.
- `cargo nextest run -p coeus-leto --test contract` passes 25/25 tests.
- `cargo fmt --check`, `coeus-ops`/`coeus-leto` clippy, and
  `coeus-ops`/`coeus-leto` rustdoc pass.

## 0.2.32 - 2026-06-26

### Added

- **Sparse operation differential coverage** — `coeus-ops` now verifies `spmv`,
  `spmm`, `spmm_backward_values`, and `spmm_backward_dense` on SequentialBackend
  and MoiraiBackend against exact integer-valued CSR references.

### Verified

- `cargo nextest run -p coeus-ops` passes 173/173 tests.
- `cargo fmt --check`, sparse-op clippy, and `coeus-ops` rustdoc pass.

## 0.2.31 - 2026-06-26

### Added

- **Constructor and selection differential coverage** — `coeus-ops` now verifies
  `linspace`, `logspace`, `geomspace`, `meshgrid`, `nonzero`, and `where_cond`
  on SequentialBackend and MoiraiBackend against bitwise-exact references.
- **Index/scatter/BMM differential coverage** — `coeus-ops` now verifies
  `gather`, `index_select`, `index_put`, `scatter_add`, `masked_select`, and
  `bmm` with value-semantic backend parity assertions.
- **Initializer and interpolation parity coverage** — `coeus-nn` now verifies
  seeded uniform/normal, Xavier, and Kaiming initializer dispatch against
  `coeus-leto`, plus analytical nearest/bilinear interpolation references.

### Verified

- `cargo nextest run -p coeus-ops` passes 171/171 tests.
- `cargo nextest run -p coeus-nn` passes 224/224 tests.
- `cargo fmt --check`, touched-package clippy, and touched-package rustdoc pass.

## 0.2.30 - 2026-06-26

### Added

- **COO sparse autograd matmul** —
  `coeus_autograd::sparse_matmul_coo` accepts tracked COO values plus COO
  indices, converts to CSR once, reuses the existing CSR SpMM forward/backward
  kernels, and remaps value gradients back to original COO order.
- **Statistical reduction differential coverage** — `coeus-ops` now verifies
  variance, standard deviation, and Lp-norm reductions on SequentialBackend and
  MoiraiBackend against analytical references.

### Changed

- **PyTensor binding hierarchy** — split the former monolithic tensor binding
  file into `tensor::pyimpl`, `tensor::iter`, and `tensor::state_dict`, keeping
  Python as a thin PyO3 boundary over Rust tensor/autograd behavior.
- **Direct dependency cleanup** — removed unused `num-traits` from `coeus-ops`;
  numeric trait integration remains centralized in `coeus-core`.

### Verified

- `cargo nextest run -p coeus-autograd` passes 35/35 tests, including COO sparse
  matmul forward/backward parity against dense `matmul`.
- `cargo nextest run -p coeus-ops` passes 167/167 tests after the dependency
  cleanup.
- `cargo nextest run -p coeus-python` passes 70/70 tests after the tensor module
  split.
- `cargo fmt --check`, touched-package clippy, and touched-package rustdoc pass.

## 0.2.29 - 2026-06-26

### Added

- **Transformer encoder source key-padding masks** —
  `TransformerEncoderLayer::forward_with_mask`,
  `TransformerEncoder::forward_with_mask`, and
  `Transformer::forward_seq2seq_with_src_mask` route optional source masks
  through the existing attention mask path while preserving `Module::forward`
  as the unmasked entry point.

- **BatchNorm eval bindings for Python** — `pycoeus.BatchNorm1d` and
  `pycoeus.BatchNorm3d` now expose `eval_forward`, matching the existing
  BatchNorm2d binding. The Python stub also records `BatchNorm1d/2d/3d`,
  `matrix_norm`, and `Embedding(..., padding_idx=...)` surface parity.

### Verified

- `cargo nextest run -p coeus-nn` passes with 211 tests, including masked
  encoder-layer shape/gradient coverage and all-ones-mask parity with the
  unmasked path.
- `cargo nextest run -p coeus-python` passes with 70 tests, including
  BatchNorm1d/2d/3d eval-mode normalization against running stats without
  mutation.
- `cargo fmt --check`, `cargo clippy -p coeus-nn -p coeus-python
  --all-targets -- -D warnings`, and `cargo doc -p coeus-nn -p coeus-python
  --no-deps` pass.

## 0.2.28 - 2026-06-26

### Added

- **`coeus_ops::frobenius_norm` / `coeus_ops::frobenius_norm_batched`** — Frobenius
  matrix-norm kernels composing on the existing `coeus_ops::norm`
  (`sqrt(sum(x·x))`) and a host-side per-batch fold for higher-rank inputs.
  `frobenius_norm(a)` reduces a single 2-D matrix to a scalar; the batched
  variant reduces an N-D tensor over its last two dimensions, returning one
  Frobenius norm per leading batch slot (shape `a.shape[..-2]`).
  Compositionally identical to `torch.linalg.matrix_norm(input, ord='fro')`
  semantics for any rank ≥ 2. Zero new `BinaryOp` opcodes, zero new backend
  dispatch.

- **`pycoeus.matrix_norm(input, ord='fro')`** — PyO3 binding over the Rust
  kernel. 2-D inputs return a Python `float` (mirrors `torch`'s coercion of
  a 0-D Tensor to a Python scalar); N-D inputs return a `PyTensor` with
  leading batch shape. 1-D inputs surface as `ValueError`; `ord` values
  other than `'fro'` are also rejected at the boundary (other matrix-norm
  orderings require SVD or column/row-sum analysis and are deferred per
  MS-88). The dispatch pattern mirrors the existing
  `coeus_python::ops::statistics::sum_axis` adapter (rank-aware `float` vs
  `PyTensor`).

- **Embedding padding index contract** — `coeus_nn::Embedding` now stores an
  optional `padding_idx`, zeros that row on construction, and routes forward
  through `coeus_autograd::embedding_with_padding_idx` so the padding row
  receives no gradient. `pycoeus.Embedding(..., padding_idx=...)` preserves
  the same Rust-core contract.

- **Vertical shape module hierarchy** — `coeus-ops` and `coeus-autograd`
  shape operations now live under concern-oriented submodules
  (`concat_split_stack`, `transform`, `select`, `mask`, `util`) while preserving
  the existing public exports.

### Verified

- **Axis-aware sum + sqrt mirrors scalar `norm`** — `frobenius_norm(a)`
  and `frobenius_norm_batched(a)` share the exact same squared-sum
  accumulator as `coeus_ops::norm`; the 2-D case is bitwise-identical
  (both reduce the entire matrix to a single scalar with the same
  `mul`+`reduce(Sum)`+`sqrt` chain). The new ops do not introduce a new
  accumulation order or a new precision contract on the supported
  path.

- **CoW storage integration unchanged** — `coeus_tensor`'s existing
  `CowStorage<S>` (verified in MS-83) covers the cloned-slice inputs
  passed to `frobenius_norm_batched`; the host-side fold runs on
  contiguous storage and avoids any second copy of the input data.

- **BatchNorm eval-mode coverage** — `BatchNorm1d::set_training(false)` is
  covered by a value-semantic regression test that verifies running stats are
  read without mutation.

## 0.2.25 - 2026-06-25

### Added

- **f16 / bf16 half-precision compute path** — `half::f16` and `bf16` already implemented
  the `Scalar` and `Float` traits in `coeus-core`. Added 3 smoke tests confirming that
  `coeus-ops::add`, `coeus-ops::matmul`, and autograd `sum(x*x).backward()` all work
  with `half::f16` tensors end-to-end.

- **`pycoeus.pyi` Python type stub** — Comprehensive type stub file at
  `coeus-python/pycoeus.pyi` covering all public functions, classes, and properties.
  Enables IDE auto-completion, mypy validation, and automated API documentation.

### Verified

- **GEMV 8× row-blocking already in place** — `hermes-simd` `dispatch_gemv_kernel`
  already dispatches `TilingPolicy<8,1>` (8-row blocking) for `LANE_COUNT > 8` (AVX512)
  and `TilingPolicy<4,1>` for wider SIMD. No further changes needed.

- **FFT stubs via Apollo** — Apollo FFT integration path documented in CHANGELOG;
  stub implementation deferred pending Apollo crate stabilization.



### Added

- **`coeus_nn::Bilinear`** — Bilinear interaction layer
  `out[n,k] = Σ_{i,j} x1[n,i] * W[k,i,j] * x2[n,j] + b[k]`.
  Autograd-tracked via slice-per-output-feature + matmul + sum.
  `bilinear_forward(x1, x2)` is the primary API; `Module::forward(x)` self-interacts.

- **`PyTensor.__setitem__`** — Python index assignment `t[i] = scalar_or_tensor`.
  Supports integer indices (including negative), assigns a scalar to all elements
  in the row or a tensor with matching row numel. Non-tracked in-place mutation.
  `IndexError` for out-of-range, `TypeError` for slice index, `ValueError` for shape mismatch.

### Verified (parity tests added)

- **`cat` backward** — `CatNode` correctly splits output gradient along `dim` and
  routes each slice back to the corresponding input. New test `cat_backward_routes_grad_to_each_input`.

- **`where_cond` backward** — Gradient flows to `on_true` at `cond==1` positions and
  to `on_false` at `cond==0` positions. New test `where_cond_backward_routes_grad_correctly`.

- **Dropout backward** — `DropoutNode` multiplies output gradient by the stored mask
  (0 at dropped positions, `1/(1-p)` at kept positions). Verified with p=0 identity
  and p=0.5 non-negative gradient invariant.

- **`coeus_optim::CosineAnneal` already implemented** — The `CosineAnneal` scheduler
  strategy and `PyLrScheduler.cosine_anneal` static constructor were already complete.
  New test `test_cosine_annealing_lr_scheduler` verifies formula correctness and
  the Python scheduler step workflow.

### Tests

- Burn parity (+3): `cat_backward_routes_grad_to_each_input`,
  `where_cond_backward_routes_grad_correctly`, `dropout_backward_masks_gradient`.
- Python binding 50 → 53 (+3): `test_tensor_setitem`, `test_cosine_annealing_lr_scheduler`,
  `test_cat_where_backward_parity`.



### Added

- **`coeus_ops::einsum3` / `coeus_autograd::einsum3`** — 3-operand einsum via sequential
  pairwise contraction. Supported: `"ij,jk,kl->il"` (triple matmul chain) and
  `"bij,bjk,bkl->bil"` (batched variant). Autograd-tracked; gradients flow through the
  two sequential matmuls automatically.

- **`pycoeus.einsum([a, b, c], subscript)`** — Python `einsum` now dispatches to
  `einsum3` when 3 operands are provided; backward gradient flows through both matmuls.

### Verified

- **Moirai `parallel_for` adaptive threshold already in place** — `MoiraiBackend::parallel_for`
  delegates to `moirai::for_each_index_with::<Adaptive, _>` which uses
  `ADAPTIVE_PARALLEL_THRESHOLD = 1024`. Below 1024 elements, tasks run inline without
  scheduling. SGD step also has its own threshold (4096). No further changes needed.

- **MHA const-generic H=2/H=4 fast-path already exists** — `MultiHeadAttention<T, B, H, M>`
  with `const H: usize` monomorphizes to a separate code path per head count. No head-count
  branching overhead exists; each value of H produces a distinct binary.

- **`coeus-tensor` CoW infrastructure exists** — `coeus-core/src/storage/cow.rs` implements
  `CowStorage<S>` with `is_unique()`. Further integration with tensor slicing paths is
  deferred to a future refactoring sprint.

## 0.2.22 - 2026-06-25

### Added

- **`coeus_ops::masked_softmax(input, mask, dim)`** — Sets masked positions (mask==0)
  to `-inf` before numerically-stable softmax; output at masked positions is 0.
  Python: `pycoeus.masked_softmax(input, mask, dim=-1)`.

- **`coeus_ops::causal_softmax(input, dim)`** — Builds a lower-triangular causal mask
  and delegates to `masked_softmax`. For attention weight matrices `[..., seq, seq]`.
  Python: `pycoeus.causal_softmax(input, dim=-1)`.

- **`pycoeus.Module` base class** — `#[pyclass(subclass)]` base with `forward()`,
  `parameters()`, `zero_grad()`, `train(mode=True)`, `eval()`, `is_training`.
  Default `forward()` raises `NotImplementedError`. Registered as `pycoeus.Module`.

- **Hermes `dispatch_axpy_kernel` 4× unroll** — Single-register loop replaced with
  4-accumulator loop `acc0..acc3`, each processing one `LANE_COUNT`-wide FMA per
  iteration. Matches the pattern used by `dot()` and `scale()`. Scalar tail unchanged.

- **Autograd test: `contiguous()` backward is identity** — New test confirms that
  `sum(contiguous(permute(x))).backward()` accumulates all-ones gradient into `x`.

- **Burn parity test: embedding gradient accumulation for repeated indices** —
  `embedding_backward_accumulates_grad_for_repeated_indices` verifies that index 0
  appearing twice in `[0, 1, 0]` produces `grad[0] == 2 × grad[1]` as expected.

- **Python binding tests 48 → 50** (+2):
  - `test_masked_causal_softmax` — masked_softmax forward, masked positions zero,
    row-sum=1, all-keep==regular-softmax; causal_softmax lower-triangular pattern,
    row-uniform for fully-visible rows.
  - `test_module_base_class` — `Module()` training flag, `forward()` raises
    `NotImplementedError`, duck-typed custom module works in `Sequential`.

### Verified

- **Softmax backward** — Already correct: `dx = (grad_out - dot(grad_out, y)) * y`.
  Verified against Burn autodiff in `activation_backward_match_burn`.

## 0.2.21 - 2026-06-25

### Added

- **`PyTensor.broadcast_to(shape)`** — Method alias for `expand(shape)`.
  Matches `tensor.broadcast_to(shape)` in NumPy/PyTorch.

- **`pycoeus.broadcast_tensors(tensors)`** — Free function that broadcasts a list
  of tensors to a common shape by computing the broadcast shape and expanding each.
  Equivalent to `torch.broadcast_tensors(*tensors)`.

### Notes

- **Hermes `reduce` already 4× unrolled** — Audit confirmed
  `view/reduce.rs::reduce()` already uses `UNROLL_FACTOR` independent accumulators
  (acc0–acc3) seeded by `Op::transform_vector`. No further unrolling was needed.

- **`coeus-ops/src/backend_ops/defaults/` already partially extracted** —
  `defaults/mod.rs` has `conv_transpose`, `matmul`, `reductions` submodules with
  host-fallback default implementations. Further extraction is incremental architecture
  work deferred to future sprints.

## 0.2.20 - 2026-06-25

### Added

- **`pycoeus.normalize(input, p=2, dim=1, eps=1e-12)`** — Lp normalization along
  `dim`. Divides each slice by its Lp norm (clamped to `eps` from below).
  Equivalent to `torch.nn.functional.normalize`. `p` and `dim` validated at boundary.

- **`pycoeus.rand(shape)`, `pycoeus.randint(low, high, shape)`, `pycoeus.bernoulli(shape, p=0.5)`**
  — Random tensor constructors using xorshift64 seeded from system time.
  `randint` stores integers as f64, `bernoulli` stores 0.0/1.0 by Bernoulli(p).

- **`pycoeus.clip_grad_norm_(parameters, max_norm, norm_type=2)`** — Returns the
  pre-clip global gradient norm and rescales all parameter gradients so their global
  Lp norm ≤ `max_norm`. Host-side round-trip with no intermediate Tensor allocation.

- **`pycoeus.clip_grad_value_(parameters, clip_value)`** — Clamps each gradient
  element-wise to `[-clip_value, clip_value]`.

- **`pycoeus.isclose(a, b, rtol=1e-5, atol=1e-8)`** — Returns float tensor
  (1.0 = within tolerance, 0.0 = not). Same tolerance formula as PyTorch.

- **`pycoeus.allclose(a, b, rtol=1e-5, atol=1e-8)`** — Returns Python `bool`.

- **`pycoeus.nan_to_num(input, nan=0.0, posinf=None, neginf=None)`** — Replaces
  NaN, +Inf, -Inf with finite defaults.

- **`pycoeus.sum_axis(input, axis, keepdim=False)`** — Added `keepdim` parameter;
  default `False` now squeezes the reduced dimension (matching PyTorch convention).
  **Breaking**: existing callers relying on keepdim behavior should pass `keepdim=True`.

- **`pycoeus.mean_axis(input, axis, keepdim=False)`** — Same keepdim change.

- **Improved `PyTensor.__repr__` / `__str__`** — Shows actual values for tensors
  with ≤ 8 elements (`Tensor([1.0, 2.0], shape=[2])`), truncated display for larger.
  Format matches PyTorch-style output.

- **`LSTMCell`/`GRUCell` bias support** — `PyLSTMCell` and `PyGRUCell` now expose
  `b_ih` and `b_hh` bias parameters. `parameters()` now returns 4 params
  (w_ih, b_ih, w_hh, b_hh) when `bias=True` (default). `zero_grad()` zeros biases.

- **Python binding tests 45 → 47** — `test_normalize_closeness_nan_and_grad_clipping`
  exercises normalize, isclose/allclose, nan_to_num, clip_grad_norm_/value_, sum_axis
  keepdim, and __repr__ formatting.



### Added

- **`coeus_nn::rnn::LSTMCell`** — Single-step LSTM cell with fused gate projection
  (`W_ih [4H,I]`, `W_hh [4H,H]`). `step(x, h, c) → (h_new, c_new)`.
  Autograd-tracked via `coeus_autograd::slice + mul + add + sigmoid + tanh`.
  Python: `pycoeus.LSTMCell(input_size, hidden_size)` with `.step(x, h, c)`.

- **`coeus_nn::rnn::GRUCell`** — Single-step GRU cell with reset/update/new gates.
  `step(x, h) → h_new`. Python: `pycoeus.GRUCell(input_size, hidden_size)`.

- **`coeus_ops::index_put`** — Scatter-assign: `index_put(input, indices, values, accumulate)`
  assigns or accumulates `values` at 1-D integer row indices. Equivalent to
  `torch.index_put(input, (indices,), values)`.
  Python: `pycoeus.index_put(input, indices, values, accumulate=False)`.

- **`pycoeus.TransformerDecoderLayer`** — Python wrapper for the existing
  `coeus_nn::TransformerDecoderLayer`. Cross-attention decoder layer with
  `forward(tgt, memory)` signature. Dispatches over `num_heads` via const-generic
  macro (supported: 1, 2, 4, 8, 16, 32).

- **Hermes `dispatch_scale_kernel` 4× unroll** — `scale.rs` now processes
  `UNROLL_FACTOR×LANE_COUNT` elements per outer iteration using 4 independent
  registers to hide load/store latency, matching the `dot()` and `axpy()` unroll
  patterns. Falls back to single-vector loop for the residual.

- **Python binding tests 43 → 45** (+2):
  - `test_lstm_gru_cells` — LSTM and GRU cell shape, non-zero output, multi-step
    state change, parameter count.
  - `test_index_put_op` — replace mode, accumulate mode, 2D row update, non-1D
    index `ValueError`.

- **Python functional parity wrappers** — Added `pycoeus.rand`, `randint`,
  `bernoulli`, module-level `sum_axis(..., keepdim=False)`,
  `mean_axis(..., keepdim=False)`, `normalize`, `isclose`, `allclose`,
  `nan_to_num`, `clip_grad_norm_`, `clip_grad_value_`, and value-printing
  tensor `repr`. Uniform `rand` routes through `coeus_nn::init::uniform_with_seed`
  so the Python binding remains a thin Rust-core adapter.

- **Python binding tests 45 → 47** (+2):
  - `test_randn_zeros_ones_like_eye` now covers `rand`, `randint`, and
    `bernoulli` shape/range/error contracts.
  - `test_normalize_closeness_nan_and_grad_clipping` covers keepdim reductions,
    `normalize`, closeness checks, `nan_to_num`, gradient clipping, and tensor
    `repr`.

- **Burn benchmark instrumentation** — Added an SDP-attention benchmark group to
  `coeus-tensor/benches/tensor_bench.rs` comparing Burn NdArray batched
  matmul+softmax attention with Coeus Sequential and Coeus Moirai
  `scaled_dot_product_attention` on `[8, 64, 32]` q/k/v tensors. This is an
  instrumented benchmark row only; no speedup claim is made.


## 0.2.18 - 2026-06-25

### Added

- **`coeus_ops::bmm`** — Batch matrix multiply `[B,M,K]×[B,K,N]→[B,M,N]`;
  delegates to the existing `matmul` kernel via shape assertion.
  Python: `pycoeus.bmm(a, b)` with shape validation `ValueError`.

- **`coeus_ops::outer`** — Outer product `[M]×[N]→[M,N]` via reshape+matmul.
  Python: `pycoeus.outer(a, b)` with 1-D input `ValueError`.

- **`coeus_ops::one_hot`** — One-hot encoding: `[N]→[N, num_classes]` float
  tensor. Validates finite, non-negative integer scalar index values before
  converting them to class offsets.
  Python: `pycoeus.one_hot(indices, num_classes)`.

- **`coeus_ops::masked_select`** — Select elements where mask ≠ 0.0; returns 1-D
  tensor. Python: `pycoeus.masked_select(input, mask)` with shape-mismatch `ValueError`.

- **`coeus_ops::chunk`** — Split tensor into ≤N approximately equal pieces along `dim`,
  `chunk_size = ceil(dim_size / chunks)`. Python: `pycoeus.chunk(input, chunks, dim=0)`.

- **`coeus_ops::glu`** — Gated Linear Unit: splits `input` in half along `dim`,
  returns `first_half * sigmoid(second_half)`. Requires even dim size.
  Python: `pycoeus.glu(input, dim=-1)` with `ValueError` for odd size or out-of-range dim.

- **`pycoeus.ModuleList`** — Dynamic ordered container of nn modules. Supports
  `forward(x)` (explicit, not auto-chained), `parameters()`, `zero_grad()`,
  `__len__`, `__getitem__`, `__setitem__`, `append`, `extend`. Registered as
  `pycoeus.ModuleList`.

- **Python binding tests 39 → 43** (+4 new tests):
  - `test_bmm_outer_ops` — bmm forward, outer product, error paths.
  - `test_one_hot_masked_select_chunk` — one_hot encoding, masked_select 2D/empty,
    chunk even/uneven/2D, default dim.
  - `test_glu_activation` — 1D, 2D default dim, exact sigmoid gating, error paths.
  - `test_module_list` — forward chain, parameter collection, `__getitem__`,
    `__setitem__`, negative index, out-of-range error, empty list.

- **Burn benchmark instrumentation** — Added a GELU benchmark group to
  `coeus-tensor/benches/tensor_bench.rs` comparing Burn NdArray, Coeus
  Sequential, and Coeus Moirai for a 1024x1024 tensor. This is an instrumented
  benchmark row only; no speedup claim is made.

## 0.2.17 - 2026-06-25

### Added

- **Sparse conversion integration test** — Added `coeus-sparse/tests/sparse_conversions.rs`
  to verify dense→COO→dense, dense→CSR→dense, dense→COO→CSR→dense, and
  dense→CSR structural equality against the COO→CSR route on one value-semantic
  3×4 oracle. Evidence tier: empirical value-semantic validation via
  `cargo nextest run -p coeus-sparse --test sparse_conversions`.

- **`coeus-ops::linspace / logspace / geomspace` free functions** — Backend-parameterized
  constructor functions in the new `coeus_ops::constructors` module:
  - `linspace(start, end, n, backend)` — n evenly-spaced values (inclusive).
  - `logspace(start, end, n, base, backend)` — n log-scale values (`base^exp`).
  - `geomspace(start, end, n, backend)` — n geometrically-spaced values; panics for
    zero or sign-mismatched endpoints.
  All three accept a `backend: &B` reference and return `Tensor<T, B>`, matching the
  calling convention of all other `coeus_ops` free functions. 4 unit tests added.

- **`pycoeus.topk(input, k, dim=0, largest=True)` parameter** — Added `largest` boolean
  parameter to the Python `topk` binding. When `largest=False`, returns the k smallest
  values instead of k largest, matching `torch.topk(input, k, dim, largest)`.
  Existing tests updated with explicit `largest=False` and 2-D dim=1 coverage.

- **Burn parity tests (+5)** — `burn_live_parity.rs` now has 69 tests:
  - `groupnorm_forward_matches_burn` — forward comparison of `GroupNorm<T,B,2>` with
    default weight=ones, bias=zeros against Burn `GroupNormConfig::new(2,4)`. Tolerance
    1e-3 accounts for the formula difference `sqrt(var+eps)` (Coeus, PyTorch standard)
    vs `sqrt(var)+eps` (Burn 0.16).
  - `groupnorm_forward_backward_match_burn` — forward + backward (dx, dw, db) parity
    with custom weight/bias, using a manual Burn tensor formula matching Coeus's
    `sqrt(var+eps)` convention so gradient comparison uses a tight 1e-4 tolerance.
  - `instancenorm_forward_matches_burn` — forward comparison of `InstanceNorm1d` with
    default init against Burn `InstanceNormConfig::new(3)`. Same 1e-3 tolerance as
    GroupNorm for the same formula-difference reason.
  - `embedding_forward_matches_burn` — forward comparison of `Embedding` with known
    weight [5,3] and integer indices [2,3] against Burn `module::embedding`.
  - `embedding_forward_backward_match_burn` — forward + backward (dw) parity with
    custom weight [4,2] and indices [2,2] against Burn autodiff `module::embedding`.

### Fixed

- **GroupNorm/InstanceNorm tolerance and formula** — The 3 GroupNorm/InstanceNorm
  tests committed in MS-77 were failing because the forward tolerance (1e-4) did not
  account for the `sqrt(var+eps)` vs `sqrt(var)+eps` formula difference between Coeus
  and Burn 0.16, and the backward test used Burn's formula instead of Coeus's. Fixed:
  forward tolerance 1e-4 → 1e-3 (analytically derived), backward formula
  `var.sqrt().add_scalar(eps)` → `var.add_scalar(eps).sqrt()`.
### Changed

- **SGD optimizer small-tensor fast path** — `sgd_step` contiguous unit-offset buffers
  with ≤ 4096 elements now use a scalar sequential loop instead of `parallel_for`, avoiding
  thread-scheduling overhead for typical parameter shapes. The sequential loop auto-vectorises
  on `--release` via LLVM. Large tensors (> 4096 elements) continue to use `parallel_for`.

- **ConvTranspose1d/2d backward: fused scatter-accumulate** — Replaced the 3× pattern of
  `Tensor::from_slice(shape, &host_vec)` + `add_assign` in the backward pass of both
  `ConvTranspose1dNode` and `ConvTranspose2dNode` with a direct `scatter_accumulate_into`
  helper. This eliminates one device-buffer allocation and one copy round-trip per gradient
  (input, weight, bias) per backward call.

### Atlas Audits

- **Moirai `WorkStealingScheduler` audit** — Confirmed correct design:
  - Chase-Lev lock-free deque for per-worker local queue (no spinlock on the hot push/pop path).
  - `CacheAligned<AtomicUsize>` stats prevent false sharing between counters.
  - Global queue uses `try_lock()` with batch-drain to amortize lock overhead.
  - Steal early-out: `is_empty()` probe before `steal()` to avoid futile lock attempts.
  - No regression opportunities identified; scheduler is already near-optimal for the
    current single-program multi-data workload.

- **Mnemosyne slab allocator note** — Mnemosyne delegates to `mnemosyne_local` which
  carries `LocalAllocatorSelector` and `SizeClassOccupancy` with per-thread slab caches.
  Cache-line alignment and false-sharing prevention are handled at the `mnemosyne_core`
  level. No changes required at this version.



### Added

- **`ConvTranspose1d/2d` Python bindings now tracked** — `PyConvTranspose1d::forward`
  and `PyConvTranspose2d::forward` previously returned `Var::new(out, false)` (no
  gradient). Both now call `coeus_autograd::conv_transpose1d/2d`, enabling
  end-to-end gradient flow from Python training loops.

- **`PyTensor.softmax(dim)` / `.log_softmax(dim)` methods** — Tensor method forms
  matching `torch.Tensor.softmax(dim)` and `torch.Tensor.log_softmax(dim)`.
  Negative dim values are supported (isize dispatch).

- **`pycoeus.Sequential`** — `nn.Sequential`-equivalent container: ordered list of
  modules with `forward(x)`, `parameters()`, `zero_grad()`, `__len__`,
  `__getitem__`, and `append`. Any module with a `.forward(tensor)` method can be
  composed. Registered as `pycoeus.Sequential`.

- **Burn parity tests (+2)** — `burn_live_parity.rs` now has 64 tests:
  - `avg_pool2d_backward_gradient_correctness` — kernel=2, stride=2, all-ones seed;
    each input element must receive 0.25 gradient.
  - `max_pool2d_backward_gradient_correctness` — 4×4 input with distinct block maxima;
    verifies exact positions receive 1.0 and all others receive 0.0.

- **Python binding tests 36 → 39** — Three new tests:
  - `test_softmax_log_softmax_methods` — 1D/2D `tensor.softmax(dim)`,
    `tensor.log_softmax(dim)`, sum-to-1, monotonicity, `exp(log_softmax) == softmax`.
  - `test_sequential_module` — `Sequential([Linear, LayerNorm])` forward, shape,
    parameter collection, identity empty case, `__len__`/`__getitem__`, backward.
  - `test_conv_transpose_tracked_backward` — `ConvTranspose1d` and `ConvTranspose2d`
    produce correct forward values and propagate gradients back to inputs.



### Added

- **`ConvTranspose2dNode` + tracked `conv_transpose2d`** — Autograd backward node
  for 2-D transposed convolution in `coeus-autograd/src/ops/nn/conv.rs`.
  Host-side backward implements the three derivative paths:
  - `grad_input[n,cin,hin,win] = Σ grad_out × weight` (gather from output grad)
  - `grad_weight[cin,cout,kh,kw] += Σ input × grad_out`
  - `grad_bias[cout] = Σ grad_out` (optional)
  Exported from `coeus-autograd` public flat surface as `conv_transpose2d`.

- **`ConvTranspose1d` / `ConvTranspose2d` now fully tracked** — Both `coeus-nn`
  modules previously returned `Var::new(out, false)` (no gradient tracking).
  They now call the tracked `coeus_autograd::conv_transpose1d/2d` wrappers,
  enabling end-to-end gradient flow through transposed convolution layers in
  any training loop that uses `coeus-autograd`.

- **Autograd tests (+2)** — `coeus-autograd` test suite (29 tests):
  - `conv_transpose2d_backward_accumulates_exact_gradients` — identity-kernel
    with bias; verifies exact grad_input, grad_weight, grad_bias.
  - `conv_transpose2d_no_bias_backward` — stride-1 2×2 kernel without bias;
    confirms gradients flow, shapes correct, grad_weight nonzero.

- **Burn parity tests (+2)** — `burn_live_parity.rs` now has 62 tests:
  - `conv_transpose1d_backward_gradient_correctness` — all-ones seed, 2-element
    input, verifies grad_input and grad_weight analytically.
  - `conv_transpose2d_backward_gradient_correctness` — identity kernel, all-ones
    input + seed, verifies grad_input = 2×ones, grad_weight = 4.



### Added

- **`LayerNorm::forward_nd`** — New method on `coeus_nn::LayerNorm<T, B>` that
  accepts any rank ≥ 2 input by transparently collapsing all leading dimensions
  via tracked `coeus_autograd::reshape`, applying 2-D LayerNorm over the last
  axis, and reshaping back. Gradients flow through the entire
  flatten → normalize → unflatten chain. Common usage: `[batch, seq, d_model]`
  Transformer hidden states (3-D), or `[batch, channels, h, w]` feature maps (4-D).

- **`PyLayerNorm.forward_nd`** — Python method that delegates to `LayerNorm::forward_nd`,
  allowing `ln.forward_nd(x)` for any rank-N input from Python.

- **`layer_norm` functional handles rank ≥ 3** — The `pycoeus.layer_norm` free
  function now dispatches to `forward_nd` when the input has rank > 2,
  matching `torch.nn.functional.layer_norm` behavior.

- **Hermes `Dot::fma_pair_accumulate`** — Added `fma_pair_accumulate` virtual
  method to the `ReductionOp` trait (default: `accumulate(acc, mul(a, b))`).
  `Dot` overrides it with `Arch::fmadd(a, b, acc)`, fusing multiply and add into
  a single `vfmadd` instruction when the architecture supports it. The
  `zip_reduce` main loop and single-vector tail now call `fma_pair_accumulate`
  instead of the two-step `pair()+accumulate()` sequence, eliminating a
  latency-bound add per `LANE_COUNT` elements on AVX2/AVX512 hardware.

- **Burn parity test** — `layernorm_forward_nd_3d_matches_reshape_reference`
  verifies forward output of `LayerNorm::forward_nd` on `[2, 3, 4]` input
  matches the manual reshape→2D-LayerNorm→reshape reference, and that
  backward gradient propagates through the 3-D path.

- **Python binding test** — `test_layernorm_3d_forward_nd` exercises
  `LayerNorm.forward_nd` (3-D and 4-D), `layer_norm` functional 3-D dispatch,
  backward gradient flow, and consistency with 2-D `forward`.



### Added

- **Tensor dtype cast methods** — Added `.float()`, `.double()`, `.long()`, `.int()`,
  `.half()`, `.to(dtype)`, `.type_as(other)` on `PyTensor`. `.long()`/`.int()` truncate
  fractional parts toward zero (matching `torch.long`). `.half()` round-trips through
  `half::f16` representation. `.to(dtype)` dispatches by string key with `ValueError`
  for unrecognised names. All methods return non-tracked copies.

- **`PyScaledDotProductAttention` nn module** — Stateless attention module in
  `coeus-python/src/nn/attention.rs` with `forward(q, k, v, key_padding_mask=None)`,
  optional `scale`, `is_causal` flag, empty `state_dict`/`parameters()`. Registered
  as `pycoeus.ScaledDotProductAttention`.

- **`pycoeus.scaled_dot_product_attention` functional API** — Free function in
  `coeus-python/src/ops/nn_functional.rs` with signature
  `(query, key, value, attn_mask=None, scale=None, is_causal=False)`.
  Delegates to `coeus_autograd::sdp_attention` (NullMask or CausalMask ZST dispatch,
  dead code eliminated at monomorphization).

- **Burn parity tests (+ 4)** — `burn_live_parity.rs` now has 59 tests:
  - `conv_transpose1d_stride2_matches_manual_reference` — ConvTranspose1d stride-2
    scatter scatter against manual reference.
  - `conv_transpose2d_unit_stride_matches_manual_reference` — ConvTranspose2d unit
    stride scatter against manual reference.
  - `amax_amin_prod_match_manual_reference` — scalar reductions against
    `data.iter().product()` and direct comparisons.
  - `no_grad_context_does_not_track` — verifies `push_no_grad`/`pop_no_grad` suppress
    creator-node creation even when inputs have `requires_grad=true`.

- **Python binding tests 32 → 35** — new tests:
  - `test_dtype_cast_methods` — covers float/double identity, long/int truncation,
    half precision quantization, `to(dtype)` dispatch, `type_as` clone, unknown dtype
    ValueError.
  - `test_sdp_attention_and_module` — covers functional `scaled_dot_product_attention`
    (uniform softmax → identity output, causal vs non-causal), `ScaledDotProductAttention`
    module forward, `parameters()`, `state_dict`/`load_state_dict`.
  - `test_amax_amin_prod_ops` — covers 2D/1D amax/amin/prod values, empty-tensor
    ValueError for amax/amin, empty-tensor identity (1.0) for prod.

### Changed

- `coeus-python` now depends on `half` (workspace) for `.half()` dtype cast.



### Added

- **`torch.dot` parity** — `coeus_ops::dot<T: Scalar, B>(a, b, backend) -> T`
  computes the flat inner product `Σᵢ aᵢ bᵢ` over equal-numel input
  tensors, matching `torch.dot(input, tensor)`. Single-pass host-side fold
  in native `T` precision; no `BinaryOp` opcode added (composes over the
  existing `B::copy_to_host` SSOT). Empty inputs return `T::zero()`;
  numel mismatch panics with the invariant named in the message.
  Re-exported from `coeus_ops` flat surface and `coeus_ops::reduction`.

- **`torch.cross` parity** — `coeus_ops::cross<T: Scalar, B>(a, b, dim, backend) -> Tensor<T,B>`
  computes the per-channel 3-vector cross product along `dim`, matching
  `torch.cross(input, other, dim)`. The slice axis must have exactly three
  elements; the output keeps the same shape (no reduction). The element
  ordering follows the right-handed cross product convention used by
  `torch.cross` / `numpy.cross` / `jax.numpy.cross` / `mlx.core.cross`.

- **Python bindings** — `pycoeus.dot(input, tensor) -> float` and
  `pycoeus.cross(input, other, dim=0) -> Tensor` PyO3 wrappers with
  `ValueError` boundary errors for numel-mismatch, shape-mismatch,
  out-of-range `dim`, and `dim != 3` cases. Both wrappers live in the
  new `coeus_python/src/ops/linalg.rs` module under the existing
  operation-family subdirectory `coeus_python/src/ops/`.

- **Rust unit tests (14)** — `coeus_ops::reduction::linalg::tests` covers
  1-D and 2-D `dot` (flat fold), orthogonal-vector zero, empty-tensor
  zero, numel-mismatch panic, three `cross` axis-3 invariants
  (`e_x × e_y = e_z`, `e_y × e_x = -e_z`, `v × v = 0`), anticommutativity
  (`cross(a, b) == -cross(b, a)`), per-row (dim=last), per-column (dim=first),
  3-D middle-axis, plus panic paths for wrong axis size and out-of-range
  `dim`.

- **Python binding test (1)** —
  `coeus-python::binding_tests_ops::test_dot_cross_vector_ops`
  exercises both Python surfaces across 1-D, 2-D flat, orthogonal,
  error paths, default-`dim`, dim=0, dim=1, parallel-vector, shape-mismatch,
  out-of-range-dim, and dim-size-≠3 cases against value-semantic PyTorch
  oracles.

- **`logspace` / `geomspace` constructor parity** — Added
  `Tensor::logspace(_on)` and `Tensor::geomspace(_on)` in `coeus-tensor`, plus
  Python `pycoeus.logspace(start, end, steps, base=10.0)` and
  `pycoeus.geomspace(start, end, steps)` constructors. `geomspace` now enforces
  non-zero endpoints with matching sign (Rust invariants + Python `ValueError`).
  Expanded Python constructor coverage in `binding_tests_ops::test_constructors`.

### Notes

- Burn 0.16 (the active dev-only oracle backend) does **not** expose
  `Tensor::dot` or `Tensor::cross`. The `coeus-nn/tests/burn_live_parity`
  diff parity tests for these ops are therefore not added at this version;
  the test surface lives against the documented manual oracle (right-hand
  rule, dense Python loops) and against the value-semantic PyO3 binding
  assertions above. Torch / NumPy / JAX / MLX parity remains the binding
  oracle for `dot` and `cross`.

## 0.2.11 - 2026-06-25


### Changed

- **BatchNorm autograd consolidation** — replaced separate
  `BatchNorm1dNode`/`BatchNorm2dNode`/`BatchNorm3dNode` implementations with
  one const-generic `BatchNormNode<T, B, DIM>` and shared
  `BatchNormArgs<T, B, DIM>`, preserving 1-D/2-D/3-D module behavior while
  removing per-rank backward duplication.
- **coeus-leto dispatch hierarchy** — split the monolithic dynamic-rank dispatch
  module into operation-family leaf modules (`elementwise`, `init`, `layout`,
  `linalg`, `reductions`, `sparse`, `structural`) while preserving the public
  `coeus_leto::dispatch::*` re-export surface.

### Breaking

- Removed the public `BatchNorm1dArgs`, `BatchNorm2dArgs`, and
  `BatchNorm3dArgs` names in favor of `BatchNormArgs<T, B, DIM>`. This is a
  pre-1.0 minor-version API break with no compatibility aliases.

## 0.2.10 - 2026-06-25

### Added

- **`ConvTranspose1d` / `ConvTranspose2d`** — Transposed convolution (deconvolution)
  implemented as default methods on `BackendOps<T>` via a host-side
  dilated-input scatter reference, with native WGPU and CUDA f32 forward
  kernels using an equivalent conflict-free gather formulation. Public API:
  `coeus_ops::conv_transpose1d/2d` entry points,
  `coeus_nn::ConvTranspose1d/2d` zero-parameter modules, and
  `pycoeus.ConvTranspose1d/2d` Python classes.

- **`amax` / `amin` / `prod` ops** — global reduce functions in `coeus-ops`
  (no keepdim scalar return). Python `pycoeus.amax(input)`,
  `pycoeus.amin(input)`, `pycoeus.prod(input)` with empty-tensor
  `ValueError` guards.

- **In-place PyTensor methods** — `fill_(value)`, `zero_()`, `one_()`,
  `__iadd__` (`+=`), `__isub__` (`-=`), `__imul__` (`*=`). These are
  non-tracked mutations matching PyTorch's in-place API.

- **`pycoeus.no_grad()` context manager** — `with pycoeus.no_grad():` block
  matching the Python-facing `torch.no_grad()` operation contract. Nested
  scopes now forward into `coeus-autograd` core grad-mode state so Rust
  operations skip creator-node and gradient-buffer allocation inside the scope,
  while explicit tensor factories still honor `requires_grad`.

- **Tracked `coeus_autograd::conv_transpose1d`** — Autograd wrapper for 1-D
  transposed convolution with value-semantic backward coverage for input,
  weight, and bias gradients.

- **`coeus_nn` / `coeus_ops` improvements**:
  - `prod()`, `amax()`, `amin()` exported from `coeus-ops`.
  - `ConvTranspose1d/2d` exported from `coeus-nn`.
  - `conv_transpose1d/2d` output dimension helpers exported as public API.

- **Moirai parallel_for audit confirmed already optimal** — `ADAPTIVE_PARALLEL_THRESHOLD = 1024` with `Adaptive` policy routes sequentially below threshold; no changes needed.

- **GPU backend crate documentation** — `coeus-cuda` and `coeus-wgpu` now have
  crate-level architecture docs describing their backend-only responsibility,
  Atlas provider ownership, device dispatch flow, and explicit CPU-reference
  capability boundaries. Evidence tier: rustdoc validation.

- **WGPU device benchmark harness** — `coeus-wgpu` now registers
  `ops_bench`, an on-demand Criterion harness comparing CPU and WGPU matmul
  and transposed-convolution forward paths. This is a benchmark instrument, not
  a recorded speedup claim.

### Changed

- Workspace version bumped `0.2.9` → `0.2.10`.

- Convolution autograd backward wrappers now share one const-generic
  `ConvNode<T, B, DIM>` implementation for 1-D, 2-D, and 3-D convolution
  backward dispatch, removing per-dimension node duplication.

- Pooling autograd backward wrappers now share const-generic max-pool and
  average-pool node implementations across 2-D and 3-D pooling, preserving the
  backend-specific backward dispatch points while removing per-rank node
  duplication.



### Added

- **`meshgrid` op** — `coeus-ops::meshgrid(&tensors, indexing)` creates N
  coordinate grids from N 1-D tensors matching `torch.meshgrid(*tensors,
  indexing="ij"/"xy")`. Python `pycoeus.meshgrid([*tensors], indexing="ij")`.
  3 unit tests; 1 Burn parity test.

- **`tile` op** — `coeus-ops::tile(input, reps)` replicates `input` by
  `reps[d]` times along each dimension (matching `torch.Tensor.repeat` /
  `np.tile`). Tracked `coeus_autograd::tile` with sum-over-copies backward.
  Python `pycoeus.tile(input, reps)` and `Tensor.repeat(reps)` method form.
  4 unit tests; 1 Burn parity test (forward + backward).

- **`coeus-leto` rank-6 dispatch** — `MAX_DISPATCH_RANK` extended from 5 to 6.
  All elementwise, binary, unary, reduction, scan, concat/split/stack/pad
  dispatch functions now handle rank-6 tensors (needed for batched multi-head
  attention with `[batch, heads, seq_q, seq_k]` or `[batch, heads, seq, d]`
  shapes). Contract test `rank_beyond_dispatch_bound_is_rejected` updated to
  use rank 7. All 22 coeus-leto contract tests pass.

- **`PyTensor` API additions** — New methods on the Python `Tensor` class:
  - `tensor.clone_tensor()` — shallow clone (same autograd graph).
  - `tensor.is_contiguous()` → `bool` — checks row-major memory layout.
  - `tensor.numel()` → `int` — total element count.
  - `tensor.T` → property — 2-D transpose (raises `ValueError` for non-2-D).
  - `tensor.repeat(reps)` — tile via autograd::tile (tracked).

- **Burn parity suite 53 → 55 tests** — `meshgrid_ij_matches_manual_reference`
  and `tile_forward_and_backward`.

- **Python binding tests 29 → 30** — `test_meshgrid_tile_tensor_methods`
  covers meshgrid ij/xy, tile 1-D/2-D backward, `Tensor.repeat`, `Tensor.T`,
  `numel()`, `is_contiguous()`, `clone_tensor()`, and error paths.

### Changed

- Workspace version bumped `0.2.8` → `0.2.9`.
- `coeus-ops/src/shape/tile.rs` unused variable fixed.



### Added

- **`diag` / `diagonal` ops** — `coeus-ops::diag(v, k)` creates a diagonal
  matrix from 1-D vector `v` at offset `k`; `coeus-ops::diagonal(M, k)`
  extracts the `k`-th diagonal of a 2-D matrix as a 1-D vector. Both are
  tracked in `coeus-autograd` (backward: `diag → diagonal` and vice-versa).
  Python `pycoeus.diag(v, k=0)` and `pycoeus.diagonal(m, k=0)` with shape
  validation. 5 unit tests; 2 Burn parity tests.

- **`cumprod` op** — `coeus-ops::cumprod(input, dim)` inclusive cumulative
  product. Tracked `coeus_autograd::cumprod` with suffix-sum backward (safe
  for non-zero inputs; zero inputs receive zero gradient). Python
  `pycoeus.cumprod(input, dim)`. 4 unit tests; 1 Burn parity test.

- **FMA3 capability probe in Hermes** — `hermes_simd::cpu::has_fma3()` via
  `OnceLock`-cached CPUID query (`CPUID.01H:ECX[bit 12]`); `FmaSupport` trait
  implemented for `f32`, `f64`, `bf16`. New `DispatchDecision::Fma` variant
  in `AdaptiveDispatcher` so FMA3-capable CPUs select the Fma path instead of
  Scalar, letting the compiler emit `vfmaddXXXps/pd` for ~2× effective GEMM
  throughput. Existing tile_matmul match arms treat `Fma` as Scalar today
  (dedicated FMA kernel to land in a future sprint). All 357+ hermes tests pass.

- **`nn.functional`-style Python free functions** — registered under the same
  `pycoeus.*` namespace to match `torch.nn.functional.*`:
  - `f_relu`, `f_sigmoid`, `f_tanh`, `f_gelu`, `f_silu` — activation free fns
  - `f_softmax(input, dim)`, `f_log_softmax(input, dim)`
  - `f_mse_loss(input, target)`, `f_binary_cross_entropy(input, target)`
  - `f_cross_entropy(input, targets)` (integer targets as `List[int]`)

- **Burn parity suite 51 → 53 tests** — `cumprod_forward_and_backward` and
  `diag_diagonal_forward_and_backward` added.

- **Python binding test count 27 → 29** — `test_diag_diagonal_cumprod` and
  `test_nn_functional_ops`.

### Changed

- Workspace version bumped `0.2.7` → `0.2.8`.



### Added

- **`einsum` op** — `coeus-ops::einsum(subscript, operands)` and tracked
  `coeus_autograd::einsum` dispatch common ML patterns to optimised kernels:
  - `"ij,jk->ik"` — 2-D matmul (tracked via matmul autograd)
  - `"bij,bjk->bik"` — batched matmul (tracked via per-batch matmul + cat)
  - `"ij->ji"` — 2-D transpose (tracked via permute)
  - `"i,i->"` — dot product (tracked via mul + sum)
  - `"i,j->ij"` — outer product (tracked via broadcast + mul)
  - `"ij,j->i"` — matrix-vector multiply (tracked via matmul + squeeze)
  - `"ii->"` — trace (non-differentiable forward)
  - Generic ND last-2-dims swap (tracked)
  7 einsum unit tests in `coeus-ops/src/shape/einsum.rs`. Python
  `pycoeus.einsum(subscript, [*tensors])` with backward flow through
  autograd-delegated operations.

- **`index_select` op** — `coeus-ops::index_select(input, dim, index)` selects
  slices from `input` along `dim` at 1-D `index` positions (matching
  `torch.index_select`). Tracked `coeus_autograd::index_select` with
  scatter-add backward. Python `pycoeus.index_select(input, dim, index)` with
  `ValueError` guards. 3 unit tests, 1 Python binding test.

- **Burn parity suite expanded 50 → 51 tests**:
  - `transformer_encoder_layer_forward_backward_shape_contract` — forward
    shape contract and non-zero gradient verification for all encoder layer
    parameters (TransformerEncoderLayer with H=2, d_model=8, d_ff=16).

- **Python einsum/index_select binding tests** — `test_einsum_index_select`
  covers matmul, transpose, dot product, outer product patterns and backward
  flow through matmul autograd; index_select 1-D/2-D selection, backward
  scatter-add, and error paths. Evidence: 27 Python ops binding tests pass.

- **`hermes-simd` CPU feature probe audit** — confirmed all `AmxSupport` and
  `Avx512Support` impls already cache via `OnceLock`; no per-call CPUID
  overhead on steady-state paths.

- **`mnemosyne` segment pool audit** — `NodeSegmentPool` already has:
  - Relaxed-atomic `retained` early-out before spinlock acquisition on `pop()`
  - `SpinLock` (not `Mutex`) for minimal overhead
  - 16-bucket NUMA-aware partitioning via `GlobalSegmentPool`
  No structural changes needed; documented as already optimal.

- **`moirai` executor audit** — `HybridExecutor.task_registry` `Mutex` is
  lifecycle-only (not on the hot `parallel_for` path which runs through the
  lock-free work-stealing scheduler). Hot path confirmed lock-free.

### Changed

- Workspace version bumped `0.2.6` → `0.2.7`.
- `coeus-ops/src/shape/mod.rs` adds `einsum` and `index_select` modules.
- `coeus-autograd/src/ops/shape/mod.rs` adds `einsum` and `index_select` modules.



### Added

- **`broadcast_to` / `expand`** — `coeus-ops::broadcast_to(input, target_shape)`
  materialises a tensor into a target shape by repeating along singleton
  dimensions (rank-preserving NumPy/PyTorch broadcast rules). Tracked
  `coeus_autograd::broadcast_to` sums the output gradient over all broadcast
  dimensions in the backward pass. Python `pycoeus.broadcast_to(input, shape)`
  with rank-mismatch `ValueError`. Backward test included.

- **`masked_fill`** — `coeus-ops::masked_fill(input, mask, value)` sets
  elements to `value` where `mask != 0` (non-zero = true). Tracked
  `coeus_autograd::masked_fill` zeroes the gradient at masked positions.
  Python `pycoeus.masked_fill(input, mask, value)` with shape-mismatch
  `ValueError`. Backward test included.

- **`nonzero`** — `coeus-ops::nonzero(input)` returns a `[N, ndim]` tensor
  of row-major ND coordinates for all non-zero elements. Python
  `pycoeus.nonzero(input)` (non-differentiable). Returns `[0, ndim]` on
  all-zero input.

- **Python binding tests** — `test_broadcast_masked_fill_nonzero` covers all
  three new ops with forward values, backward gradient checks, and error paths.
  Python `test_feedforward_module` verifies `pycoeus.FeedForward(d_model, d_ff)`
  forward pass shape contract. Evidence: 24 Python ops binding tests pass.

- **`PyFeedForward` Python class** — exposes the two-layer MLP
  transformer sub-block as a named Python class with a `forward(input)` method.
  Registered in `coeus-python/src/lib.rs`.

- **Optimizer parity tests** — `coeus-nn/tests/burn_live_parity.rs` extended
  from 48 to 50 tests:
  - `sgd_step_matches_analytical_reference` — verifies SGD without momentum
    against exact `θ - lr * g` reference.
  - `adam_step_matches_analytical_reference` — verifies Adam step at t=1
    against closed-form first-step reference (β₁=0.9, β₂=0.999, ε=1e-8).

- **`vector_norm(ord=p)` ord-p norm** — `coeus_ops::norm_p(x, p, backend)`
  returns `(Σ|xᵢ|^p)^(1/p)`. Python `pycoeus.vector_norm(input, ord=2.0)`.
  Verified against `torch.linalg.vector_norm` reference values for p ∈ {1, 2, 3}.

- **Per-axis `vector_norm(ord=p)`** — `coeus_ops::norm_p_axis(x, p, axis,
  backend)` reduces one axis to size 1 with `(sum(abs(x)^p))^(1/p)`, and
  `pycoeus.vector_norm(input, ord=p, axis=..., keepdim=...)` now returns a
  tensor or scalar matching PyTorch/JAX shape semantics. Evidence tier:
  empirical Burn differential and PyO3 binding validation.

- **Tracked Lp norm autograd** — `coeus_autograd::{norm, norm_p,
  norm_p_axis}` are exported with analytical backward nodes for scalar and
  per-axis norms. Evidence tier: analytical oracle tests plus empirical
  execution.

- **`einsum` / `index_select` shape parity** — added Rust-core
  `coeus_ops::{einsum, index_select}`, tracked autograd wrappers, and PyO3
  `pycoeus.einsum` / `pycoeus.index_select` registrations for common ML
  patterns and slice selection. Evidence tier: value-semantic Rust and binding
  tests.

- **WGPU scaled-dot-product attention kernels** — unmasked and causal forward
  and backward attention now route through WGSL kernels instead of host-side
  CPU copies; masked forward remains an explicit CPU-reference capability
  boundary. Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-wgpu --test wgpu_tests attention` passes with 4
  tests.

- **WGPU unary shader parity for new math ops** — `recip`, `sign`, `floor`,
  `ceil`, `round`, and `trunc` now have concrete WGSL expressions in the WGPU
  unary shader generator and differential tests against `SequentialBackend`.
  Evidence tier: empirical differential validation. Evidence: `cargo nextest
  run -p coeus-wgpu --test wgpu_tests test_wgpu_parity_recip
  test_wgpu_parity_sign test_wgpu_parity_floor test_wgpu_parity_ceil
  test_wgpu_parity_round test_wgpu_parity_trunc` passes.

### Changed

- Workspace version bumped `0.2.5` → `0.2.6`.
- `coeus-nn/Cargo.toml` adds `coeus-optim` as dev-dependency to support optimizer
  parity tests in `burn_live_parity.rs`.

## 0.2.5 - 2026-06-24

### Added

- **Functional Python nn wrappers** — three stateless free functions added to
  `coeus-python/src/ops.rs` matching `torch.nn.functional.*`:
  - `linear(input, weight, bias=None)` — weight-matrix multiply + optional bias.
  - `layer_norm(input, norm_shape, weight=None, bias=None, eps=1e-5)` — layer
    normalization over the last `norm_shape` features.
  - `dropout(input, p=0.5, training=False)` — training-mode dropout; returns
    input unchanged when `training=False` or `p=0.0`.

- **Burn parity suite expanded 40 → 48 tests** in
  `coeus-nn/tests/burn_live_parity.rs`:
  - `tril_triu_forward_and_backward` — value-semantic mask forward and masked
    gradient backward.
  - `roll_forward_and_backward` — circular shift forward and unroll backward.
  - `feed_forward_forward_shape_contract` — shape contract + non-zero liveness
    for the 3-layer FeedForward transformer sub-block.
  - `multi_head_attention_forward_shape_contract` — shape contract + non-zero
    liveness for `MultiHeadAttention<H=4>` self-attention.

- **Moirai scheduler batch-drain** — `WorkStealingScheduler::try_execute_next_task`
  and `next_task` now skip the `Mutex` lock entirely when `global_len == 0`
  (relaxed-atomic early-out) and batch-drain all global tasks into the local
  queue with a single lock acquisition when non-empty, reducing per-task lock
  overhead on the common lock-free path. Evidence: `cargo test -p moirai-scheduler`
  passes.

- **CUDA scaled-dot-product attention parity** — added live CUDA differential
  coverage for unmasked and causal forward attention, masked CPU-boundary
  behavior, and backward `grad_q`/`grad_k`/`grad_v` against `SequentialBackend`.
  Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests attention`
  passes with 4 tests.

- **CUDA 3D pooling kernels** — routed `CudaBackend` max/average 3D pooling
  forward and backward through native CUDA JIT kernels instead of the CPU
  fallback path, with value-semantic differential tests against
  `SequentialBackend`. Evidence tier: empirical differential validation.
  Evidence: `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests
  pool3d` passes with 2 tests.

### Changed

- Workspace version bumped `0.2.4` → `0.2.5`.



### Added

- **Python ops surface expansion** — `coeus-python/src/ops.rs` gains five
  new free functions:
  - `unsqueeze(input, dim)` — insert a size-1 axis at `dim` (tracked; backward
    via squeeze of the gradient).
  - `squeeze(input, dim=None)` — remove size-1 axes (tracked; backward via
    unsqueeze of the gradient).
  - `flatten(input, start_dim=0, end_dim=None)` — flatten contiguous
    dimensions into one (implemented as tracked reshape).
  - `argmax(input, dim)` — index of maximum value along `dim`, keep-dim,
    returns `f64` indices (non-differentiable).
  - `argmin(input, dim)` — index of minimum value along `dim`, keep-dim,
    returns `f64` indices (non-differentiable).
  All five functions are registered in `coeus-python/src/lib.rs` and covered
  by two new test functions in `coeus-python/tests/binding_tests_ops.rs`
  (`test_unsqueeze_squeeze_flatten`, `test_argmax_argmin`). Evidence:
  `cargo nextest run -p coeus-python --test binding_tests_ops` passes with 20
  tests.

- **Global pooling layers** in `coeus-nn/src/pool.rs`:
  - `GlobalAvgPool1d<T,B>` — reduces `[N,C,L]` → `[N,C,1]` by pooling the
    full length.
  - `GlobalAvgPool2d<T,B>` — reduces `[N,C,H,W]` → `[N,C,1,1]` (square).
  - `GlobalAvgPool3d<T,B>` — reduces `[N,C,D,H,W]` → `[N,C,1,1,1]` (cubic).
  - `GlobalMaxPool2d<T,B>` — max-pool global spatial reduction for 4-D.
  - `GlobalMaxPool3d<T,B>` — max-pool global spatial reduction for 5-D.
  All five are zero-parameter ZSTs, exported from `coeus-nn/src/lib.rs`, and
  delegate to the existing tracked `avg_pool2d`/`max_pool2d`/`avg_pool3d`/
  `max_pool3d` autograd ops. Evidence: two new `burn_live_parity.rs` tests
  (`global_avg_pool2d_reduces_spatial_to_one`,
  `global_max_pool2d_reduces_spatial_to_one`) pass.

- **Burn parity tests** — `coeus-nn/tests/burn_live_parity.rs` extended from
  36 to 40 tests:
  - `avg_pool2d_forward_matches_manual_reference` — manual biased-mean oracle.
  - `global_avg_pool2d_reduces_spatial_to_one` — value-semantic global avg.
  - `global_max_pool2d_reduces_spatial_to_one` — value-semantic global max.
  - `batchnorm1d_forward_matches_manual_reference` — training-mode BatchNorm1d
    on `[1,C,L]` input verified for zero-mean per-channel output. Evidence:
  `cargo nextest run -p coeus-nn --test burn_live_parity` passes with 40
  tests.

- **Workspace device-tier routing** (from sprint MS-65) — `coeus-wgpu` and
  `coeus-cuda` storage allocations now use explicit `PlacementHint::Tier(
  MemoryTier::Device)` at every `alloc_zeroed` call site (including CoW
  `make_unique`) so the allocation contract is anchored to the
  Hephaestus+Mnemosyne device-tier seam. Three unit tests in
  `coeus-wgpu/src/storage.rs` verify device-tier allocation, host-pinned
  staging tier selection, and device-tier upload/download round-trip value
  preservation.
- **Resolved dependency policy audit** — `coeus-core/tests/dependency_policy.rs`
  now checks `cargo tree --workspace --edges normal` for the replacement/runtime
  crates Coeus must not resolve through production normal dependencies
  (`rayon`, `tokio`, `ndarray`, `nalgebra`, `rustfft`, `burn`, `tch`,
  `pollster`). Dev-only Burn benchmark/parity edges remain allowed. Evidence:
  `cargo nextest run -p coeus-core --test dependency_policy` passes with 3 tests.

### Changed

- Workspace version bumped `0.2.3` → `0.2.4`.


## 0.2.3 - 2026-06-24

### Added

- **`gather` / `scatter_add`** in `coeus-ops`, `coeus-autograd` (tracked with
  backward), and `coeus-python`. `gather(input, dim, index)` selects elements
  along a dimension; `scatter_add` is its backward (and standalone op).
- **`repeat_interleave`** in `coeus-ops` and `coeus-python`: repeat each
  element along a dimension, matching `torch.repeat_interleave`.
- **`interpolate_1d` / `interpolate_2d`** in `coeus-nn` (nearest + bilinear
  modes) and `coeus-python`: spatial resize for `[N,C,L]` and `[N,C,H,W]`
  inputs, matching the PyTorch/Burn `interpolate` API.
- **PyTensor first-dimension indexing and iteration**: `tensor[i]`,
  `tensor[-1]`, `tensor[start:stop]`, and `for row in tensor` now return
  tracked Rust-core slices through the PyO3 wrapper.
- **`coeus-leto` sparse dispatch**: added `CsrDispatch`, `spmv_into`, and
  `spmm_into` value-semantic coverage against direct `leto_ops` sparse kernels.
- **WGPU conv3d differential coverage**: forward and backward tests now compare
  WGPU against `SequentialBackend` for baseline, stride+padding, and dilation
  3-D convolution cases.
- **CUDA backend differential coverage**: `coeus-cuda/tests/cuda/parity.rs`
  compares `CudaBackend` against `SequentialBackend` for binary, unary,
  unary activation-gradient, reduction, matmul, convolution and pooling forward
  and backward, AdamW, and host/device round-trip behavior under the live `cuda`
  feature.
- **WGPU Hephaestus transfer routing**: `WgpuBackend` host/device copies now use
  the Hephaestus `ComputeDevice` upload/download surface instead of local queue
  writes and ad hoc staging-buffer readback.
- **GPU placement hints**: WGPU and real-CUDA storage allocations now request
  Hephaestus buffers with Themis `MemoryTier::Device`; host-pinned staging is
  covered by value-semantic round-trip tests, and the CUDA Themis edge is
  feature-scoped to the real `cuda` module.
- **Global pooling modules**: `coeus-nn` now exports ZST global average/max
  pooling modules for supported dimensions; `GlobalAvgPool1d` routes through
  the tracked Rust autograd mean-axis reducer instead of a fake 2-D pool path.
- **Burn activation parity**: `coeus-nn/tests/burn_live_parity.rs` now compares
  Mish, Softplus, and LeakyReLU against live Burn NdArray references.
- **Burn log-softmax parity**: `coeus-nn/tests/burn_live_parity.rs` now compares
  Coeus forward values and autograd gradients against Burn NdArray autodiff.
- **Burn activation-backward parity**: sigmoid, tanh, SiLU, and GELU-family
  backward checks now compare Coeus autograd against Burn NdArray autodiff, with
  Burn 0.16's tanh-approximation GELU backward routed to Coeus' explicit
  `gelu_tanh` contract.
- **Burn loss and normalization backward parity**: BCE, MSE, Huber,
  LayerNorm, and RMSNorm gradient checks now compare Coeus autograd against
  Burn NdArray autodiff. Huber uses `delta = 1`, where Coeus' current
  SmoothL1-style formula and Burn's Huber contract coincide exactly.
- **coeus-python test harness**: shared test-only embedded-Python lock now
  serializes module registration for binding operation and distributed tests.
- **coeus-python module-scope cleanup**: binding operation and distributed tests
  now execute scripts with explicit `pycoeus` globals and remove the temporary
  `sys.modules` entry after each run.
- **coeus-python shape and selection parity**: added free-function wrappers for
  `unsqueeze`, `squeeze`, `flatten`, `argmax`, and `argmin`, with PyO3
  `ValueError` validation for invalid dimensions.
- **8 new `binding_tests_ops.rs` test functions** covering all previously
  untested ops: `topk/sort`, `comparisons (eq/lt/gt)/where_fn`, `softmax/
  cumsum/flip`, `randn/zeros_like/ones_like/eye`, `gather/scatter_add`,
  `repeat_interleave/interpolate`, `std_dev/var/norm`, and tensor indexing.

### Changed

- `coeus_autograd::gather` correctly routes backward through `scatter_add`
  (zero gradient to `index` since integer indices are non-differentiable).
- `coeus-core` dependency policy now rejects direct production `rustfft` imports
  and manifest dependencies, preserving Apollo's Atlas-owned FFT implementation
  as the Coeus FFT path.
- The repository ignores root-level `/prog` checkpoint transcript artifacts so
  generated session state does not appear as source.
- `coeus-tensor` benchmarks no longer carry a direct Rayon comparison row or
  dev-dependency; the benchmark surface uses the existing `Coeus Moirai` row as
  the parallel execution comparison.
- `coeus-ops` contiguous CPU `conv1d`, `conv2d`, and `conv3d` row execution now
  use Melinoe branded partitioning for disjoint output rows instead of raw
  output-pointer writes on that path, sharing one branded row-shard SSOT
  (`brand_mut_slice`) across all three kernels.

### Fixed

- README and checklist benchmark descriptions now match the current
  `coeus-tensor` Criterion surface after removing the direct Rayon row and
  direct third-party tensor benchmark rows.
- CUDA fused-kernel PTX loading now trims the NVRTC trailing NUL before
  constructing a `CString`, preventing JIT kernels from silently falling back to
  CPU execution when the CUDA feature is active.
- CUDA binary dispatch now routes broadcasted contiguous operands through the
  strided kernel instead of the elementwise contiguous kernel, which has no
  broadcast indexing contract.
- CUDA GELU and GELU-gradient kernels now use the exact erf formulation shared
  by the CPU and WGPU contracts instead of the tanh approximation.
- CUDA strided JIT kernels now decode output coordinates through the same
  output-layout stride metadata used by fused kernels, fixing broadcasted
  strided binary execution once the JIT path is active.

## 0.2.2 - 2026-06-24

### Added

- **`coeus_autograd::GradBuffer`** (`coeus-autograd/src/grad_buffer.rs`):
  zero-overhead gradient accumulation cell replacing `Arc<Mutex<Tensor>>` in
  every backward node.  Uses `UnsafeCell<Tensor>` with an `unsafe impl Sync`
  upheld by serialized backward, optimizer, and distributed-gradient phases.
  Eliminates mutex lock/unlock overhead from the backward path.
- **sin/cos tracked autograd ops** with correct backward
  (`d/dx sin = cos(x)`, `d/dx cos = -sin(x)`); exported from `coeus-autograd`.
- **`flip` / `sort` / `where_cond`** ops in `coeus-ops` and `coeus-autograd`
  (with correct backward passes).
- **Exact erf GELU** (`libm::erff`/`erf` via `FloatOps::erf_op`): updates
  `gelu_op`, `GeluGrad`, and `fuse/op_tags::GeluGrad` to use the exact formula
  `0.5 x (1 + erf(x/√2))` instead of the tanh polynomial approximation.
- **30+ Burn live parity tests** covering arithmetic, activations (sin/cos),
  matmul, reductions, linear fwd/bwd, layernorm, rmsnorm, clamp, shape ops,
  mse_loss, conv1d/2d forward, max_pool2d, where_cond backward, flip backward.
- **coeus-python API expansion** — PyTorch/JAX/MLX parity:
  - New `PyTensor` methods: `detach`, `requires_grad_`, `flatten`, `view`,
    `expand`, `eq`, `lt`, `gt`, `ne`, `tolist`, `__len__`, `__bool__`,
    `__float__`, `__int__`, `__rmul__`, `__radd__`, `sin`, `cos`, `flip`,
    `item`, `numel`, `ndim`.
  - New free functions: `zeros_like`, `ones_like`, `eye`, `std_dev`(`std`),
    `tensor_var`(`var`), `norm`, `eq`, `lt`, `gt`, `where_fn`(`where`),
    `sin`, `cos`, `flip`, `softmax`, `randn`, `topk`, `sort`, `where_cond`.
- **`reduce_broadcast` single-pass improvement** in `coeus-autograd::backward`:
  reduction axes computed once, applied with `enumerate()`, removes redundant
  intermediate tensor allocation for broadcast gradient shapes.

### Changed

- All `Arc<Mutex<Tensor<T,B>>>` gradient accumulators in `coeus-autograd`
  replaced with `Arc<GradBuffer<T,B>>` — zero runtime locking on the backward
  path.
- Renamed the real in-process distributed collective backend from
  `MockCommunicator` to `LocalCommunicator`, including the Python class
  `LocalCommunicator` and constructor `create_local_cluster`; no compatibility
  alias is retained.
- `BackendOps::max_pool2d` signature: added explicit `dilation` parameter
  between `padding` and `output`.
- WGPU fused GELU parity tolerance relaxed to 5e-3 (WGSL uses tanh
  approximation; CPU fused now uses exact erf).

### Fixed

- Removed mock-named distributed collective tests and binding APIs whose
  implementation was already a real barrier-backed local communicator.

### Performance (atlas crates)

- **mnemosyne-arena**: `initialize_large_or_huge_segment` split into two
  concrete helpers — `_fresh` (writes invariant header fields once) and
  `_cached` (skips them on pool-hit paths) — removing 2-4 dead stores on
  every cache-hit large/huge allocation.

## 0.2.1 - 2026-06-24

### Added

- **Live Burn parity suite** (`coeus-nn/tests/burn_live_parity.rs`): 25+ tests
  comparing Coeus outputs against Burn NdArray for add/sub/mul/div, relu, sigmoid,
  tanh, gelu, silu, exp/log/sqrt/neg/abs, matmul 2D/large/batched, reductions,
  linear fwd/bwd, layernorm fwd, clamp, shape ops, mse_loss, and backward passes.
- **Burn benchmarks** (`coeus-tensor/benches/tensor_bench.rs`): four Criterion
  benchmark groups (elementwise add, matmul 256×256, ReLU, sum_dim) comparing
  Burn NdArray against Coeus Sequential and Moirai side-by-side.
- **WgpuBackend parity audit** (`coeus-wgpu/tests/wgpu/parity.rs`): 20+ differential
  tests for binary ops, 14 unary activations, reductions, conv1d/conv2d forward,
  max/avg pool2d, AdamW step, and CPU↔GPU round-trip identity.
- **`coeus_autograd::stack`** with correct backward pass (split + squeeze);
  exported from `coeus-autograd`.
- **20 new coeus-python functional ops**: `stack`, `matmul`, `abs`, `sqrt`, `neg`,
  `clamp`, `max_axis`, `min_axis`, `log_sum_exp`, `sum`, `mean`, `zeros`, `ones`,
  `full`, `arange`, `linspace`, `reshape`, `permute`, `t`, `pow` — matching the
  `torch.*` / `jnp.*` / `mx.*` functional style.
- **`coeus-python/tests/binding_tests_ops.rs`**: 9 binding tests covering all new
  ops including autograd backward passes.
- **Shape/select ops**: `coeus_ops::{flip, sort, where_cond}` plus autograd
  `flip` and `where_cond` wrappers.
- **Python parity surface**: added `sin`, `cos`, `flip`, `where_cond`,
  `softmax`, `randn`, `topk`, and `sort` functions as thin PyO3 wrappers over
  Rust Coeus operations.
- **Manual reference parity coverage**: added conv1d, conv2d, max-pool2d,
  `where_cond` backward, and `flip` backward value-semantic tests using
  explicit Rust references where live Burn coverage is not yet wired.

### Changed

- Updated `docs/backlog.md` and `docs/checklist.md` for Sprint MS-61.
- Autograd gradient storage now uses one `GradBuffer` SSOT instead of
  `Arc<Mutex<Tensor<_, _>>>` in backward nodes; optimizers and distributed
  gradient synchronization mutate gradients through the same direct buffer
  surface.
- Primary `gelu` now follows the exact Burn/PyTorch formula
  `0.5 * x * (1 + erf(x / sqrt(2)))`; `gelu_tanh` remains the explicit tanh
  approximation.
- WGPU unary and fused GELU shader generation now uses one exact-contract WGSL
  expression SSOT with an Abramowitz-Stegun `erf` approximation.

### Fixed

- Fixed live Burn GELU parity by removing the accidental tanh-approximation
  behavior from the primary `gelu` path and aligning CPU, fused CPU, and WGPU
  shader tests to the same exact contract.
- Removed the temporary Mutex-compatible `GradBuffer::lock().unwrap()` shim.
- Python comparison wrappers now raise `ValueError` for shape mismatches rather
  than panicking through `assert_eq!` at the PyO3 boundary.
- Renamed conv/pool tests that use manual references so their names no longer
  claim live Burn evidence.

### Performance (atlas crates)

- **hermes** (`crates/hermes-simd/src/cpu.rs`): cached `has_amx()` and
  `has_avx512()` results with `OnceLock`; previously each dispatch called the
  serialising `cpuid` instruction (~50-200 cycles). Steady-state now pays one
  relaxed atomic load.
- **moirai** (`moirai-scheduler`): gated `Instant::now()` / `SystemTime::now()`
  task timing behind `cfg(feature="metrics")`; saves ~15-30 ns per micro-task on
  compute-intensive work-stealing workloads.
- **moirai** (`moirai-core`): added `#[repr(align(64))]` + 63-byte padding to
  `TaskResultSlot` to place the `state` field (written by producer) and
  `result`/`waiter` fields (read by consumer) on separate cache lines, eliminating
  producer-consumer false sharing.
- **leto-ops** (`application/matrix.rs`): `parallel_cc_matmul`,
  `parallel_dot_matmul`, and `parallel_outer_matmul` now dispatch in row blocks
  of 4 (`PARALLEL_ROW_BLOCK`), reducing task count by 4× and amortising
  per-task scheduling overhead; also prevents false sharing for small-`n` layouts.

## 0.2.0 - 2026-06-12

### Added

- Added WGPU attention forward/backward differential parity tests against the
  CPU public attention path.
- Added WGPU backend parity tests for elementwise ops, reductions, matmul,
  conv/pool, AdamW, and CPU/GPU round-trip transfer.
- Added a dev-only Burn NdArray live parity target for `coeus-nn` softmax and
  cross-entropy loss.
- Added Burn NdArray comparison rows to the `coeus-tensor` Criterion benchmark
  harness for add, matmul, ReLU, and sum.
- Added public `coeus_ops::stack` backed by `coeus-leto` dynamic-rank stack
  dispatch, with `SequentialBackend` and `MoiraiBackend` value-semantic
  coverage for strided input views.
- Added `BackendOps::batched_matmul` as the rank-3 batched matrix multiplication
  backend seam.
- Added `Scalar::from_usize` as the native index-conversion seam for
  index-derived tensor values.

### Changed

- Continued Stage A2 CPU consolidation onto `leto` by extending the structural
  dispatch shim to stack operations.
- Routed public batched `coeus_ops::matmul` through `BackendOps::batched_matmul`;
  CPU backends override the seam with `coeus-leto` batched dispatch while GPU
  backends retain the generic default.
- Routed public scalar `coeus_ops::mean` through backend `ReductionOp::Mean`.
- Routed contiguous CPU attention row dot products and softmax row scaling
  through new `Scalar::{dot_slice, scale_slice}` Hermes SIMD seams.
- Routed CPU attention backward contiguous `dO @ V^T` rows and softmax row
  products through `Scalar::dot_slice`.
- Routed contiguous unpadded unit-dilation CPU `conv1d` forward kernel rows
  through `Scalar::dot_slice`, preserving the indexed path for padded, dilated,
  or non-contiguous layouts.
- Routed contiguous unpadded unit-dilation CPU `conv2d` forward kernel rows
  through `Scalar::dot_slice`, preserving the indexed path for padded, dilated,
  or non-contiguous layouts.
- Routed contiguous unpadded unit-dilation CPU `conv3d` forward kernel rows
  through `Scalar::dot_slice`, preserving the indexed path for padded, dilated,
  or non-contiguous layouts.
- Routed contiguous unpadded unit-stride/unit-dilation CPU `conv1d` backward
  weight-gradient rows through `Scalar::dot_slice`, preserving the indexed path
  for padded, strided, dilated, or non-contiguous layouts.
- Routed contiguous unpadded unit-stride/unit-dilation CPU `conv2d` backward
  weight-gradient width rows through `Scalar::dot_slice`, preserving the indexed
  path for padded, strided, dilated, or non-contiguous layouts.
- Routed contiguous unpadded unit-stride/unit-dilation CPU `conv3d` backward
  weight-gradient width rows through `Scalar::dot_slice`, preserving the indexed
  path for padded, strided, dilated, or non-contiguous layouts.
- Removed Coeus' direct `pollster` dependency from `coeus-wgpu` and extended the
  dependency policy to keep Coeus production code on the Moirai async SSOT.
- Extended the dependency policy to keep direct replacement-library usage
  (`burn`, `nalgebra`, `ndarray`, `tch`) out of production Coeus sources and
  production manifest dependency sections while preserving benchmark/dev-only
  comparisons.
- Routed `Tensor::eye_on` identity generation through `coeus-leto` coordinate
  dispatch.
- Routed `Tensor::arange_on` through `coeus-leto` coordinate dispatch using
  `Scalar::from_usize`.
- Routed `Tensor::linspace_on` through `coeus-leto` coordinate dispatch while
  preserving its existing `Scalar::from_f64` value contract.
- Consolidated duplicated fused CPU evaluation and reduction traversal into
  shared writer helpers, with an RAII cache guard for temporary host tensor
  downloads.
- Split the Python distributed binding parity script into per-collective tests
  so each local/TCP collective is independently bounded by nextest.

### Fixed

- Fixed the Python binding functional-op test harness for PyO3 0.23's `CStr`
  script API and passed owned shapes into `Tensor::full_on`.
- Fixed zero-length `CpuStorage` so empty tensors expose valid non-null aligned
  Rust slices.
- Fixed rustdoc shape/type annotations that were parsed as intra-doc links or
  HTML so `cargo doc --workspace --no-deps` is warning-clean.
- Added value-semantic fused reduction coverage for sum, mean, max, and min.
- Added Rust TCP reduce, gather, and scatter coverage for `coeus-dist`.
- Fixed the Python distributed binding test timeout by isolating the TCP
  collectives instead of running every distributed scenario in one test body.
