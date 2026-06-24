# Changelog

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

### Changed

- Updated `docs/backlog.md` and `docs/checklist.md` for Sprint MS-61.

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
  so each mock/TCP collective is independently bounded by nextest.

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
