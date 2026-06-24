# Changelog

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
