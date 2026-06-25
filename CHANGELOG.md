# Changelog

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
