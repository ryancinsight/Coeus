# Global Progress Checklist: Coeus

## Active Epic: Burn Parity, GPU Audit & Python Surface Expansion

### Current Sprint: Sprint MS-64 (Python Tensor Parity / Shape Ops) [IN PROGRESS]
**Objective**: Extend coeus/coeus-python shape and indexing parity while keeping
Python as a thin PyO3 wrapper over Rust core operations.
**Target version**: 0.2.3.

> **Roadmap (docs/backlog.md MS-61)**: live Burn comparison starts replacing hardcoded
> oracle values; wgpu parity.rs verifies implemented GPU paths against the CPU reference;
> coeus-python gains 20+ new functional ops (stack, matmul, constructors, abs/sqrt/neg,
> clamp, max/min_axis, sum/mean, reshape, permute, t, pow, arange, linspace, etc.).

### Current Verification Note (2026-06-24)

- [x] [minor] Added `burn 0.16` as dev-dep to `coeus-nn` and `coeus-tensor`; production
  dependency policy test unaffected (burn forbidden in `[dependencies]`, allowed in
  `[dev-dependencies]`).
- [x] [patch] Added `coeus-nn/tests/burn_live_parity.rs` with live Burn NdArray
  reference checks for softmax and cross-entropy loss.
- [x] [minor] Added four Burn benchmark groups to `tensor_bench.rs`: elementwise add,
  matmul (256×256), ReLU (1024×1024), and sum_dim (1024×1024).  Each group shows Burn
  NdArray, Coeus Sequential, and Coeus Moirai side-by-side under Criterion.
- [x] [minor] Created `coeus-wgpu/tests/wgpu/parity.rs` with 20+ differential tests:
  binary ops, 14 unary activations (macro), reductions, matmul 2D + batched,
  conv1d/conv2d forward, max_pool2d/avg_pool2d, adamw step, round-trip identity.
- [x] [patch] Added `coeus_autograd::stack` (`shape/stack.rs`) with correct backward
  via split+squeeze; exported from `coeus-autograd/src/lib.rs`.
- [x] [minor] Expanded `coeus-python/src/ops.rs` with 20 new free functions and added
  `coeus-python/tests/binding_tests_ops.rs` with 9 test functions including backward.
- [x] [patch] `cargo check --workspace` passes: 0 errors.
- [x] [patch] `cargo clippy --workspace --all-targets -- -D warnings` passes: 0
  errors, 0 warnings.
- [x] [patch] Promoted primary `gelu` to the exact Burn/PyTorch contract
  `0.5 * x * (1 + erf(x / sqrt(2)))` through the scalar SSOT; retained
  `gelu_tanh` as the explicit tanh approximation.
- [x] [patch] Added WGSL exact-contract GELU/GELU-gradient expressions using an
  Abramowitz-Stegun `erf` approximation for WGPU unary and fused shader paths.
- [x] [minor] Expanded live Burn parity to 25 value-semantic tests, including
  exact GELU, SiLU, sin/cos forward/backward, matmul/linear backward, layernorm,
  RMSNorm, clamp, stack/cat/reshape/transpose, flip, sort, and where-cond.
- [x] [patch] Extended live Burn activation parity to Mish, Softplus, and
  LeakyReLU in `coeus-nn/tests/burn_live_parity.rs`, using the derived
  epsilon helper for value-semantic comparisons against Burn NdArray.
  Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-nn --test burn_live_parity` passes with 31 tests.
- [x] [minor] Added `coeus_ops::{flip, sort, where_cond}`, autograd
  `flip`/`where_cond`, and Python wrappers for `sin`, `cos`, `flip`,
  `where_cond`, `softmax`, `randn`, `topk`, and `sort`.
- [x] [patch] Replaced autograd gradient `Arc<Mutex<Tensor<_, _>>>` storage
  with the `GradBuffer` UnsafeCell SSOT and removed the temporary
  Mutex-shaped compatibility shim; optimizers, distributed gradient sync, and
  attention tests now read/write through `GradBuffer` directly.
- [x] [patch] Corrected conv/pool parity test names whose oracles are manual
  references rather than live Burn tensors, preserving the evidence tier stated
  by the test names.
- [x] [patch] Python comparison wrappers now return `ValueError` on shape
  mismatch instead of panicking at the PyO3 boundary.
- [x] [patch] Renamed the real barrier-backed distributed test communicator
  from `MockCommunicator` to `LocalCommunicator`, including the PyO3 class and
  `create_local_cluster` constructor, with no compatibility alias.
- [x] [minor] Added Rust-core `gather`, `scatter_add`, `repeat_interleave`,
  and `interpolate_1d`/`interpolate_2d` surfaces with coeus-python wrappers.
- [x] [patch] Added PyTensor first-dimension indexing and iteration
  (`tensor[i]`, `tensor[-1]`, `tensor[start:stop]`, `for row in tensor`) using
  tracked Rust-core slice/squeeze operations.
- [x] [patch] Added `coeus-leto::CsrDispatch` sparse SpMV/SpMM dispatch coverage
  against direct `leto_ops` sparse kernels.
- [x] [patch] Routed contiguous CPU `conv1d`, `conv2d`, and `conv3d` row
  execution through one shared Melinoe branded row-partition SSOT
  (`brand_mut_slice` in `conv/mod.rs`), preserving the existing
  value-semantic conv parity tests as the current evidence tier.
- [x] [minor] Extended WGPU conv3d forward/backward differential parity beyond
  the baseline case: stride+padding and dilation cases now compare WGPU results
  against `SequentialBackend` values for output, input gradient, weight
  gradient, and bias gradient. Evidence: `cargo nextest run -p coeus-wgpu
  --test wgpu_tests conv3d` passes with 4 tests.
- [x] [minor] Added CUDA feature parity coverage for binary, unary, reduction,
  matmul, convolution, pooling, AdamW, and host/device round-trip behavior
  against `SequentialBackend`; fixed NVRTC PTX trailing-NUL trimming so fused
  CUDA kernels load through `CString` instead of silently falling back, routed
  broadcasted contiguous operands through strided binary kernels, corrected CUDA
  GELU/GELU-gradient to the exact erf contract, and aligned strided JIT
  coordinate decoding with fused-kernel layout metadata.
  Evidence tier: empirical differential validation. Evidence:
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests` passes
  with 42 tests.
- [x] [patch] Extended CUDA live parity coverage to unary activation-gradient
  kernels (`ReluGrad`, `SigmoidGrad`, `TanhGrad`, `GeluGrad`, `SiluGrad`,
  `MishGrad`) against the CPU unary reference, including exact-erf `GeluGrad`
  inputs where the tanh approximation would diverge. Evidence tier: empirical
  differential validation. Evidence:
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests` passes
  with 48 tests.
- [x] [patch] Extended CUDA live parity coverage to backward kernels for
  `conv2d`, `max_pool2d`, and `avg_pool2d`, comparing device gradients against
  `SequentialBackend` references for gradient input, weight, and bias where
  applicable. Evidence tier: empirical differential validation.
  Evidence: `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests`
  passes with 51 tests.
- [x] [patch] Consolidated the `coeus-python` embedded-Python test lock into
  `tests/common/mod.rs` and routed binding ops/distributed tests through that
  test-only SSOT. Evidence: `cargo nextest run -p coeus-python --test
  binding_tests_dist --test binding_tests_ops` passes with 26 tests.
- [x] [patch] Removed the direct Rayon comparison row and dev-dependency from
  `coeus-tensor` benchmarks; `Coeus Moirai` remains the parallel execution row.
  Evidence tier: compile-time dependency audit plus benchmark build. Evidence:
  `cargo check -p coeus-tensor --benches` and
  `cargo nextest run -p coeus-core --test dependency_policy` pass.
- [x] [patch] Reconciled README and checklist benchmark descriptions with the
  Rayon-free harness surface: Coeus Sequential, Coeus Moirai, direct Leto,
  Coeus-Leto dispatch, and dev-only Burn NdArray oracle rows. Evidence tier:
  documentation/dependency-surface consistency.
- [x] [patch] Extended `coeus-core/tests/dependency_policy.rs` to reject direct
  production `rustfft` imports and manifest dependencies, keeping Apollo FFT as
  the FFT SSOT for Coeus. Evidence tier: compile-time dependency audit.
  Evidence: `cargo nextest run -p coeus-core --test dependency_policy` passes
  and `rg -n "rustfft|apollo" -g "Cargo.toml" -g "*.rs" -g "*.md"` shows no
  production Coeus `rustfft` use.
- [x] [patch] Added a root-scoped `/prog` ignore entry for transient checkpoint
  transcript artifacts so generated session state is not staged as project
  source. Evidence tier: repository hygiene.
- [x] [patch] Verification: `cargo fmt --check`,
  `cargo check -p coeus-tensor --benches`,
  `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo clippy -p coeus-nn --test burn_live_parity -- -D warnings`,
  `cargo nextest run -p coeus-nn --test burn_live_parity` (31 passed),
  `cargo nextest run -p coeus-core --test dependency_policy`, and
  `cargo nextest run -p coeus-cuda --features cuda --test cuda_tests`
  (51 passed), `cargo nextest run --workspace` (421 passed),
  `cargo test --doc --workspace`, and `cargo doc --workspace --no-deps` pass on
  2026-06-24.

---

## Previous Sprint: Sprint MS-60+ (Atlas burn-replacement & GPU roadmap) [COMPLETE]
**Objective**: Route CPU `BackendOps` through `coeus-leto`; Hermes SIMD integration;
GPU backends over Hephaestus; dependency policy hardening.
**Target version**: 0.2.0.

> **Roadmap (docs/backlog.md MS-60+)**: the Atlas burn-replacement program now stages
> (A2) routing the CPU backend's `BackendOps` through `coeus-leto` and deleting the
> duplicated CPU traversal — keeping `Tensor<T,B>` and the `ComputeBackend` seam; and
> (D) the GPU program: ADR to migrate `coeus-cuda` from cutile to **cuda-oxide**, finish
> wgpu op parity, consume mnemosyne device pools / melinoe device-buffer ownership.
> burn is eliminated end-to-end in Stage E.

### Verification Note (2026-06-12)

- [x] [patch] Added committed nextest timeout config at `.config/nextest.toml`.
- [x] [patch] Synced README verification commands to `cargo nextest run`,
  doctests, and clippy with `-D warnings`.
- [x] [patch] `coeus-cuda` now defaults to a CPU-backed no-CUDA provider so the
  full workspace can check on hosts without `CUDA_TOOLKIT_PATH`; real cutile
  CUDA integration is retained behind the explicit `cuda` feature.
- [x] [patch] The default no-CUDA `CudaBackend` implements the full
  `BackendOps` surface by delegating to the existing CPU fallback path, with
  value-semantic coverage in `coeus-cuda/tests/no_cuda_fallback.rs`.
- [x] [patch] Replaced high-arity `coeus-wgpu` attention and convolution helper
  calls with typed request structs; verified by clippy and wgpu nextest.
- [x] [patch] `cargo clippy --workspace --exclude coeus-cuda --all-targets
  -- -D warnings` passes after the `coeus-wgpu` request-struct refactor.
- [x] [patch] `cargo fmt --check` passes after workspace formatting.
- [x] [patch] `cargo check --workspace` passes without excluding `coeus-cuda`.
- [x] [patch] `cargo clippy --workspace --all-targets -- -D warnings` passes
  without excluding `coeus-cuda`.
- [x] [patch] `cargo nextest run --workspace` passes: 255 tests passed, 0
  skipped. CUDA integration tests are feature-gated under `cuda` because they
  require `CUDA_TOOLKIT_PATH` and a working CUDA driver.
- [x] [patch] `cargo test --doc --workspace` passes; four doctests are
  intentionally ignored.
- [x] [patch] Added `coeus-core/tests/dependency_policy.rs` to enforce the
  Moirai parallel/async SSOT: production sources and production manifest
  dependency sections may not import or depend on `rayon` or `tokio`. Evidence:
  `cargo nextest run -p coeus-core --test dependency_policy` passes; normal
  dependency tree checks show no production `rayon` edge and no resolved
  `tokio` package.
- [x] [patch] Removed Coeus' direct `pollster` dependency from `coeus-wgpu` and
  extended `coeus-core/tests/dependency_policy.rs` to reject Coeus production
  `pollster` imports/dependencies. Evidence:
  `cargo nextest run -p coeus-core --test dependency_policy` and
  `cargo tree -p coeus-wgpu --edges normal -i pollster` pass; the remaining
  resolved `pollster` edge is isolated inside
  `hephaestus-wgpu`.
- [x] [patch] Extended the dependency policy to reject direct production imports
  and direct production manifest dependencies on replacement libraries (`burn`,
  `nalgebra`, `ndarray`, `tch`) while preserving benchmark/dev-only comparisons.
  Evidence: `cargo nextest run -p coeus-core --test dependency_policy` passes.
- [x] [patch] Expanded `coeus-leto` contract coverage for the CPU consolidation
  seam: binary dispatch covers `Sub`/`Mul`/`Div`, unary dispatch covers
  `Relu`/`Abs`/`Neg`, and keep-dim reductions cover `Sum`/`Max`/`Min`. Evidence:
  `cargo nextest run -p coeus-leto` passes; the current contract suite contains
  14 tests.
- [x] [patch] Added `coeus-ops/tests/unary_leto_diff.rs` to prove
  `SequentialBackend` and `MoiraiBackend` unary `BackendOps` dispatch matches
  direct scalar `CpuUnaryDispatch::eval_unary` for the full `CpuUnaryOp` surface.
  Evidence: `cargo test -p coeus-ops --test unary_leto_diff` passes.
- [x] [patch] Added `coeus-ops/tests/matmul_leto_diff.rs` to prove
  `SequentialBackend` and `MoiraiBackend` `BackendOps::matmul` dispatch matches
  an independent row-major triple-loop reference for contiguous and strided
  transposed input layouts. Evidence: `cargo test -p coeus-ops --test
  matmul_leto_diff` passes.
- [x] [patch] Added `coeus-ops/tests/batched_matmul_leto_diff.rs` to prove the
  public `coeus_ops::matmul` batching layer matches an independent reference on
  `SequentialBackend` and `MoiraiBackend` for equal batch counts and RHS 2-D
  broadcast. Evidence: `cargo test -p coeus-ops --test
  batched_matmul_leto_diff` passes.
- [x] [patch] Routed `coeus_ops::cumsum` and `suffix_sum` through
  `coeus-leto` scan dispatch and added value-semantic coverage in
  `coeus-leto/tests/contract.rs` plus `coeus-ops/tests/scan_leto_diff.rs`.
  Evidence: focused scan tests pass.
- [x] [patch] Added public CPU reduction differential coverage for
  `sum`/`mean`/`sum_axis`/`mean_axis`/`max_axis`/`min_axis` on
  `SequentialBackend` and `MoiraiBackend`, including transposed input views.
  Evidence: `cargo test -p coeus-ops --test public_reduction_leto_diff`
  passes.
- [x] [patch] Routed public scalar `mean` through backend
  `ReductionOp::Mean`, so CPU scalar mean now uses the `coeus-leto` mean
  reducer instead of local `sum / count` division. Evidence: `cargo test -p
  coeus-ops --test public_reduction_leto_diff` passes.
- [x] [patch] Promoted mean to `ReductionOp::Mean` and routed public
  `mean_axis` through backend reduction dispatch. CPU dispatch uses Leto
  `MeanAxis`; WGPU/CUDA generated reducers and CPU fused reductions cover the
  same variant. Evidence: focused CPU, Leto, WGPU fused, and CUDA fallback tests
  pass.
- [x] [patch] Routed public `argmax` and `argmin` through `coeus-leto`
  keep-dim arg-reduction dispatch for CPU-addressable tensors and added
  transposed-view coverage for `SequentialBackend` and `MoiraiBackend`.
  Evidence: `cargo test -p coeus-leto
  arg_reduction_dispatch_covers_keepdim_axis_ops` and `cargo test -p coeus-ops
  --test arg_reduction_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::pad` through `coeus-leto` structural
  pad dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo test -p coeus-leto pad_dispatch_covers_strided_input_view` and
  `cargo test -p coeus-ops --test pad_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::cat` through `coeus-leto` structural
  concat dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo test -p coeus-leto concat_dispatch_covers_strided_input_views` and
  `cargo test -p coeus-ops --test concat_leto_diff` pass.
- [x] [patch] Routed public `coeus_ops::split` through `coeus-leto` structural
  split dispatch for CPU-addressable tensors and added strided/transposed-view
  coverage for `SequentialBackend` and `MoiraiBackend`. Evidence:
  `cargo test -p coeus-leto split_dispatch_covers_strided_input_view` and
  `cargo test -p coeus-ops --test split_leto_diff` pass.
- [x] [patch] Routed `coeus_nn::init::{uniform_with_seed, normal_with_seed}`
  through `coeus-leto` seeded random dispatch, deleting the local Xorshift
  initializer implementation. Constructor-only `RandomScalar` bounds preserve
  forward/module surfaces for existing `Float` APIs. Evidence: `cargo test -p
  coeus-leto random_dispatch_matches_leto_seeded_constructors` and `cargo test
  -p coeus-nn --test init_leto_diff` pass.
- [x] [patch] Routed `Tensor::to_contiguous_on` for CPU-addressable storage
  through `coeus-leto` view materialization, deleting the local strided
  materialization loop from that path. Evidence: `cargo test -p coeus-leto
  contiguous_dispatch_matches_leto_view_materialization` and `cargo test -p
  coeus-tensor --test contiguous_leto_diff` pass.
- [x] [patch] Routed `Tensor::{reshape, permute}` plus `t`/`t_nd` through
  `coeus-leto` layout validation, preserving zero-copy storage sharing while
  deleting the local reshape/permute metadata duplication from the public tensor
  path. Evidence: `cargo test -p coeus-leto layout_dispatch` and `cargo test -p
  coeus-tensor --test shape_view_leto_diff` pass.
- [x] [patch] Routed non-contiguous cross-backend `Tensor::to_backend_on`
  materialization through `coeus-leto`, deleting the remaining local strided
  transfer loops from the public tensor transfer path. Evidence: `cargo test -p
  coeus-tensor --test backend_transfer_leto_diff` passes.
- [x] [patch] Routed `Tensor::from_fn_on` coordinate generation through
  `coeus-leto`, deleting the local row-major dynamic-index generation loop from
  the public tensor constructor path. Evidence: `cargo test -p coeus-leto
  shape_function_dispatch_matches_leto_coordinate_order` and `cargo test -p
  coeus-tensor --test from_fn_leto_diff` pass.
- [x] [patch] Routed `Tensor::eye_on` identity value generation through
  `coeus-leto`, deleting the local diagonal mutation loop from the public tensor
  constructor path. Fixed zero-length `CpuStorage` to expose non-null aligned
  zero-length slices for empty tensors. Evidence: `cargo test -p coeus-core
  --test cow_storage_tests` and `cargo test -p coeus-tensor --test
  identity_leto_diff` pass.
- [x] [minor] Added `Scalar::from_usize` as the native index-conversion seam
  and routed `Tensor::arange_on` through `coeus-leto`, deleting the local
  mutation loop and the constructor's f64 index conversion. Evidence: `cargo
  test -p coeus-core --test scalar_index_conversion` and `cargo test -p
  coeus-tensor --test arange_leto_diff` pass.
- [x] [patch] Routed `Tensor::linspace_on` coordinate traversal through
  `coeus-leto`, deleting the local mutable fill loop while preserving the
  existing `Scalar::from_f64` value contract. Evidence: `cargo test -p
  coeus-tensor --test linspace_leto_diff` passes.
- [x] [patch] Routed tensor broadcast shape and zero-copy broadcast layout
  validation through `coeus-leto`, deleting local dynamic broadcast metadata
  construction from `Tensor::broadcast` while preserving scalar rank-0
  broadcasts. Evidence: `cargo test -p coeus-leto
  broadcast_layout_dispatch_matches_leto_validation` and `cargo test -p
  coeus-tensor --test broadcast_leto_diff` pass.
- [x] [minor] Added public `coeus_ops::stack` through dynamic-rank
  `coeus-leto` stack dispatch, covering equal-shaped strided input views on
  `SequentialBackend` and `MoiraiBackend`. Evidence: `cargo test -p coeus-leto
  stack_dispatch_covers_strided_input_views` and `cargo test -p coeus-ops
  --test stack_leto_diff` pass.
- [x] [minor] Added `BackendOps::batched_matmul` as the batched matmul seam,
  routed public batched `coeus_ops::matmul` through it, and overrode the CPU
  `SequentialBackend`/`MoiraiBackend` path with `coeus-leto` rank-3 batched
  dispatch. Evidence: `cargo test -p coeus-leto
  batched_matmul_dispatch_covers_rhs_batch_broadcast`, `cargo test -p coeus-ops
  --test batched_matmul_leto_diff`, and `cargo test -p coeus-wgpu
  wgpu::transfers_and_matmul::test_wgpu_backend_ops_unified` pass.
- [x] [patch] Added `Scalar::{dot_slice, scale_slice}` Hermes SIMD seams and
  routed CPU forward attention contiguous Q/K row dot products plus softmax row
  scaling through them. Evidence: `cargo test -p coeus-core --test
  scalar_dot_scale` and `cargo test -p coeus-nn --test nn_attention_tests`
  pass.
- [x] [patch] Routed CPU attention backward contiguous `dO @ V^T` rows and
  softmax row products through `Scalar::dot_slice`. Evidence: `cargo test -p
  coeus-ops --test attention_backward_hermes_diff` passes.
- [x] [patch] Routed contiguous unpadded unit-dilation CPU `conv1d` forward
  kernel rows through `Scalar::dot_slice`, preserving the indexed path for
  padded, dilated, or non-contiguous layouts. Evidence: `cargo test -p
  coeus-ops --test conv1d_hermes_diff` passes.
- [x] [patch] Routed contiguous unpadded unit-dilation CPU `conv2d` forward
  kernel rows through `Scalar::dot_slice`, preserving the indexed path for
  padded, dilated, or non-contiguous layouts. Evidence: `cargo test -p
  coeus-ops --test conv2d_hermes_diff` passes.
- [x] [patch] Routed contiguous unpadded unit-dilation CPU `conv3d` forward
  kernel rows through `Scalar::dot_slice`, preserving the indexed path for
  padded, dilated, or non-contiguous layouts. Evidence: `cargo test -p
  coeus-ops --test conv3d_hermes_diff` passes.
- [x] [patch] Routed contiguous unpadded unit-stride/unit-dilation CPU `conv1d`
  backward weight-gradient rows through `Scalar::dot_slice`, preserving the
  indexed path for padded, strided, dilated, or non-contiguous layouts.
  Evidence: `cargo test -p coeus-ops --test conv1d_backward_hermes_diff`
  passes.
- [x] [patch] Routed contiguous unpadded unit-stride/unit-dilation CPU `conv2d`
  backward weight-gradient width rows through `Scalar::dot_slice`, preserving
  the indexed path for padded, strided, dilated, or non-contiguous layouts.
  Evidence: `cargo test -p coeus-ops --test conv2d_backward_hermes_diff`
  passes.
- [x] [patch] Routed contiguous unpadded unit-stride/unit-dilation CPU `conv3d`
  backward weight-gradient width rows through `Scalar::dot_slice`, preserving
  the indexed path for padded, strided, dilated, or non-contiguous layouts.
  Evidence: `cargo test -p coeus-ops --test conv3d_backward_hermes_diff`
  passes.
- [x] [patch] Fixed rustdoc shape/type annotations that were parsed as links or
  HTML, making workspace docs warning-clean. Evidence: `cargo doc --workspace
  --no-deps` passes.
- [x] [patch] Current full gate after CPU `conv3d` backward Hermes dot routing:
  `cargo fmt --check`,
  `git diff --check`, `cargo check --workspace`, `cargo clippy --workspace
  --all-targets -- -D warnings`, `cargo nextest run --workspace` (307 passed,
  0 skipped), `cargo test --doc --workspace`, and `cargo doc --workspace
  --no-deps` pass.
- [x] [minor] Added Criterion baselines in `coeus-tensor/benches/tensor_bench.rs`
  for direct Leto and Coeus-Leto dispatch alongside Coeus Sequential, Coeus
  Moirai, and later dev-only Burn NdArray oracle rows.
- [x] [patch] Consolidated duplicated fused CPU value/reduction traversal in
  `coeus-ops::fuse` behind shared writer helpers and replaced manual temporary
  host-cache cleanup with an RAII guard. Added value-semantic coverage for fused
  sum/mean/max/min reductions. Evidence: `cargo clippy -p coeus-ops
  --all-targets -- -D warnings` and `cargo nextest run -p coeus-tensor --test
  fused_ops_tests` pass.
- [x] [patch] Fixed the Python distributed binding timeout by splitting the
  monolithic local/TCP collective script into independently timed value-semantic
  tests, and added missing Rust TCP reduce/gather/scatter coverage. Evidence:
  `cargo nextest run -p coeus-python --test binding_tests_dist` passes in
  0.620s; `cargo nextest run -p coeus-dist` passes with 16 tests.
- [x] [patch] Added WGPU scaled-dot-product attention forward/backward
  differential coverage against the public CPU attention path, including causal
  masking and Q/K/V gradients. Evidence: `cargo nextest run -p coeus-wgpu
  --test wgpu_tests attention` passes.
- [x] [patch] Reconciled the WGPU parity test module with the current
  `BackendOps` pooling, convolution, and AdamW signatures. Evidence:
  `cargo nextest run -p coeus-wgpu --test wgpu_tests parity` passes with 33
  tests.
- [x] [patch] Completed the dev-only Burn live parity target for `coeus-nn`
  softmax and cross-entropy loss. Burn remains outside production dependency
  sections and is used only as a reference oracle. Evidence: `cargo nextest run
  -p coeus-nn --test burn_live_parity` passes.
- [x] [patch] Added Burn NdArray comparison rows to the `coeus-tensor`
  Criterion benchmark harness for add, matmul, ReLU, and sum. Evidence:
  `cargo clippy --workspace --all-targets -- -D warnings` passes after switching
  the ReLU benchmark to Burn's public activation API.
- [x] [patch] Fixed the Python binding functional-op test harness for PyO3
  0.23's `CStr` script API and passed owned shapes into `Tensor::full_on`.
  Evidence: `cargo clippy --workspace --all-targets -- -D warnings` and
  `cargo nextest run --workspace` pass.
- [x] [patch] Added `[profile.bench]` thin LTO with one codegen unit so
  cross-crate generic kernels are benchmarked after production-grade
  monomorphization. Evidence tier: empirical Criterion measurement.
- [x] [minor] Ran a short historical empirical benchmark pass:
  `cargo bench -p coeus-tensor --bench tensor_bench -- --warm-up-time 1
  --measurement-time 2 --sample-size 10`. Evidence tier: empirical Criterion
  measurement. The current harness no longer carries direct third-party tensor
  or Rayon rows; it retains Coeus Sequential/Moirai, direct Leto,
  Coeus-Leto dispatch, and dev-only Burn NdArray oracle rows. Focused
  post-profile 256x256 matmul measurement: Coeus Sequential 1.0006 ms, Coeus
  Moirai 1.1146 ms, direct Leto 1.1012 ms, Coeus-Leto dispatch 1.0905 ms.
  Rejected upstream Hermes tiled-GEMM route: Leto 256x256 f64 regressed to
  3.6848 ms and Coeus f32 direct Leto regressed to 8.7577 ms; source change was
  removed. Dense matmul remains a measured optimization target against the
  dev-only Burn oracle.

---

### Workspace Crate Status Matrix

| Crate Name | Path | Primary Responsibilities | Compilation Status | Notes / Blockers |
| :--- | :--- | :--- | :--- | :--- |
| **coeus-core** | [coeus-core](file:///d:/coeus/coeus-core) | Scalar types, layouts, storage traits, backend traits, CPU backends | ✅ Compiles | Clean compilation |
| **coeus-tensor** | [coeus-tensor](file:///d:/coeus/coeus-tensor) | N-dimensional strided tensor representation (`Tensor<T, B, S>`) | ✅ Compiles | Clean compilation |
| **coeus-ops** | [coeus-ops](file:///d:/coeus/coeus-ops) | Element-wise math, matrix operations, reductions | ✅ Compiles | Zero-copy layout traversal and thread-safe |
| **coeus-autograd** | [coeus-autograd](file:///d:/coeus/coeus-autograd) | Tape-based automatic differentiation (`Var`, `Tape`) | ✅ Compiles | Clean compilation |
| **coeus-optim** | [coeus-optim](file:///d:/coeus/coeus-optim) | Optimizers (SGD, Adam, RMSProp) | ✅ Compiles | Borrow checker conflicts resolved |
| **coeus-nn** | [coeus-nn](file:///d:/coeus/coeus-nn) | Neural network modules (Linear, activations, losses, normalization) | ✅ Compiles | Clean compilation |
| **coeus-sparse** | [coeus-sparse](file:///d:/coeus/coeus-sparse) | Sparse format storage primitives (COO, CSR) | ✅ Compiles | Clean compilation |
| **coeus-python** | [coeus-python](file:///d:/coeus/coeus-python) | Thin PyO3 bindings | ✅ Compiles | Clean compilation |

---

## Action Items Checklist

### Phase 1: Core CPU Workspace Stabilization [COMPLETE]
Resolve the remaining compiler and thread-safety blockers in `coeus-ops` and `coeus-optim`.

- [x] **coeus-optim Fixes**:
  - [x] Resolve `E0502` borrow conflicts in SGD step loop by caching `.numel()` before borrowing `param.tensor` mutably.
  - [x] Resolve `E0502` borrow conflicts in Adam step loop.
  - [x] Resolve `E0502` borrow conflicts in RMSProp step loop.
- [x] **coeus-ops Thread-Safety Fixes**:
  - [x] Implement `SendPtr<T>` and `SendPtrMut<T>` wrapper types in `coeus-ops` implementing `Send + Sync` to wrap raw pointers (`*const T` / `*mut T`).
  - [x] Refactor `coeus-ops/src/matmul/kernel.rs` to pass `SendPtr`/`SendPtrMut` into the `parallel_for` closure.
  - [x] Refactor `coeus-ops/src/reduction/sum.rs` to pass `SendPtr`/`SendPtrMut` into the `parallel_for` closure.
- [x] **Workspace Verification**:
  - [x] Run `cargo check --workspace` to verify zero compilation errors.
  - [x] Run `cargo test --workspace` to verify that all existing CPU tests pass.

### Phase 2: Parity Testing & Performance Benchmarks [COMPLETE]
Establish numerical validation, autograd equivalence, and performance measurements against baseline libraries.

- [x] **Numerical Parity Validation**:
  - [x] Implement `coeus-tensor/tests/parity_tests.rs` comparing Coeus tensor
    operations against self-contained row-major references.
  - [x] Verify exact parity across various strides, shapes, and layouts.
- [x] **Autograd Parity & Design Equivalence**:
  - [x] Implement autograd validation tests and verify gradient correctness.
- [x] **Performance Benchmarks**:
  - [x] Configure `criterion` benchmarks in
    `coeus-tensor/benches/tensor_bench.rs` comparing Sequential/Moirai backends
    against direct Leto, Coeus-Leto dispatch, and dev-only Burn NdArray oracle
    rows.

### Phase 3: GPU Integration Abstractions (Sprint MS-55 Phase 1) [COMPLETE]
Introduce generic, zero-cost associated-type abstractions to support device-specific execution.

- [x] **Associated-Type Backend Overhaul**:
  - [x] Define the `ComputeBackend` trait in `coeus-core::backend` to replace/extend `Backend`.
  - [x] Define associated types:
    - `type DeviceBuffer<T>` (device memory allocation type representing device-allocated storage).
    - `type KernelDescriptor` (information needed to run kernels on the backend).
    - `type DispatchFuture<T>` (async execution handle for non-blocking queue operations).
- [x] **Unified Storage Design**:
  - [x] Update `Storage<T>` and `StorageMut<T>` traits to support non-CPU-accessible buffers (preventing direct CPU slicing assumptions).
- [x] **Device Transfer API**:
  - [x] Implement `Tensor::to_backend::<NewB>(&self, backend: &NewB)` on `Tensor` to handle copying between host/device memory boundaries (CPU-to-GPU, GPU-to-CPU, GPU-to-GPU).

### Phase 4: WebGPU Backend Crate (coeus-wgpu) [COMPLETE]
Implement a cross-platform GPU backend powered by `wgpu`.

- [x] **Crate Structure**:
  - [x] Create the new `coeus-wgpu` crate and add it to the workspace members.
  - [x] Add `wgpu` and `bytemuck` dependencies in `coeus-wgpu/Cargo.toml`.
- [x] **Storage & Memory Management**:
  - [x] Implement `WgpuStorage<T>` wrapping a `wgpu::Buffer` handle with unified allocation and staging logic.
- [x] **Compute Pipelines**:
  - [x] Write WGSL (WebGPU Shading Language) compute shaders for element-wise unary operations, binary operations, matrix multiplication, sum reductions, and Conv1D/Conv2D forward and backward passes.
  - [x] Implement compilation cache for `wgpu::ComputePipeline` instances.
- [x] **Backend Trait implementation**:
  - [x] Implement `ComputeBackend` for `WgpuBackend` mapping operations to GPU shader dispatch.
  - [x] Implement `BackendOps` for `WgpuBackend` to route math operations.

### Phase 5: CUDA Backend Crate (coeus-cuda) [COMPLETE]
Implement an optimized native NVIDIA GPU backend dynamically loading the CUDA driver.

- [x] **Crate Structure**:
  - [x] Create the new `coeus-cuda` crate in the workspace.
  - [x] Add dynamic CUDA driver bindings (`cuInit`, `cuMemcpy`, etc.) using `libloading` for robust cross-host compilation.
- [x] **Storage & Context Management**:
  - [x] Implement `CudaStorage<T>` wrapping CUDA `CUdeviceptr` raw handles.
  - [x] Integrate thread-safe CUDA context management and memory allocation.
- [x] **Kernel Management**:
  - [x] Implement embedded PTX source containing element-wise, tiled matrix multiplication, sum reduction, Conv1D/Conv2D forward and backward passes, and strided/broadcasted element-wise operations.
  - [x] Implement direct driver dispatch launchers and remove CPU staging fallbacks for device execution.


---

## Success Criteria & Definition of Done

- [x] **Clean Build**: `cargo check --workspace` passes with zero compiler errors.
- [x] **Zero-Cost Dispatch Verified**: Monomorphization checked; calling operations on specific backends resolves to direct static execution paths with zero abstraction overhead.
- [x] **Numerical Parity**: Parity tests verify absolute numerical equivalence between CPU, `wgpu`, and `cuda` execution (absolute tolerance $\le 10^{-5}$).
- [x] **Test Coverage**: 100% of tensor operations have verification tests covering contiguous, non-contiguous, broadcasted, and sliced tensor views.
- [x] **Memory Safety**: No memory leaks or data races on GPU backends under parallel operations (RAII wrapper verification).
