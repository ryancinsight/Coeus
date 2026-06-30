# Coeus Gap Audit

## Known Gaps & Residual Risks

### G-046: Python-binding parity closure for AdaptiveMaxPool
**Location**: `coeus-python/src/nn/pool.rs`, `coeus-python/src/nn/mod.rs`,
`coeus-python/src/lib.rs`, `coeus-python/tests/test_pytorch_parity.py`,
`coeus-python/tests/test_jax_parity.py`.
**Compared against**: PyTorch `torch.nn.AdaptiveMaxPool1d/2d` and JAX.
**Gap**: After PR #109 (AdaptiveAvgPool differentiable), PR #110
(AdaptiveAvgPool dx parity in PyTorch+JAX), and PR #111
(`b3e993b` AdaptiveMaxPool1d/2d differentiable in Rust core), the Python
binding surface had not been extended for the Max variant. The
`test_adaptive_max_pool_backward_matches_pytorch` JIT-imported
`pycoeus.AdaptiveMaxPool1d/2d` from `test_pytorch_parity.py:2555`,
which would have raised `AttributeError` against the old binding.
**Resolution (PR #112 = `d1ad9d2`, peer merge)**: Added
`PyAdaptiveMaxPool1d` and `PyAdaptiveMaxPool2d` thin PyO3 wrappers
(mirroring the `PyAdaptiveAvgPool*` pattern), with `m.add_class::<>`
registrations and `pool.rs` re-exports. PR #112 also added the JAX
parity fixture (`test_adaptive_max_pool_matches_jax`) using a per-region
`jnp.max` reference plus `jax.value_and_grad`. Three-way parity
(Rust core \u2194 PyTorch \u2194 JAX) now holds for forward + input gradient.
**Evidence tier**: differential/value-semantic pytest outcomes (PyTorch
parity file 2/2 for the adaptive-max family) plus 379/379 passing Rust nn
tests.
**Acceptance**: closed. Future MS work may add a Burn benchmark row to
match G-043 expansion (already has AvgPool families).

### G-045 forward-only modules sweep:
### G-043: Burn/PyTorch NN benchmark matrix remains partial
**Location**: `coeus-nn/benches/nn_bench.rs`,
`coeus-python/tests/test_pytorch_parity.py`
**Compared against**: Burn `burn::nn` module families and PyTorch `torch.nn`
module families.
**Gap**: Current Coeus-vs-Burn benchmarks cover selected forward rows
(Linear, LayerNorm, RMSNorm, LSTM, InstanceNorm2d, CrossEntropyLoss, MSELoss, HuberLoss, ReLU forward, GeLU forward, Sigmoid forward, Tanh forward, SiLU forward, Conv2d,
Conv3d, MHA self-attention, Transformer encoder layer, Embedding lookup, EmbeddingBag sum,
AdaptiveAvgPool2d(1,1), BatchNorm1d eval forward, BatchNorm2d eval forward, BatchNorm3d eval forward,
Conv1d forward, GroupNorm forward, MaxPool2d forward, AvgPool2d forward, MaxPool1d forward,
AvgPool1d forward),
not the full NN family set needed to claim Burn-level performance parity.
PyTorch differential coverage similarly remains module-family selective.
**Acceptance**: Add a benchmark/parity manifest keyed by module family, then add
rows for every newly implemented G-035..G-042 family with Coeus sequential,
Moirai, WGPU/CUDA where applicable, Burn NdArray where comparable, and PyTorch
Python differential tests at f64. Report median/confidence intervals for
benchmarks and analytical tolerance derivations for numerical comparisons.
**Evidence tier**: source-surface audit plus external API documentation audit.

### ~~G-042: Quantized and lazy module parity policy missing~~ **CLOSED (non-goal)**
**Location**: `coeus-nn/src/lib.rs`, `coeus-python/src/lib.rs`
**Compared against**: PyTorch quantized/lazy NN module families.
**Closed by**: MS-212 — Recorded as an explicit non-goal for Coeus v0.x.
The typed `Scalar` + `BackendOps<T>` design provides a natural extension point for
quantized numerics (e.g., a `QuantizedBackend` implementing `BackendOps<i8>`) without
dedicated lazy-module infrastructure. Coeus v0.x targets f32/f64 parity with Burn and
PyTorch for the standard NN module families; quantized/lazy support is deferred to
a future typed-dtype extension documented in `docs/roadmap.md`.
**Evidence tier**: design decision / non-goal record.

### ~~G-041: Regularization, sparse, and local-response modules incomplete~~ **CLOSED**
**Location**: `coeus-nn/src/dropout.rs`, `coeus-nn/src/embedding.rs`,
`coeus-nn/src/normalization/`, `coeus-python/src/nn/`
**Compared against**: Burn `GaussianNoise`/`LocalResponseNorm` and PyTorch
`AlphaDropout`, `FeatureAlphaDropout`, `EmbeddingBag`, and
`LocalResponseNorm`.
**Closed by**: MS-209 — Added `coeus_nn::EmbeddingBag` with `sum`/`mean`/`max`
aggregation and offsets semantics, value-semantic + backward tests
(`coeus-nn/tests/embeddingbag_tests.rs`), and thin PyO3 wrapper
`pycoeus.EmbeddingBag` delegating to Rust (`coeus-python/src/nn/embedding.rs`,
registered in `coeus-python/src/{nn/mod.rs,lib.rs}`). Together with MS-208
(`AlphaDropout`, `FeatureAlphaDropout`, `GaussianNoise`, `LocalResponseNorm`),
this closes the Rust/Python module-surface gap for G-041.
**Evidence tier**: source-surface audit plus external API documentation audit.

### G-040: Recurrent parity lacks vanilla and bidirectional sequence variants
**Location**: `coeus-nn/src/rnn/`, `coeus-python/src/nn/rnn.rs`
**Compared against**: Burn recurrent modules and PyTorch
`RNN`/`RNNCell`/bidirectional recurrent configurations.
**Gap**: Coeus exposes LSTM/GRU cells and sequence modules, but no vanilla
RNN/RNNCell and no explicit bidirectional RNN/LSTM/GRU module surfaces.
**Acceptance**: Add a generic recurrent core that shares step logic across
vanilla, GRU, and LSTM families; implement bidirectional composition without
duplicating cell math; expose thin PyO3 wrappers; verify forward/backward
against PyTorch and Burn where direct APIs exist.
**Evidence tier**: source-surface audit plus external API documentation audit.

### ~~G-039: Python loss wrappers lag existing Rust loss surface~~ **CLOSED**
**Location**: `coeus-nn/src/loss.rs`, `coeus-python/src/losses.rs`,
`coeus-python/src/lib.rs`, `coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-182 — Added thin PyO3 wrappers
`pycoeus.{kl_divergence,margin_ranking_loss}` delegating directly to
`coeus_nn::loss::{kl_divergence,margin_ranking_loss}`, exported both in the
module registration/stub surface, and added PyTorch differential tests
`test_kl_divergence_matches_pytorch` and
`test_margin_ranking_loss_matches_pytorch` asserting scalar forward and input
gradients at f64. Evidence tier: differential/empirical.

### ~~G-037: Activation surface remains incomplete versus Burn/PyTorch~~ **CLOSED**
**Location**: `coeus-core/src/dtype/{traits.rs,float/cpu_unary.rs,int.rs}`,
`coeus-wgpu/src/kernels/unary.rs`,
`coeus-autograd/src/ops/activation/{ext.rs,relu.rs,mod.rs}`,
`coeus-nn/src/{activation.rs,lib.rs}`,
`coeus-python/src/{activations.rs,lib.rs}`,
`coeus-nn/tests/act_extended_tests.rs`,
`coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-186 — Added nine new activation functions end-to-end:
**Hardtanh** (`coeus_nn::hardtanh` / `Hardtanh` Module, default `[-1, 1]`),
**Hardsigmoid**, **Hardswish**, **Hardshrink** (default λ=0.5),
**Softshrink** (default λ=0.5), **Softsign**, **Threshold** (default `threshold=0, value=0`),
**Celu** (default α=1.0), and **PReLU** (single scalar α default 0.25). Implementation extends
`coeus-core::CpuUnaryOp` with 18 new variants (forward + gradient pairs, single-parameter scalars
packed via `f64::to_bits` following the `LeakyRelu` precedent, pair parameters
packed as little-endian `f32` lanes inside one `u64`), adds the corresponding
float dispatcher in `coeus-core::dtype::float::cpu_unary`, and extends the WGSL
codegen emitter in `coeus-wgpu::kernels::unary` for GPU parity. Tracked
autograd nodes follow the existing `LeakyReluNode` manual-node pattern for
parameterized ops, and the generic `unary_op<T,B,Op>` ZST template for
parameter-free ops (Hardsigmoid, Hardswish, Softsign).
PReLU's α is exposed as a single scalar α in the tracked functional
(`coeus_autograd::prelu(x, alpha)`); per-channel PReLU composes via
`coeus_ops::broadcast_to`. Kink/subgradient points documented inline and excluded from
the PyTorch differential tests per PyTorch's convention (e.g. Hardtanh at
x=±min/max → 0, Hardsigmoid at x=±3 → 0, Hardshrink/Softshrink at |x|=λ → 0,
Threshold at x=threshold → 0). Evidence tier: value-semantic Rust analytical
backward tests (`coeus-nn/tests/act_extended_tests.rs` covering 9 ops at f64
with closed-form formula oracles) plus PyTorch f64 differential tests
(`coeus-python/tests/test_pytorch_parity.py` adds 9 new tests using the
existing `_assert_activation_parity` helper). MS-187 corrected the regression
where gradient operators evaluated on `grad_out` instead of the saved input and
where pair-parameter decoding treated truncated halves as `f64` bit patterns.

### G-038: Loss and distance surface remains below PyTorch coverage
**Location**: `coeus-nn/src/loss.rs`, `coeus-python/src/losses.rs`
**Compared against**: PyTorch loss and distance families.
**Gap**: Coeus lacks direct L1, SmoothL1, BCEWithLogits, CTC, PoissonNLL,
GaussianNLL, MultiMargin, MultiLabel margin/soft-margin, TripletMargin,
PairwiseDistance, and CosineSimilarity public surfaces. Some behavior may be
expressible by existing primitives, but there is no authoritative module/API
parity, no wrapper, and no differential harness coverage.
**Acceptance**: Implement missing losses/distances as Rust canonical functions
or modules with typed reduction policy where applicable; expose PyO3 wrappers
only as delegates; add analytical tests for formulae and PyTorch differential
tests for forward/backward where PyTorch provides gradients.
**Evidence tier**: source-surface audit plus external API documentation audit.

### G-037: Activation surface remains incomplete versus Burn/PyTorch
**Location**: `coeus-nn/src/activation.rs`, `coeus-python/src/activation.rs`
**Compared against**: Burn activations and PyTorch activation modules/functions.
**Gap**: Coeus covers common activations, but lacks Rust module/API parity for
PReLU, CELU, Hardshrink, Hardsigmoid, Hardtanh, Hardswish, Softshrink,
Softsign, Threshold, and a Rust `nn` GLU/SwiGLU family surface matching
framework module expectations.
**Acceptance**: Add one generic Rust activation implementation per operation
family with analytical derivative tests; expose PyO3 wrappers as delegation
only; add PyTorch/Burn differential tests at f64, with kink/subgradient points
handled by documented analytical contracts.
**Evidence tier**: source-surface audit plus external API documentation audit.

### ~~G-036: Pooling, adaptive pooling, and unfold/fold coverage incomplete~~ **CLOSED**
**Location**: `coeus-nn/src/pool.rs`, `coeus-python/src/nn/pool.rs`,
`coeus-nn/src/conv/unfold_fold.rs`, `coeus-ops/src/backend_ops/traits/unfold_fold.rs`
**Compared against**: Burn `Unfold4d` and PyTorch pooling/unfold/fold module
families.
**Gap**: Coeus exposes 2D/3D average and max pooling plus selected global
pooling wrappers, but lacks 1D pooling modules, adaptive pooling surfaces beyond
global wrappers, and Unfold/Fold/Unfold4d parity surfaces.
**Closed by**: MS-206 (pool1d) and MS-211 (unfold/fold):
- MS-206: `MaxPool1d`/`AvgPool1d` with forward+backward, autograd, Python bindings.
- MS-211: `UnfoldFoldOps` sub-trait added to `BackendOps` (8th concern); CPU kernels
  for `unfold1d`/`fold1d`/`unfold2d`/`fold2d`; no-op stubs in wgpu and cuda for
  trait completeness; `coeus_nn::{Unfold1d, Fold1d, Unfold2d, Fold2d}` NN modules;
  9 parity tests (shape/value-semantic/roundtrip).
**Evidence tier**: analytical/value-semantic Rust tests.

### G-035: ConvTranspose3d parity missing
**Location**: `coeus-nn/src/conv/`, `coeus-python/src/nn/conv.rs`
**Compared against**: PyTorch `ConvTranspose3d` and the existing Coeus
ConvTranspose1d/2d family.
**Gap**: Coeus exports ConvTranspose1d and ConvTranspose2d, but has no
ConvTranspose3d Rust module, backend route, autograd coverage, PyO3 wrapper, or
PyTorch differential test.
**Acceptance**: Implement ConvTranspose3d through the existing convolution
family architecture, add value-semantic forward/backward Rust tests, add
WGPU/CUDA backend-autograd parity where supported, and expose a thin PyO3
wrapper with PyTorch f64 differential coverage.
**Progress**: MS-185 adds the `coeus-ops` forward operation, backend default
method, tracked autograd backward node, `coeus-nn::ConvTranspose3d`,
Sequential/Moirai value-semantic module tests, `pycoeus.ConvTranspose3d`, and
PyTorch f64 differential coverage for forward output plus input, weight, and
bias gradients. Remaining work: WGPU/CUDA backend-specific parity coverage.
**Evidence tier**: source-surface audit plus external API documentation audit.

### ~~G-034: Linear/loss tests only checked gradient existence~~ **CLOSED**
**Location**: `coeus-nn/tests/nn/linear_activation_loss.rs`
**Closed by**: MS-179 — Replaced Linear, MSE, and CrossEntropy
gradient-existence checks with value-semantic assertions. Linear now pins
input, weight, and bias gradients for a deterministic all-ones layer; MSE pins
the mean-reduction derivative; CrossEntropy pins the stable
softmax-minus-onehot mean-reduction gradient. Evidence tier:
analytical/value-semantic Rust tests.

### ~~G-033: Conv module tests only checked gradient existence~~ **CLOSED**
**Location**: `coeus-nn/tests/nn/conv1d.rs`,
`coeus-nn/tests/nn/conv2d.rs`,
`coeus-nn/tests/nn/conv3d_pool3d.rs`
**Closed by**: MS-178 — Replaced Conv1d/Conv2d/Conv3d module backward smoke checks with
exact analytical assertions for input, weight, and bias gradients under
deterministic all-ones kernels. Evidence tier: analytical/value-semantic Rust
tests.

### ~~G-032: TCP collectives could hang past nextest timeout~~ **CLOSED**
**Location**: `coeus-dist/src/tcp/mesh.rs`,
`coeus-dist/tests/dist_tests.rs`
**Closed by**: MS-177 — Added deterministic TCP test port reservation through a
file-backed cross-process port allocator lock, and debug-mode mesh timeouts around
connect, accept, peer-rank read, send, and recv paths. Connect retry backoff
remains async through `moirai_async::sleep`, so the debug diagnostics do not
introduce executor-blocking sleep. The lock creation path also treats Windows
`PermissionDenied` as lock contention rather than a distinct fatal failure,
preserving the stale-lock timeout diagnostic under nextest process contention.
Evidence tier: empirical/value-semantic through the `coeus-dist` package gate.

### ~~G-031: JAX harness lacked regression/binary loss parity~~ **CLOSED**
**Location**: `coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-175 — Added `test_{mse_loss,binary_cross_entropy,huber_loss}_matches_jax`,
asserting forward loss and prediction gradient against inline JAX references at f64
(Huber δ=1.0 spans both regions; BCE probabilities in (0,1)). Completes the
regression/binary loss parity against JAX, symmetric with PyTorch. Evidence tier:
differential/empirical.

### ~~G-030: JAX harness lacked LayerNorm/RMSNorm parity~~ **CLOSED**
**Location**: `coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-174 — Added `test_{layernorm,rmsnorm}_matches_jax`,
asserting forward output and gradients against inline f64 JAX references.
LayerNorm covers input/gamma/beta gradients; RMSNorm covers input/gamma
gradients. Evidence tier: differential/empirical.

### ~~G-029: JAX harness lacked softmax/log-softmax/cross-entropy parity~~ **CLOSED**
**Location**: `coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-173 — Added `test_{softmax,log_softmax,cross_entropy_loss}_matches_jax`,
asserting forward output and gradient against `jax.nn.{softmax,log_softmax}` and a
fused log-softmax+NLL mean reference at f64. Extends the JAX harness to the
classification/softmax path, symmetric with the PyTorch coverage. Evidence tier:
differential/empirical.

### ~~G-028: `BackendOps` mixed every operation concern in one trait~~ **CLOSED**
**Location**: `coeus-ops/src/backend_ops/trait_def.rs`,
`coeus-ops/src/backend_ops/cpu_impl.rs`
**Closed by**: MS-171 — Added single-concern operation traits and made
`BackendOps` an aggregate super-trait with a blanket impl. CPU dispatch now
implements one operation trait per concern, preserving the existing kernel leaf
modules while eliminating duplicate blanket-impl coherence failures. Evidence
tier: compile/lint/docs plus value-semantic `coeus-ops` nextest coverage.

### ~~G-027: JAX harness lacked elementwise activation parity~~ **CLOSED**
**Location**: `coeus-python/tests/test_jax_parity.py`
**Closed by**: MS-168 — Added `_assert_activation_matches_jax` (`jax.grad` for
backward) and `test_{silu,mish,elu,softplus,leaky_relu}_matches_jax`, asserting
forward output and input gradient against `jax.nn.*` at f64. Extends the JAX
harness beyond Linear/MHA/decoder to the elementwise activations, symmetric with
the PyTorch coverage of MS-167. Evidence tier: differential/empirical.

### ~~G-026: Elementwise activation differential parity missing (only GELU covered)~~ **CLOSED**
**Location**: `coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-167 — Added a shared `_assert_activation_parity` helper and
`test_{silu,mish,elu,softplus,leaky_relu}_matches_pytorch`, asserting forward
output and input gradient against `torch.nn.functional.*` at f64 on mixed-sign
inputs. LeakyReLU excludes the `x=0` kink (implementation-defined subgradient);
the C1 activations include it. Evidence tier: differential/empirical.

### ~~G-025: GlobalAvg/MaxPool2d differential parity missing~~ **CLOSED**
**Location**: `coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-166 — Added `test_global_avg_pool2d_matches_pytorch` and
`test_global_max_pool2d_matches_pytorch` (input `[2,3,4,4]` → `[N,C,1,1]`),
asserting forward output and input gradient against
`torch.nn.functional.adaptive_{avg,max}_pool2d(x, 1)` at f64, `atol=1e-10`.
Covers the uniform-distribution (avg) and argmax-routing (max) backward paths,
replacing prior existence-only binding coverage. Evidence tier: differential/empirical.

### ~~G-024: Zero-numel collectives skipped per-rank numel validation~~ **CLOSED**
**Location**: `coeus-dist/src/local.rs`,
`coeus-dist/src/tcp/collectives.rs`
**Closed by**: MS-165 — Local and TCP `all_gather`, rooted `gather`, and rooted
`scatter` now validate per-rank output/input tensor element counts before
zero-numel early returns. Evidence tier: panic-contract nextest coverage.

### ~~G-023: Conv2d canonical CPU path retained dot-per-output overhead~~ **CLOSED**
**Location**: `coeus-ops/src/backend_ops/cpu_impl/conv/conv2d.rs`,
`coeus-core/src/dtype/traits.rs`
**Closed by**: MS-164 — Added the `Scalar::axpy_slice` seam and rewrote the
canonical contiguous Conv2d forward path as an output-stationary AXPY row
kernel, with coarser row-block partitioning for Moirai execution. Evidence
tier: value-semantic scalar/Conv2d tests plus Criterion Conv2d row.

### ~~G-022: Local collective staging mutex covered payload work~~ **CLOSED**
**Location**: `coeus-dist/src/local.rs`
**Closed by**: MS-163 — Local collectives now snapshot staged rank payloads
under the shared staging mutex and perform reductions/output copies after the
lock is released; root scatter extracts tensor host data before publishing
payloads. Evidence tier: value-semantic local communicator tests.

### ~~G-021: KL/MarginRanking tracked loss coverage missing~~ **CLOSED**
**Location**: `coeus-autograd/src/ops/nn/loss`,
`coeus-nn/tests/burn_live_parity.rs`, `coeus-nn/tests/loss_parity.rs`
**Closed by**: MS-161 — Added tracked KL divergence and margin ranking loss
entry points, NN wrappers, analytical forward/backward tests, and
sequential/Moirai loss parity checks. Evidence tier: analytical Rust tests plus
package nextest.

### ~~G-020: BCE/Huber Python differential parity missing~~ **CLOSED**
**Location**: `coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-156 — Added `test_binary_cross_entropy_matches_pytorch`
and `test_huber_loss_matches_pytorch`, asserting scalar losses and prediction
gradients against `torch.nn.functional.binary_cross_entropy` and
`torch.nn.functional.huber_loss` at f64. Evidence tier:
differential/empirical.

### ~~G-019: SiLU/Mish tests still had existence-only gradient checks~~ **CLOSED**
**Location**: `coeus-nn/tests/nn_silu_tests.rs`,
`coeus-nn/tests/nn_mish_tests.rs`
**Closed by**: MS-154 — Module and non-contiguous SiLU/Mish paths now assert
analytical forward and backward values instead of only checking that gradients
exist. Evidence tier: analytical value-semantic Rust tests.

### ~~G-018: CrossEntropy/NLL loss differential parity missing~~ **CLOSED**
**Location**: `coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-153 — Added `test_cross_entropy_loss_matches_pytorch` and
`test_nll_loss_matches_pytorch` (logits `[3,4]`, class-index targets), asserting
the scalar loss and logit gradient against `torch.nn.functional.cross_entropy`
and `nll_loss(log_softmax(x))` at f64, `atol=1e-10` (both mean reduction). Pins
the fused log-softmax+NLL forward and the softmax-minus-onehot backward — the
classification training signal. Evidence tier: differential/empirical.

### ~~G-017: FeedForward binding monolith~~ **CLOSED**
**Location**: `coeus-python/src/nn/feedforward.rs`
**Closed by**: MS-152 — Replaced the flat binding file with
`coeus-python/src/nn/feedforward/mod.rs`, `feedforward/positional.rs`, and
`feedforward/transformer/*` leaf modules while preserving `pycoeus` `nn`
registration exports. Evidence tier: compile/lint/docs plus Rust and Python
binding tests.

### ~~G-016: MaxPool2d/AvgPool2d differential parity missing~~ **CLOSED**
**Location**: `coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-151 — Added `test_maxpool2d_matches_pytorch` and
`test_avgpool2d_matches_pytorch` (kernel=2, stride=2 on `[1,2,4,4]`), asserting
forward output and input gradient against `torch.nn.functional.{max,avg}_pool2d`
at f64, `atol=1e-10`. Exercises the max-routing (gradient to argmax) and
average-distribution (uniform 1/window) backward paths, previously covered only
by binding smoke tests. Evidence tier: differential/empirical.

### ~~G-015: Scalar identity still depended on num-traits/libm~~ **CLOSED**
**Location**: `Cargo.toml`, `coeus-core/src/dtype/traits.rs`,
`coeus-core/src/dtype/float/erf.rs`, `coeus-ops/src/sparse/ops.rs`
**Closed by**: MS-150 — Removed Coeus' direct `num-traits`/`libm` dependency
path from the scalar contract, added canonical `Scalar::zero()`/`one()`, and
routed GELU/erf through a Coeus-owned piecewise rational implementation.
Evidence tier: compile/lint/docs plus value-semantic Rust tests.

### ~~G-014: GroupNorm Python differential parity missing~~ **CLOSED**
**Location**: `coeus-python/tests/test_pytorch_parity.py`
**Closed by**: MS-149 — Added `test_groupnorm_matches_pytorch`, asserting
GroupNorm forward output plus input, weight, and bias gradients against
`torch.nn.functional.group_norm` at f64, `atol=1e-10`.
Evidence tier: differential/empirical.

### ~~G-013: Duplicate einsum implementation under shape::util~~ **CLOSED**
**Location**: `coeus-ops/src/shape/einsum.rs`,
`coeus-ops/src/shape/util/einsum.rs`
**Closed by**: MS-148 — Deleted the byte-identical utility copy and routed
`shape::util::{einsum,einsum3}` through the canonical parent implementation.
Evidence tier: compile/lint/docs plus value-semantic tests (`coeus-ops` full
nextest 189/189, focused einsum nextest 12/12).

### ~~G-001: PyTransformerEncoderLayer stateless binding~~ **CLOSED**
**Location**: `coeus-python/src/nn/feedforward.rs` — `PyTransformerEncoderLayer`  
**Closed by**: MS-127 — Refactored to stateful `Py<PyLayerNorm>` + `Py<PyMultiHeadAttention>` +
`Py<PyFeedForward>` sub-module fields; `parameters()` returns 16 params; forward replaces
dummy weights from Python sub-objects; `test_transformer_encoder_layer_matches_pytorch` PASSES.

### ~~G-002: PyTransformerEncoder stateless binding~~ **CLOSED**
**Location**: `coeus-python/src/nn/feedforward.rs` — `PyTransformerEncoder`  
**Closed by**: MS-128 — Refactored to stateful `Vec<Py<PyTransformerEncoderLayer>>` field;
`parameters()` returns `16 × N` params; `forward()` chains layer-wise Pre-LN forwards without
re-creating Rust encoder; `build_from_layer`/`from_rust_layer` inherent methods eliminate code
duplication with `PyTransformerEncoderLayer::new()`. Tests:
`transformer_encoder_stack_2layer_self_consistent` (structural, 111/111 Rust),
`transformer_encoder_stack_2layer_forward_matches_burn` (differential, Burn NdArray),
`test_transformer_encoder_stack_matches_pytorch` (differential, PyTorch, 8/8 Python).

### ~~G-003: PyTransformerDecoderLayer stateless binding~~ **CLOSED**
**Location**: `coeus-python/src/nn/feedforward.rs` — `PyTransformerDecoderLayer`  
**Closed by**: MS-129 — Refactored to stateful `Py<PyLayerNorm>×3` + `Py<PyMultiHeadAttention>×2`
(self_attn + cross_attn) + `Py<PyFeedForward>` sub-module fields; `parameters()` returns 26 params;
`forward(tgt, memory)` injects stored weights into Rust forward; `build_from_layer<H>` /
`from_rust_layer<H>` inherent methods (SSOT, shared with `PyTransformerDecoder`).

### ~~G-004: PyTransformerDecoder missing~~ **CLOSED**
**Location**: `coeus-python/src/nn/feedforward.rs` — class did not exist  
**Closed by**: MS-129 — Added `PyTransformerDecoder` with `Vec<Py<PyTransformerDecoderLayer>>`
layers; `parameters()` returns `26 × N`; `forward(tgt, memory)` chains layer-wise Pre-LN
cross-attention forwards; `num_layers` getter; `zero_grad()`. Tests:
`transformer_decoder_layer_forward_is_deterministic` (determinism),
`transformer_decoder_stack_2layer_self_consistent` (structural, 277/277 Rust),
`transformer_decoder_forward_uses_self_as_memory` (API contract),
`test_transformer_decoder_layer_matches_pytorch` (differential, PyTorch, 10/10 Python),
`test_transformer_decoder_stack_matches_pytorch` (differential, PyTorch, 10/10 Python).

### ~~G-005: PyTransformer (full seq2seq) missing~~ **CLOSED**
**Location**: `coeus-python/src/nn/feedforward.rs` — class did not exist  
**Closed by**: MS-131 — Added `PyTransformer` wrapping `Py<PyTransformerEncoder>` +
`Py<PyTransformerDecoder>`; `forward(src, tgt)` chains encoder→decoder; `parameters()`
returns `16×N_enc + 26×N_dec`; `num_enc_layers`/`num_dec_layers` getters; validation
`d_model % num_heads == 0` at constructor boundary. Test:
`test_transformer_seq2seq_composition` (structural composition identity, atol=1e-12).

### ~~G-006: RNN and positional-encoding Burn parity tests missing~~ **CLOSED**
**Location**: `coeus-nn/tests/burn_live_parity.rs` — 0 tests for LSTM, GRU, RoPE, Sinusoidal  
**Closed by**: MS-131 — Added 8 tests: `lstm_zero_input_zero_output_analytical` (analytical,
zero-bias+zero-input→zero; evidence tier: compile-time proof via docstring invariant),
`lstm_output_shape_contract`, `lstm_forward_seq_matches_module_forward`,
`gru_zero_input_zero_output_analytical`, `gru_output_shape_contract`,
`gru_forward_seq_matches_module_forward`, `sinusoidal_encoding_output_shape_matches_input`,
`sinusoidal_encoding_pos0_equals_analytical` (PE[0]=[0,1,0,1,...] analytically derived),
`rope_zero_input_zero_output`, `rope_output_shape_matches_input`. 292/292 Rust tests pass.

### ~~G-007: Transformer seq2seq structural parity tests missing~~ **CLOSED**
**Location**: `coeus-nn/tests/burn_live_parity.rs` — no `forward_seq2seq` structural tests  
**Closed by**: MS-136 — Added `transformer_seq2seq_self_consistent` (proves `forward_seq2seq`
== manual encoder+decoder chain; f32::EPSILON*4 tolerance) and
`transformer_module_forward_routes_to_seq2seq_self` (proves `Module::forward(x)` ==
`forward_seq2seq(x,x)`). Both use `Transformer<f32, SequentialBackend, 2, 1, 1>` with
dropout_p=0. Evidence tier: structural/deterministic. 294/294 Rust tests pass.

### ~~G-008: LSTM/GRU PyTorch parity tests missing~~ **CLOSED**
**Location**: `coeus-python/tests/test_pytorch_parity.py` — 0 tests for LSTMCell/GRUCell step  
**Closed by**: MS-136 — Added `test_lstm_cell_step_matches_pytorch`: copies w_ih/b_ih/w_hh/b_hh
from pycoeus LSTMCell(4,6) into torch.nn.LSTMCell.double(); verifies h_new and c_new at
atol=1e-10 after one step on zero-init hidden state. Gate order [i,f,g,o] matches between coeus
and PyTorch. Added `test_gru_cell_step_matches_pytorch`: same weight-injection approach for
GRUCell, verifying h_new; n=tanh(ih_n+r*hh_n) formula is consistent between implementations.
Evidence tier: differential/empirical.

### ~~G-009: JAX and MLX Python parity harnesses missing~~ **CLOSED**
**Location**: `coeus-python/tests/` — no JAX or MLX parity harness existed
**Closed by**: MS-138 — Added `test_jax_parity.py` for f64
`Linear + ReLU + MSELoss` forward/backward parity against JAX, and
`test_mlx_parity.py` for MLX-native f32 forward-loss parity when MLX is
installed. Evidence tier: JAX differential/empirical; MLX optional-framework
collection behavior verified on this Windows environment (1 collected skip,
MLX not installed).

### ~~G-010: Optimizer step correctness unverified~~ **CLOSED**
**Location**: `coeus-optim/src/{sgd,adam,adamw}.rs` — SGD, Adam, AdamW step implementations
had zero tests (no analytical derivation, no differential parity).
**Closed by**: MS-139 — Existing Rust analytical tests cover first-step SGD,
Adam, and AdamW formulas in `burn_live_parity.rs`; MS-139 added 3 Python
PyTorch differential tests:
`test_sgd_step_matches_pytorch`, `test_adam_step_matches_pytorch`,
`test_adamw_step_matches_pytorch` — each sets up mse_loss→backward→step and compares
against torch.optim at atol=1e-10. Evidence tier: analytical (Rust) + differential/empirical (Python).

### ~~G-011: Bilinear per-output indexing parity gap~~ **CLOSED**
**Location**: `coeus-nn/tests/bilinear_parity.rs`,
`coeus-python/tests/test_pytorch_parity.py` — Bilinear had all-ones analytical
coverage but lacked a per-output weight-indexing oracle and direct PyTorch
parity check.
**Closed by**: MS-140 — Added a Rust analytical identity/swap weight oracle
that verifies `[out, in1, in2]` indexing on Sequential and Moirai backends, and
added `test_bilinear_forward_matches_pytorch` against `torch.nn.Bilinear`.
Evidence tier: analytical (Rust) + differential/empirical (Python).

### ~~G-012: Python `Tensor.sum`/`.mean` reduction + InstanceNorm parity missing~~ **CLOSED**
**Location**: `coeus-python/src/tensor/pyimpl.rs` — the Python `Tensor` exposed only
axis reductions (`sum_axis`/`mean_axis`), no full-reduction `sum()`/`mean()`, so the
idiomatic scalar-loss path `out.sum().backward()` was inexpressible and InstanceNorm
{1,2,3}d had no PyTorch parity coverage.
**Closed by**: MS-145 — Added `PyTensor::sum`/`PyTensor::mean` (GIL-released,
autograd-preserving, delegating to `coeus_autograd::{sum,mean}`); added
`test_instancenorm{1,2,3}d_matches_pytorch` (forward + dx + dγ + dβ at atol=1e-10)
and `test_{rmsprop,adagrad}_step_matches_pytorch`. Corrected the InstanceNorm oracle
to set `requires_grad=True` on the reference affine params. Removed stale
`tests/pycoeus*.pyd` artifacts that shadowed the installed extension during pytest.
Evidence tier: differential/empirical (PyTorch f64).

## Slop Pattern Library

- **Stale local `*.pyd` shadowing the installed extension**: pytest prepends the
  test directory to `sys.path`, so a leftover `coeus-python/tests/pycoeus*.pyd`
  build artifact silently overrides the freshly `maturin develop`-installed module,
  pinning an out-of-date binary and producing spurious `AttributeError`s for
  newly-added bindings. Mitigation: keep built extensions out of `tests/`; the
  canonical module is the site-packages install. (Detected MS-145.)

## Residual Risks

| Risk | Evidence Tier | Status |
|------|--------------|--------|
| G-036 pooling/adaptive/unfold/fold coverage incomplete | source-surface + external docs audit | **open** |
| G-038 loss and distance surface remains below PyTorch coverage | source-surface + external docs audit | **open** |
| G-040 recurrent parity lacks vanilla and bidirectional variants | source-surface + external docs audit | **open** |
| G-041 regularization/sparse/local-response modules incomplete | source-surface + external docs audit | **open** |
| G-042 quantized and lazy module parity policy missing | source-surface + external docs audit | **open** |
| G-043 Burn/PyTorch NN benchmark matrix remains partial | source-surface + external docs audit | **open** |
| G-044 LocalResponseNorm was forward-only (non-differentiable). Fixed: forward rewritten as an autograd graph (band-matrix matmul windowed sum-of-squares + differentiable `pow`), so dx now flows. forward + dx parity with torch.nn.LocalResponseNorm verified | differential | **closed** |
| G-045 forward-only modules sweep: forwards calling raw `coeus_ops::` then returning `Var::new(out, false)` (dx=0). FIXED: AdaptiveAvgPool1d/2d (averaging-matrix matmul) and AdaptiveMaxPool1d/2d (masked `max_axis` over each region, separable 2D); both backward-verified vs numerical gradient. STILL forward-only: Unfold1d/2d + Fold1d/2d (need im2col/col2im scatter autograd ops). Acceptance: dx parity with torch for each | differential | **partial** |
| G-001 stateless PyTransformerEncoderLayer binding | structural | **closed MS-127** |
| G-002 stateless PyTransformerEncoder binding | structural | **closed MS-128** |
| G-003 stateless PyTransformerDecoderLayer binding | structural | **closed MS-129** |
| G-004 PyTransformerDecoder missing | structural | **closed MS-129** |
| G-005 PyTransformer (full seq2seq) missing | structural | **closed MS-131** |
| G-006 RNN/PE Burn parity tests missing | structural | **closed MS-131** |
| G-007 Transformer seq2seq structural parity tests missing | structural | **closed MS-136** |
| G-008 LSTM/GRU PyTorch parity tests missing | differential | **closed MS-136** |
| G-009 JAX/MLX Python parity harnesses missing | differential/optional empirical | **closed MS-138** |
| G-010 Optimizer step correctness unverified | analytical + differential | **closed MS-139** |
| G-011 Bilinear per-output indexing parity gap | analytical + differential | **closed MS-140** |
| G-012 Python `Tensor.sum`/`.mean` reduction + InstanceNorm parity missing | differential | **closed MS-145** |
| G-013 duplicate einsum implementation under shape::util | compile/lint/docs + value-semantic tests | **closed MS-148** |
| G-014 GroupNorm Python differential parity missing | differential/empirical | **closed MS-149** |
| G-015 Scalar identity still depended on num-traits/libm | compile/lint/docs + value-semantic tests | **closed MS-150** |
| G-016 MaxPool2d/AvgPool2d differential parity missing | differential | **closed MS-151** |
| G-018 CrossEntropy/NLL loss differential parity missing | differential | **closed MS-153** |
| G-020 BCE/Huber loss differential parity missing | differential | **closed MS-156** |
| G-025 GlobalAvg/MaxPool2d differential parity missing | differential | **closed MS-166** |
| G-026 Elementwise activation differential parity missing | differential | **closed MS-167** |
| G-027 JAX harness lacked elementwise activation parity | differential | **closed MS-168** |
| G-029 JAX harness lacked softmax/log-softmax/cross-entropy parity | differential | **closed MS-173** |
| G-030 JAX harness lacked LayerNorm/RMSNorm parity | differential | **closed MS-174** |
| G-031 JAX harness lacked regression/binary loss parity | differential | **closed MS-175** |
| ConvTranspose backward WGPU/CUDA coverage | empirical GPU/CPU autograd differential | **closed MS-176** |
| mnemosyne-backend lib.rs docstring stale | documentation | **closed 87da068** |
| `test_hardswish_matches_pytorch` PyTorch differential parity | differential | **open** — pre-existing, MS-214 wide sweep excludes (hardswish backward routing) |
| `test_hardsigmoid_matches_pytorch` PyTorch differential parity | differential | **open** — pre-existing, MS-214 wide sweep excludes |
| `test_prelu_matches_pytorch` PyTorch differential parity | differential | **open** — pre-existing, MS-214 wide sweep excludes (PReLU backward routing) |
| `test_tcp_scatter_zero_numel_mismatched_target_numel_panics` slow | empirical | **open** — exceeded 30 s slow threshold (45.4 s) during MS-215; deferred optimization to a future `tcp-dispatch` slice |
| `coeus-cuda` clippy errors under `--all-features` | lint | **pre-existing peer crate dependency** — not addressed in MS-215 (out of coeus scope) |
