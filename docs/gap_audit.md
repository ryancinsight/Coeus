# Coeus Gap Audit

## Known Gaps & Residual Risks

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
| ConvTranspose backward WGPU/CUDA coverage | empirical (forward-only) | deferred |
| mnemosyne-backend lib.rs docstring stale | documentation | **closed 87da068** |
