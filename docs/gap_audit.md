# Coeus Gap Audit

## Known Gaps & Residual Risks

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

## Slop Pattern Library

*(Empty — no recurring agent slop patterns identified yet.)*

## Residual Risks

| Risk | Evidence Tier | Status |
|------|--------------|--------|
| G-001 stateless PyTransformerEncoderLayer binding | structural | **closed MS-127** |
| G-002 stateless PyTransformerEncoder binding | structural | **closed MS-128** |
| G-003 stateless PyTransformerDecoderLayer binding | structural | **closed MS-129** |
| G-004 PyTransformerDecoder missing | structural | **closed MS-129** |
| ConvTranspose backward WGPU/CUDA coverage | empirical (forward-only) | deferred |
