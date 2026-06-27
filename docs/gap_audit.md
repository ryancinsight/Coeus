# Coeus Gap Audit

## Known Gaps & Residual Risks

### ~~G-001: PyTransformerEncoderLayer stateless binding~~ **CLOSED**
**Location**: `coeus-python/src/nn/feedforward.rs` — `PyTransformerEncoderLayer`  
**Closed by**: MS-127 — Refactored to stateful `Py<PyLayerNorm>` + `Py<PyMultiHeadAttention>` +
`Py<PyFeedForward>` sub-module fields; `parameters()` returns 16 params; forward replaces
dummy weights from Python sub-objects; `test_transformer_encoder_layer_matches_pytorch` PASSES.

### G-002: PyTransformerEncoder shares the same stateless defect [minor]
**Location**: `coeus-python/src/nn/feedforward.rs` — `PyTransformerEncoder`  
**Description**: Same root cause as G-001; each `forward()` creates fresh layers
with random weights.  
**Fix**: Same pattern as G-001.

## Slop Pattern Library

*(Empty — no recurring agent slop patterns identified yet.)*

## Residual Risks

| Risk | Evidence Tier | Status |
|------|--------------|--------|
| G-001 stateless PyTransformerEncoderLayer binding | structural | **closed MS-127** |
| G-002 stateless PyTransformerEncoder binding | structural | open |
| ConvTranspose backward WGPU/CUDA coverage | empirical (forward-only) | deferred |
