# Coeus Gap Audit

## Known Gaps & Residual Risks

### G-001: PyTransformerEncoderLayer stateless binding [minor]
**Location**: `coeus-python/src/nn/feedforward.rs` — `PyTransformerEncoderLayer`  
**Description**: The Python binding re-initializes `TransformerEncoderLayer` with
Kaiming-random weights on every `forward()` call and returns `parameters() = []`.
This makes the layer untrinable from Python and prevents weight-parity testing.  
**Fix**: Refactor to store the underlying Rust `TransformerEncoderLayer` as an
`Arc<Mutex<...>>` field (or equivalent) and expose `norm1`, `norm2`, `self_attn`,
and `ffn` sub-modules as `PyObject` fields with `get/set`.  
**Evidence tier**: Type-level / structural analysis (binding source read).  
**Priority**: Blocks `test_transformer_encoder_layer_matches_pytorch`.

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
| G-001 stateless PyTransformerEncoderLayer binding | structural | open |
| G-002 stateless PyTransformerEncoder binding | structural | open |
| ConvTranspose backward WGPU/CUDA coverage | empirical (forward-only) | deferred |
