# Changelog

## 0.2.0 - 2026-06-12

### Added

- Added public `coeus_ops::stack` backed by `coeus-leto` dynamic-rank stack
  dispatch, with `SequentialBackend` and `MoiraiBackend` value-semantic
  coverage for strided input views.
- Added `BackendOps::batched_matmul` as the rank-3 batched matrix multiplication
  backend seam.

### Changed

- Continued Stage A2 CPU consolidation onto `leto` by extending the structural
  dispatch shim to stack operations.
- Routed public batched `coeus_ops::matmul` through `BackendOps::batched_matmul`;
  CPU backends override the seam with `coeus-leto` batched dispatch while GPU
  backends retain the generic default.
- Routed `Tensor::eye_on` identity generation through `coeus-leto` coordinate
  dispatch.

### Fixed

- Fixed zero-length `CpuStorage` so empty tensors expose valid non-null aligned
  Rust slices.
