# Changelog

## 0.2.0 - 2026-06-12

### Added

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

### Fixed

- Fixed zero-length `CpuStorage` so empty tensors expose valid non-null aligned
  Rust slices.
- Fixed rustdoc shape/type annotations that were parsed as intra-doc links or
  HTML so `cargo doc --workspace --no-deps` is warning-clean.
