# Changelog

## 0.2.0 - 2026-06-12

### Added

- Added public `coeus_ops::stack` backed by `coeus-leto` dynamic-rank stack
  dispatch, with `SequentialBackend` and `MoiraiBackend` value-semantic
  coverage for strided input views.

### Changed

- Continued Stage A2 CPU consolidation onto `leto` by extending the structural
  dispatch shim to stack operations.
