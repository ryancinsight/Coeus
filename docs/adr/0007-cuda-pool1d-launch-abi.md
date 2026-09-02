# ADR 0007: CUDA pool1d launch ABI

- Status: Accepted
- Date: 2026-07-23
- Change class: [arch]
- Driver: ATLAS-CUDA-SAFETY-007

## Context

The shared 1-D pooling dispatcher narrowed kernel parameters, element counts,
and grid dimensions directly. Its generated kernels assumed rank-three
`[N,C,L]` layouts and nonzero divisors, while public wrappers computed work
counts with unchecked products.

## Decision

Keep one canonical `pool1d` launch boundary for max/average forward and
backward operations. Validate positive representable kernel, stride, and
dilation parameters; checked padding, element counts, and grids; rank-three
nonempty layouts; and operation-specific batch/channel/shape relationships
before kernel compilation or dispatch. Reuse `kernels::validation` for CUDA
ABI primitives and its canonical block size. Preserve native kernels and
allocation/device-buffer ownership.

## Verification

Feature-enabled package check and warning-denied Clippy pass. Default package
Nextest passes 3/3 with zero skipped. A source audit finds no input-dependent
pooling narrowing, unchecked shape product, or local grid/block derivation.

## Revisit trigger

Extend the shared boundary when another pooling dimensionality can use the
same proven validation contract without weakening its shape semantics.
