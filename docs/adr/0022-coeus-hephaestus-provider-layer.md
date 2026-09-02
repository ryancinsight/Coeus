# ADR 0022: Share Coeus Hephaestus provider integration

## Status

Accepted

Implementation note: implemented as the first ROCm/Metal reduction increment.

## Decision

Coeus owns one generic `coeus-hephaestus` integration layer for Hephaestus
storage, transfers, rank validation, and the `ReductionOps` dispatch contract.
`coeus-rocm` and `coeus-metal` provide only typed provider implementations over
their native Hephaestus device APIs. The provider seam is scalar-specific so
unsupported scalar/kernel combinations fail at compile time rather than being
converted to a different precision or routed through a host implementation.

This increment exposes native rank-2 reductions (`sum`, `product`, `mean`,
`min`, `max`) and forward/reverse cumulative sum/product scans. Ranks above
two return the existing typed Coeus unsupported-rank error. Higher operation
families remain separate increments until their Hephaestus contracts are
integrated.

## Constraints

- Leto is the CPU value-semantic oracle for every provider contract.
- Device buffers remain owned by Hephaestus; Coeus stores reference-counted
  typed handles and does not import vendor runtime types into operation logic.
- A missing device is an unavailable capability. Tests may skip on ordinary
  developer hosts, while CI hardware lanes set a required-device environment
  variable and fail when acquisition is unavailable.
- No CPU fallback is used for a rank or operation unsupported by the native
  provider dispatch.

## Alternatives rejected

Duplicating the existing CUDA/WGPU reduction implementations in two new backend
crates was rejected because it creates three consumer-owned operation trees and
allows semantic drift between provider backends. Making ROCm and Metal aliases
of one public backend type was rejected because it hides the backend identity
and prevents backend-specific capability reporting.

## Verification

The vendor tests compare all five rank-2 reductions and forward/reverse
cumulative sum/product results against `coeus-leto`. Hosted CI compiles the
ROCm feature in the AMD development image, runs a required-device ROCm lane on
the registered runner when manually requested, and runs the required-device
Metal contracts on `macos-15`.
