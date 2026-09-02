# ADR 0044: WGPU reduction provider ownership

- Status: Accepted
- Date: 2026-07-28
- Change class: `[arch] [major]`

## Context

WGPU cumulative scans already dispatched through Hephaestus, but ordinary
sum, product, mean, minimum, and maximum reductions used a second Coeus-owned
WGSL generator and pipeline. The duplicate path maintained its own layout
validation, metadata buffers, shader source, pipeline cache key, and dispatch
logic. Backend selection therefore reached two independent accelerator
implementations for one operation family.

## Decision

`WgpuBackend` dispatches every ordinary reduction directly through the
corresponding Hephaestus rank-two provider entry point. A boundary adapter
maps Coeus rank-one layouts to a singleton-leading rank-two layout and remaps
axis zero to provider axis one. Rank-two layouts retain their original axes.
Other ranks return the typed Coeus `UnsupportedRank` validation error.

The provider is selected once at the reduction boundary. The selected
Hephaestus generic kernel remains monomorphized for the scalar and reduction
marker; no per-element runtime dispatch or host fallback is introduced. The
superseded Coeus ordinary-reduction shader and dispatcher are deleted. Fused
reduction remains a distinct expression-fusion operation and retains its
current Coeus kernel pending a provider expression contract.

## Alternatives

- Retain both implementations. Rejected because two shader generators for one
  backend operation violate provider ownership and can diverge in validation,
  numerical behavior, and tuning.
- Fall back to the Coeus shader for ranks above two. Rejected because runtime
  fallback hides the selected provider's capability boundary.
- Flatten arbitrary-rank tensors before provider dispatch. Rejected because
  preserving arbitrary strides and a selected reduction axis requires a real
  ranked provider contract, not a shape reinterpretation.

## Consequences

Ordinary WGPU reductions above rank two now return a typed error instead of
using the consumer-owned shader. This is a breaking behavioral correction.
Rank-one and rank-two operations remain device-local and use Hephaestus.
Deletion removes duplicated shader construction and metadata-staging code;
no runtime, memory, or binary-size improvement is claimed without matched
measurements.

## Verification

- Compare all five rank-two operations and rank-one sum with the Leto CPU
  oracle.
- Verify rank-one scan remains provider-backed and value-equivalent.
- Assert the exact typed rank-three rejection.
- Run package all-target check, warning-denied Clippy, focused Nextest,
  doctests, and the exact-head provider matrix.
- Scan the WGPU operation tree for the deleted `dispatch_reduce` symbol.
