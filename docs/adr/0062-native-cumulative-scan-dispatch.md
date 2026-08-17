# ADR-0062: Native cumulative-scan dispatch boundary

- Status: Accepted
- Date: 2026-07-26
- Scope: `coeus-ops`, `coeus-wgpu`, and `coeus-cuda`
- Change class: `[major]`

## Context

`ReductionOps::cumsum` and `ReductionOps::suffix_sum` returned unit and their
default bodies copied device buffers to the host before calling the CPU Leto
implementation. WGPU and CUDA therefore exposed the operation names without a
native provider execution contract, and provider errors could not reach the
caller. Hephaestus now exposes both rank-2 operations with allocated and
caller-owned output forms.

## Decision

Make both shared reduction methods return `Result<(), Self::Error>`. WGPU and
CUDA implement the rank-2 path by converting the Coeus layout into the
provider's typed `leto::Layout<2>` view and dispatching Hephaestus
`cumsum_into` or `suffix_sum_into`. Shape, rank, stride-conversion, and provider
dispatch failures remain typed. The native path never falls through to a host
copy. Dynamic-rank expansion is a subsequent increment; unsupported ranks
return the backend error at this boundary.

The high-level Coeus functions retain their existing tensor-returning contract
and terminate a backend error at their established invariant boundary. Direct
backend consumers receive the typed `Result` from `ReductionOps`.

## Rejected alternatives

- Keep the unit-returning methods and call Hephaestus conditionally: provider
  failures would remain unreportable and would require a silent fallback.
- Add a Coeus-local scan kernel: Hephaestus owns the accelerator operation and
  the provider contract already supplies the required rank-2 implementation.
- Copy device values to the CPU for unsupported ranks: that preserves output
  values but violates backend identity and hides capability gaps.

## Verification

The shared CPU implementation, WGPU integration test, and CUDA differential
test cover value-semantic prefix and suffix scans. WGPU library compilation and
the CPU package check pass locally. CUDA feature compilation is pending the
shared Cargo build lock; the hosted CUDA workflow remains the device gate.

## Revisit trigger

Add dynamic-rank provider kernels before widening the Coeus scan acceptance
surface. Do not restore host-copy fallback for a provider with an incomplete
rank surface.
