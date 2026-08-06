# ADR-0059: WGPU elementwise routing leaves

Status: Accepted
Date: 2026-08-06
Board item: `COEUS-WGPU-ELEMENTWISE-LEAVES-001`

## Context

ADR-0058 moved WGPU elementwise ownership out of `ops/mod.rs`, but the named
`elementwise.rs` leaf still combined provider-operation classification, layout
conversion, strided dispatch, contiguous dispatch, and public trait
implementations. The combined file was 730 lines and repeated the
parameterized-activation classification in two functions. Its layout bridge
also used a narrowing `usize as isize` conversion before entering Leto.

## Decision

Keep `backend::ops::elementwise` as the canonical operation-family home and
move provider routing policy and layout conversion into
`elementwise/routing.rs`. The routing leaf owns the single activation metadata
table, strided capability guard, const-generic Leto layout conversion, and
checked signed-stride boundary. The remaining dispatch implementations retain
their existing static Hephaestus calls and Coeus trait contracts.

The conversion returns a typed `WgpuBackendError` when a stride cannot be
represented by Leto's signed layout ABI. This prevents a target-dependent
wrapping cast and keeps invalid metadata out of provider dispatch.

## Alternatives rejected

- Keep the duplicate activation matches: rejected because operation metadata
  would have two sources of truth and could diverge.
- Keep the `as isize` conversion: rejected because it can truncate a valid
  WGPU `u32` stride on 32-bit targets before provider execution.
- Introduce a runtime provider interface: rejected because the existing
  generic Hephaestus calls already monomorphize the operation kernel.
- Create a generic utility module: rejected because routing and layout
  conversion belong to this operation family's boundary.

## Invariants

- CPU elementwise execution remains Leto-owned.
- WGPU elementwise execution remains Hephaestus-owned where the provider
  capability and layout contract admit dispatch.
- No host staging, CPU fallback, or trait-object dispatch is introduced. The
  non-exhaustive WGPU layout error gains one diagnostic variant; operation and
  provider trait contracts remain unchanged.
- Layout conversion pads only with broadcast-neutral size-one dimensions and
  zero strides.

## Verification

- `cargo fmt --manifest-path Cargo.toml --all -- --check`
- locked all-targets `cargo check -p coeus-wgpu`
- locked warning-denied `cargo clippy -p coeus-wgpu --all-targets -- -D warnings`
- locked WGPU doctests
- routing unit coverage for const-rank padding
- native WGPU nextest and hosted provider contracts, with adapter availability
  reported separately from code failures

No performance or resident-memory improvement is claimed without a controlled
baseline measurement.
