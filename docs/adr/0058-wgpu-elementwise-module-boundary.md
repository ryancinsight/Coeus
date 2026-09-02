# ADR 0058: WGPU elementwise module boundary

Status: Accepted
Date: 2026-08-06

## Context

`coeus-wgpu/src/backend/ops/mod.rs` contained the WGPU elementwise provider
routing helpers and the `ElementwiseOps` and `ScalarPowerOps` implementations
alongside the operation-family module declarations. The parent manifest also
provided imports consumed implicitly by the sibling `matmul` and `pool` leaves
through `super::*`.

That layout made the manifest an implementation-bearing namespace and hid
dependency direction. It increased the cost of auditing Hephaestus dispatch,
and a module move could silently break sibling leaves by removing an unrelated
parent import.

## Decision

Move the elementwise provider routing and trait implementations to the named
`backend::ops::elementwise` leaf. Keep `backend::ops::mod.rs` as a module
manifest, and make `matmul.rs` and `pool.rs` import their crate dependencies
explicitly.

The extraction preserves the existing operation boundary and static dispatch:
CPU remains Leto-owned, while WGPU elementwise provider operations remain
Hephaestus-owned. No fallback, host staging path, trait-object dispatch, or
public API change is introduced.

## Alternatives rejected

- Keep implementation in `mod.rs`: rejected because it preserves an
  implementation-bearing manifest and the hidden parent-scope dependency.
- Add a shared `utils` or `helpers` module: rejected because the functions are
  elementwise dispatch responsibilities with one canonical operation-family
  home.
- Replace static provider calls with a runtime backend interface: rejected
  because the existing generic provider seam already monomorphizes the kernel
  and a runtime vtable would add cost without a present requirement.

## Verification

- `cargo fmt --manifest-path ... --all -- --check`
- locked all-target `cargo check` for `coeus-wgpu`
- locked warning-denied all-target `cargo clippy` for `coeus-wgpu`
- focused WGPU Nextest and hosted WGPU provider contracts

The local WGPU test suite requires a compatible adapter; adapter-dependent
failures are an environment limitation and remain hosted-CI acceptance gates.
