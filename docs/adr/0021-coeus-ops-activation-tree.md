# ADR-0021: Separate Activation Operation Families

- Status: accepted
- Date: 2026-07-26

## Context

`coeus-ops/src/unary/activation.rs` contained three bounded concerns: direct
elementwise activations, softmax-family operations, and the gated linear unit.
The flat module mixed their dependencies and made the activation tree grow
horizontally instead of following operation ownership.

## Decision

Move the module manifest to `unary/activation/mod.rs` and place each secondary
operation family in its named leaf module:

- `activation/mod.rs` owns direct elementwise activation entry points;
- `activation/softmax.rs` owns log, masked, and causal softmax;
- `activation/gated.rs` owns GLU.

The manifest re-exports the same public functions. The operation bodies remain
generic over `BackendOps` and `Float`, so the refactor adds no runtime dispatch,
allocation, or compatibility layer.

## Rejected alternative

Keeping all families in `activation/mod.rs` preserves behavior but leaves
unrelated backend contracts and tests coupled to one file. Splitting by
individual function would create excessive module boundaries without a domain
owner.

## Verification

The affected files pass nightly rustfmt and `git diff --check`. Workspace
`cargo metadata --locked --no-deps --offline` passes. Full package compilation
remains blocked by the concurrent provider migration: the standalone Git
Hephaestus source declares `leto` through a local path that is absent from its
Git checkout.
