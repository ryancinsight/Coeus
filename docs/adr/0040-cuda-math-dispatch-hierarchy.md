# ADR-0040: Vertical CUDA math dispatch modules

- Status: Accepted
- Date: 2026-07-28
- Scope: `coeus-cuda` backend math dispatch

## Context

`crates/coeus-cuda/src/backend/ops/math.rs` contained elementwise, matmul, and
reduction dispatch in one 551-line implementation file. The mixed ownership
made operation-family changes cross-review boundaries and weakened the
canonical module home for provider dispatch.

## Decision

Keep `backend::ops::math` as a private module manifest and place each operation
family in its own leaf: `math::elementwise`, `math::matmul`, and
`math::reduction`. The existing `CudaBackend` method names, trait implementations,
provider routing, and fallback contracts remain unchanged. Each leaf keeps the
same monomorphized generic implementation and depends inward on the existing
backend and kernel seams.

## Rejected alternatives

- Keep the flat file: it preserves the mixed-responsibility boundary and fails
  the repository's vertical hierarchy trigger.
- Duplicate shared dispatch helpers per operation family: it creates divergent
  provider and error behavior instead of moving existing implementations.
- Introduce a dynamic operation registry: it adds runtime dispatch to a
  compile-time backend path without a present requirement.

## Verification

Rustfmt, diff hygiene, locked offline metadata, and the exact-head provider
matrix cover the migration. No runtime, memory, or performance delta is
claimed; this change only relocates existing dispatch implementations.
