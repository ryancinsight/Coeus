# ADR-0042: Fallible autograd backward contract

- Status: Accepted
- Date: 2026-07-28
- Scope: `coeus-autograd` graph traversal and operation nodes
- Change class: `[major] [arch]`

## Context

The backend mutation operations used to accumulate gradients return the
backend's typed error. `BackwardNode::backward`, `Var::backward`, and
`Var::backward_with_seed` returned `()`, so operation nodes had no path to
propagate those failures. After the mutation API became fallible, 143
accumulation and direct-dispatch calls ignored `Result` values. A failed
gradient update could therefore leave a partially accumulated graph while
the caller observed success.

## Decision

`BackwardNode::backward` returns `Result<(), B::Error>`. Graph traversal stops
at the first failed node and returns that same backend error through
`Var::backward` or `Var::backward_with_seed`. Every operation node uses `?`
for fallible backend work and returns `Ok(())` only after all requested input
gradients have been accumulated.

The existing backend error type remains the single error contract. No boxed
error, string conversion, compatibility wrapper, fallback, or error-swallowing
branch is introduced. Static dispatch and scalar monomorphization are
unchanged; successful traversal adds no allocation or virtual-dispatch work.

## Rejected alternatives

- Ignore results with `let _ =`: this reports success after a failed gradient
  mutation and can expose a partially accumulated graph.
- Panic with `expect`: backend execution failures are input- or device-
  dependent failures, not local programmer invariants.
- Add a parallel `try_backward` API: retaining the infallible path would
  preserve the unsafe behavior and split one graph contract into two paths.
- Erase backend errors behind `Box<dyn Error>`: the backend already owns the
  precise typed failure and generic propagation is zero-cost.

## Migration

Callers handle or explicitly prove success from `backward()` and
`backward_with_seed()`. In-repository examples and tests use `expect` only
where their fixed valid fixtures establish that backend execution must
succeed.

## Verification

Warning-denied all-target Clippy must report no ignored result. Focused
regressions must inject a backend accumulation failure and assert that
`backward` returns that error rather than continuing silently. Existing
value-semantic gradient tests must pass through Nextest, and public doctests
must compile with the fallible call contract.

The local verification result is:

- warning-denied all-target `coeus-autograd` Clippy passes;
- Nextest passes 102 autograd/FFT and 268 NN tests;
- all 24 executable doctests pass, with two pre-existing NN doctests ignored;
- SemVer checks against `origin/main` identify the public trait-return changes
  as requiring a major release.

Exact-head provider run `30397554467` attempt 2 passes WGPU, ROCm, CUDA, and
Metal. Attempt 1 exposed an upstream Leto stencil contract missing
`T: UnitScalar`; Leto PR #77 repaired that provider-owned bound before the
successful rerun. The required-device ROCm lane is intentionally skipped by
workflow policy.
