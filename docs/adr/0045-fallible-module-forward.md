# ADR-0045: Fallible neural-network module execution

- Status: accepted
- Date: 2026-07-28
- Scope: `coeus-nn` module execution and direct workspace consumers
- Change class: `[major] [arch]`

## Context

Backend mutation operations return typed errors. The neural-network
`Module::forward` contract returns only `Var<T, B>`, so normalization modules
cannot propagate those failures. Compilation currently reports 54 ignored
`Result` values across BatchNorm, GroupNorm, InstanceNorm, LayerNorm, and
RMSNorm. A failed operation can therefore expose a partially computed output
or partially updated running statistics while the caller observes success.

The module trait has 85 implementations in 44 source files. Its direct
consumers include composite modules, Rust tests and benchmarks, Python
bindings, doctests, and the tensor benchmark. A parallel fallible API would
leave the unsafe contract callable and split the single source of truth.

## Decision

`Module::forward` returns
`Result<Var<T, B>, ModuleError<B::Error>>`. `ModuleError` preserves the
backend's concrete error as its source and represents module-owned rank,
shape, channel, group, epsilon, and interior-state failures without dynamic
dispatch or string erasure.

Infallible leaf modules return `Ok(output)`. Composite modules propagate with
`?`, so execution stops at the first failure. Normalization modules propagate
all backend mutations and replace input-dependent panics with typed module
errors.

BatchNorm computes candidate running mean and variance tensors first. It
commits both states only after every candidate mutation succeeds and both
state borrows are acquired. A failed forward pass therefore leaves both
running-stat tensors unchanged.

The `module` bounded context is split into a manifest plus trait and error
leaves. Static backend dispatch, scalar monomorphization, and successful-path
allocation behavior are unchanged.

## Rejected alternatives

- Ignore or log failed mutations: this preserves false success and partial
  state.
- Panic with `expect`: backend and input failures are not programmer
  invariants.
- Add `try_forward`: retaining the infallible entry point creates a
  compatibility path and two execution contracts.
- Return only `B::Error`: backend errors cannot represent module-owned state
  borrow and configuration failures.
- Erase errors behind `Box<dyn Error>`: generic propagation preserves the
  concrete error at zero dispatch cost.
- Change `forward` to require `&mut self`: this expands the public break,
  obstructs composition, and still does not propagate backend failures.

## Migration

The trait signature, all implementations, internal compositions, Rust and
Python consumers, tests, benchmarks, and doctests change in one atomic
cutover. Python maps configuration failures to `ValueError` and backend or
state failures to `RuntimeError`. No compatibility wrapper or fallback is
retained.

## Verification

- Warning-denied all-target Clippy reports no ignored result.
- A failure-injecting backend returns its exact source through `ModuleError`.
- BatchNorm failure and conflicting state-borrow regressions leave both
  running-stat tensors unchanged.
- Existing analytical normalization and gradient parity passes through
  Nextest for every shipped backend selected by CI.
- Sequential compositions stop at the first failure.
- Python tests assert exception class and preserved error context.
- Doctests, examples, benchmarks, and SemVer checks cover the new public
  contract.
