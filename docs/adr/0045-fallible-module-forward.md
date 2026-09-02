# ADR 0045: Fallible neural-network module execution

- Status: Accepted
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

The module trait has 82 implementations after removing three semantically
invalid unary decoder implementations. Its direct consumers include composite
modules, Rust tests and benchmarks, Python bindings, doctests, and the tensor
benchmark. A parallel fallible API would leave the unsafe contract callable
and split the single source of truth.

## Decision

`Module::forward` returns
`Result<Var<T, B>, ModuleError<B::Error>>`. `ModuleError` preserves the
backend's concrete error as its source and represents module-owned rank,
shape, channel, group, epsilon, insufficient-reduction-width, and
interior-state failures without dynamic dispatch or string erasure.

Infallible leaf modules return `Ok(output)`. Composite modules propagate with
`?`, so execution stops at the first failure. Normalization modules propagate
all backend mutations and replace input-dependent panics with typed module
errors. Attention, transformer, recurrent, embedding, pooling, and unfold/fold
modules validate their complete input and configuration contracts before
indexing, slicing, concatenation, or unchecked output-shape arithmetic.
The benchmark smoke lane also exposed Huber loss reducing only its leading
dimension. Huber loss now uses the complete element count, reconstructs the
original gradient shape, and returns typed contract failures.
The same smoke lane proved that routing `LayerNorm` through the trait had
narrowed its documented trailing-dimension contract to rank two. Trait
dispatch now preserves the canonical rank-two-or-greater implementation.

BatchNorm computes candidate running mean and variance tensors first. It
commits both states only after every candidate mutation succeeds and both
state borrows are acquired. A failed forward pass therefore leaves both
running-stat tensors unchanged.

The `module` bounded context is split into a manifest plus trait and error
leaves. Static backend dispatch and scalar monomorphization are unchanged.
Touched implementation and binding monoliths are split into concern-owned
leaves. Runtime and allocation effects are not claimed without matched
measurements.

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
- Multidimensional Huber tests assert exact loss and gradient values plus typed
  rejection of mismatched shapes.
- Rank-three LayerNorm tests and benchmark smoke prove that `Module::forward`
  retains trailing-dimension normalization semantics.
- Doctests, examples, benchmarks, and SemVer checks cover the new public
  contract.
