# ADR-0020: Fallible WGPU layout and dispatch boundary

- Status: accepted
- Date: 2026-07-23
- Scope: `coeus-core`, `coeus-ops`, `coeus-wgpu`, and the matching CUDA/CPU
  operation implementations

## Context

`crates/coeus-wgpu/src/kernels/layout.rs` serializes `coeus_core::Layout` into a
fixed WGSL `u32` descriptor. It currently asserts the rank limit and narrows
rank, offset, shape, and stride values with `as u32`. The same conversion is
used by 23 WGPU kernel sites. Dispatch-grid and kernel-parameter conversions
also remain infallible in multiple WGPU kernels.

The shared `coeus-ops` operation traits return `()`. A WGPU validation failure
therefore has no typed path to the caller. Returning early would leave an
output buffer unchanged or partially written; a CPU fallback would silently
change backend semantics; and a local `Option` would duplicate the failure
contract at every kernel site.

## Decision

Add a backend-associated error type to the consumer-facing backend seam and
make operation dispatch return `Result<(), Self::Error>`. The WGPU backend
owns a typed error enum covering layout rank/value overflow and dispatch ABI
range violations, while CPU and CUDA implementations provide their own typed
errors or `Infallible` where the operation cannot fail under its backend
contract. `GpuLayoutInfo` becomes constructible only through a checked
conversion; all WGPU kernel families consume that validated representation.

Migrate in dependency order: define the shared error associated type and
documentation, migrate one operation family with CPU/CUDA/WGPU implementations
and high-level callers, then repeat for the remaining families. The branch is
the migration boundary; no forwarding wrappers, parallel old methods, or
silent fallback paths are retained.

## Rejected alternatives

- Keep `as u32` and rely on upstream tensor shapes: `Layout` is public and
  supports arbitrary runtime dimensions, so the WGSL ABI remains unchecked.
- Return `Option` or `()` from each WGPU kernel: the caller cannot distinguish
  an empty operation from a rejected operation, and output state is ambiguous.
- Add a WGPU-only adapter or CPU fallback: this forks the shared operation
  contract and hides backend-specific validation failures.
- Use `Box<dyn Error>` at each hot dispatch: it adds dynamic error handling and
  loses the backend-specific compile-time error contract; the associated error
  type preserves monomorphized dispatch.

## Consequences

The public operation API becomes fallible and requires a major-version
migration for external callers. In-repository callers migrate in the same
change. Error construction and conversion remain cold boundary work; kernel
loops retain the existing generic, monomorphized data path. The migration is
larger than a single WGPU file because the shared trait seam owns the failure
contract.

## Verification

Each migrated operation family must test representable and overflowing layout
metadata, dispatch-grid boundaries, and value-semantic CPU/CUDA/WGPU parity
where the backend is available. Format, warning-denied Clippy, doctests,
`cargo nextest run`, and release-sensitive checks run against the exact
revision once dependency resolution is restored. No runtime performance claim
is made by the API migration without a benchmark baseline.

## Implementation status

The elementwise, matmul, and axis-reduction families now use the associated
backend error and fallible operation seams. CPU Leto failures map to the
shared validation error, CUDA preserves provider failures, and WGPU preserves
typed layout and dispatch failures. High-level arithmetic, unary, shape,
matmul, and direct reduction callers use the result contract. The existing
autograd/NN boundary remains infallible and uses explicit invariant checks;
converting those public contracts to typed results is separate breaking work.
No compatibility adapter or silent fallback was retained. The focused
`coeus-ops` gate passes 110/110 nextest tests, 22/22 doctests, warning-denied
Clippy, locked compilation, and no-deps Rustdoc. Coeus root patches collapse
Git-sourced Aequitas/Eunomia/Themis/Hermes identities onto the local Atlas
providers,
which removes the Leto `Quantity<T>::in_unit` trait-identity failure and the
WGPU `PlacementHint` type split from the Coeus locked graph. The locked WGPU
library check passes. The public WGPU matmul wrapper returns the typed result and
checks rank, inner-dimension, output element-count, and layout-conversion
failures; the public add wrapper returns a typed shape error instead of
panicking. Full WGPU all-target verification remains gated by the incomplete
peer fallible-operation migration. No runtime performance or memory claim is
made without profile and benchmark evidence.

The unary WGPU kernel family now consumes `GpuLayoutInfo::try_from_layout`,
returns `Result` through both contiguous and strided dispatch paths, routes
`lgamma` through the provider-owned Hephaestus expression, and validates the
rounded workgroup count before the WGPU ABI boundary. Unit tests cover
rounding, overflow, out-of-range counts, and the provider expression without
initializing a device. Direct nightly rustfmt and `git diff --check` pass. The
locked `coeus-ops` check, 110/110 nextest tests, 22/22 doctests, warning-denied
Clippy, and no-deps Rustdoc now pass; WGPU all-target verification remains
outside this provider-identity integration increment.

The binary WGPU kernel family now uses the same checked layout and workgroup
boundary for contiguous and general broadcasting dispatch. Its `Result` is
propagated through `ElementwiseOps` and the public `add` wrapper. Direct
nightly rustfmt and `git diff --check` pass; the locked `coeus-ops` check and
focused tests pass after the provider-identity cutover.

The axis-reduction family now uses the same typed result boundary. CPU maps
Leto failures, CUDA propagates fallback errors, and WGPU validates layout rank,
axis range, singleton output shape, checked output element count, and checked
workgroup count before device initialization. Public core reduction functions,
direct Coeus tests/benches, and the existing infallible autograd/NN callers use
explicit result handling. The autograd graph and NN module traits remain
infallible; converting those public contracts to typed results is separate
breaking work rather than a local adapter. The locked `coeus-ops` check,
110/110 nextest tests, 22/22 doctests, warning-denied Clippy, and no-deps
Rustdoc pass. No WGPU all-target performance result is claimed because that
matrix remains outside this provider-identity integration increment.
