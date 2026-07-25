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

The elementwise and matmul families now use the associated backend error and
fallible operation seams. CPU Leto failures map to the shared validation
error, CUDA preserves provider failures, and WGPU preserves typed layout and
dispatch failures. High-level arithmetic, unary, reduction, shape, and matmul
callers propagate the result contract; no compatibility adapter or silent
fallback was retained. The focused `coeus-ops` gate passes 87/87 nextest
tests, 22/22 doctests, warning-denied Clippy, and package-local formatting.
The WGPU library check and warning-denied Clippy pass. The public WGPU matmul
wrapper now returns the typed result and checks rank, inner-dimension, output
element-count, and layout-conversion failures; the public add wrapper now
returns a typed shape error instead of panicking. The locked WGPU all-targets
check reaches compilation but remains blocked by 70 peer `coeus-nn`
normalization errors from the incomplete fallible-operation migration; peer
`coeus-autograd` also emits 143 unused-`Result` warnings. The prior local/Git
Leto dependency-resolution blocker is resolved. No runtime performance or
memory claim is made without profile and benchmark evidence.

The unary WGPU kernel family now consumes `GpuLayoutInfo::try_from_layout`,
returns `Result` through both contiguous and strided dispatch paths, rejects
the unsupported `lgamma` operation with a typed backend error, and validates
the rounded workgroup count before the WGPU ABI boundary. Unit tests cover
rounding, overflow, out-of-range counts, and the unsupported operation without
initializing a device. Direct nightly rustfmt and `git diff --check` pass. The
affected Cargo check remains unverified because the current shared-target run
stops in peer-owned Leto at `crates/leto/src/application/stencil.rs:121-122`:
`Quantity<T>::in_unit` lacks the required `eunomia::traits::float::FloatElement`
bound. No Coeus compilation or test result is claimed for this increment.
