# ADR 0079: Fallible Hephaestus provider initialization and transfer

Status: Accepted  \
Date: 2026-10-10  \
Change class: [major] [arch]  \
Board item: [ATLAS-COEUS-SAFETY-001](../../backlog.md#atlas-coeus-safety-001)
Revision 2026-10-16: implemented and accepted. The caller survey grew from
eleven files to 35 production acquisition sites (the `B::device()` seam
path, cross-entropy/stateful-update provider-associated paths, and CUDA/WGPU
fusion entry points were not counted in the original audit); `try_device()`
was added to `HephaestusSeamBackend` alongside the provider trait, and the
infallible `device()` surface was deleted from both traits, all seven vendor
and test implementations, and the `get_cuda_device`/`get_wgpu_context`
helpers. Representative-pattern unavailable-device dispatch tests
(`coeus-hephaestus/src/dispatch_tests.rs`) and a production panic scan
(zero `.expect()`/`.unwrap()`; one provably-unreachable internal invariant)
land with the migration.

## Context

`HephaestusProvider::device()` returns `&'static Self::Device` and every
vendor implements it with `.expect(...)` inside `OnceLock::get_or_init`,
so a missing or failed accelerator panics inside a library boundary
instead of reporting a typed acquisition failure. The generic
`ComputeBackend` for `HephaestusBackend<P>` extends the same boundary to
storage operations: allocation, uniqueness re-allocation, and both
transfer directions call `.expect(...)` on provider results. A direct
workspace search found the panic surface: four vendor `device()` bodies
(CUDA, WGPU, Metal, ROCm), five provider/storage/transfer expects in
`coeus-hephaestus` (allocation, uniqueness allocation, uniqueness
device copy, host-to-device copy, device-to-host copy), and eleven
caller files reaching the device through `P::device()`.

The trait already declares the fallible spelling,
`try_device() -> hephaestus_core::Result<&'static Self::Device>`, and
every vendor implements it. The migration therefore converges callers
onto the existing fallible name rather than inventing a parallel one.

## Decision

Make provider initialization and transfer fallible at the existing
seams, in place:

```text
P::device() callers  -> P::try_device(), mapping HephaestusError into
                        the backend's typed error at each call site
allocation / uniqueness / transfer expects
                     -> Result propagation through ComputeBackend
HephaestusProvider::device()
                     -> deleted once its caller closure is empty
```

`device()` is deleted rather than deprecated: ADR 0078 forbids parallel
`try_*` siblings as separate sources of truth, and `try_device()` is
already the surviving name. The deletion is mechanical (four vendor
bodies) after the caller migration. No backend silently falls back to
CPU storage; a failed acquisition or transfer surfaces the provider's
typed failure with the operation name attached. Lock-poisoning expects
(`read detached/retained storage`) and test-only expects are explicitly
out of scope: they guard internal invariants, not device fallibility.

The migration is split into dependency ordered leaves: the generic
`ComputeBackend` storage/transfer paths and their value-semantic
failure tests first (no output published on failure), then the
eleven `P::device()` caller files in dependency order, then the
`device()` deletion with its vendor bodies. The parent item closes
only after all leaves, the production panic scan, and the provider
feature-gate matrix on hosts with and without the required hardware.

The first leaf is sequenced behind the `COEUS-FALLIBLE-STORAGE`
family: the expects live in infallible signatures
(`Storage::new`/`uninitialized` returning `Self`, the infallible
`StorageMut` contract), so propagating failures changes constructor
signatures and ripples to tensor constructors and operation callers.
That ripple is the fallible-storage migration's scope, not this
item's; this migration starts where those signatures turn fallible.

## Rejected alternatives

- Keeping `device()` and adding more `try_*` siblings preserves the
  panic boundary behind a second name and violates the single
  implementation rule.
- Mapping acquisition or transfer failures to CPU-backed storage
  changes device ownership and hides the original failure.
- Deprecating `device()` instead of deleting it leaves the panic
  reachable indefinitely; the caller closure is small enough (eleven
  files) to migrate completely.

## Consequences

The provider trait loses a method, which is a breaking change and
requires a major SemVer review together with the open residual on
`coeus-hephaestus`. The CUDA vendor body deletion lands in the same
leaf as the other three vendors; that single-file mechanical edit
needs peer coordination with the active `coeus-cuda` lane. Callers
gain explicit error propagation at acquisition and transfer points,
and unavailable-device behavior becomes testable without hardware.

## Verification

Each leaf gate must include format, clippy with warnings denied,
native tests, and doctests. Failure fixtures must assert the typed
error and prove that no output is exposed. Unavailable-device tests
must run on hardware-less hosts (acquisition failure is constructible
there); transfer-failure tests run where a device exists. The final
parent gate includes the complete caller closure, a search proving no
`P::device()` caller and no provider `.expect()` remains, the feature
matrix on hosts with and without each accelerator, and release SemVer
classification.
