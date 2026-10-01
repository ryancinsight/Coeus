# ADR 0078: Fallible tensor storage operations

Status: Accepted  \
Date: 2026-09-29  \
Change class: [major] [arch]  \
Delivery: [PR #461](https://github.com/ryancinsight/Coeus/pull/461)
Revision 2026-09-29: uninitialized outputs also require unsafe full-write
contracts and validated raw-pool geometry; audits found safe providers and
pool layouts did not establish those invariants.

## Context

Before this decision, `Tensor::alloc_on` and `Tensor::zeros_on` returned `Self`
while `ComputeBackend::allocate`, `fill`, and device copies were infallible.
The backend implementations already owned the operations that could reject a
size, layout, device, or transfer. The public constructors therefore could not
report those failures and callers either panicked later or required an
infallible fallback.

The constructor surface is shared by tensor, operation, autograd, neural
network, optimizer, distributed, and Python crates. A direct workspace search
found 2,781 constructor candidates in 406 files. The migration must preserve
the existing copy-on-write ownership model and must not add parallel `try_*`
constructors, because two names would create separate sources of truth.

## Decision

Make storage operations fallible at the `ComputeBackend` seam and replace the
existing constructor signatures in place:

```text
allocate       -> Result<DeviceBuffer<T>, Error>
allocate_zeroed -> Result<DeviceBuffer<T>, Error>
fill           -> Result<(), Error>
fill_zero      -> Result<(), Error>
copy_to_device -> Result<(), Error>
copy_to_host   -> Result<(), Error>
HephaestusStorage::new -> Result<HephaestusStorage<P, T>, HephaestusError>
Tensor::alloc_on / zeros_on / ones_on / full_on / from_slice_on
               -> Result<Tensor<T, B>, B::Error>
```

The default `allocate_zeroed` and `fill_zero` implementations propagate the
underlying operation result. Backend-specific implementations retain native
zero-fill and transfer paths. Provider-backed allocation and transfers acquire
the device through the provider's typed fallible seam. No backend silently
falls back to CPU storage.
Shape validation remains before allocation, and a failed operation does not
publish a partially constructed tensor. Pure view methods remain infallible
because they only rewrite layout metadata. Materialization, transfers, and
mutable access that can detach shared storage return the backend error.
`StorageMut::make_unique`, `try_as_mut_slice`, and
`CpuAddressableStorageMut::as_mut_slice` expose allocation and copy failures;
`Tensor::storage_mut`, `storage_mut_and_layout`, `as_mut_slice`, `try_iter_mut`,
`set`, and `set_flat` propagate them. Provider transfer methods reject unequal
buffer lengths before mutation. Provider-backed storage preserves the selected
provider's error type while retaining the underlying device failure.
`ComputeBackend` is an unsafe trait because implementors must uphold safe
storage access, and its uninitialized `allocate` method is unsafe. The
`Tensor::alloc_on` and `HephaestusStorage::from_buffer` constructors are also
unsafe; callers prove every element is initialized before a read or safe
exposure. CPU storage tracks initialization per handle, refuses immutable
reads while uninitialized, and initializes before forming safe mutable typed
slices. Copy-on-write detachment copies bytes as `MaybeUninit<T>` so it never
forms a typed reference to uninitialized memory. Raw Mnemosyne allocation
failure uses a fixed-data error rather than allocating a diagnostic string.
`Backend` is unsafe to implement because `parallel_for` must invoke every
index exactly once and join all calls before returning. The overwrite
operation traits (`ElementwiseOps`, `ScalarPowerOps`, `MatmulOps`,
`ReductionOps`, `PoolOps`, `UnfoldFoldOps`, `AttentionOps`, and
`CrossEntropyOps`) and the Hephaestus provider markers are unsafe because a
successful overwrite must initialize every logical output; accumulation
methods require initialized destinations. CPU pooling validates output
geometry, storage spans, and injective write layouts before raw-pointer
access.
The separate `ATLAS-COEUS-SAFETY-001` item covers remaining provider-operation
dispatch paths that still use infallible device acquisition. `CpuStorage::new`,
`filled`, `from_slice`, and copy-on-write mutation return typed allocation
failures.

The migration is split into dependency ordered leaves: the core trait and
CPU/provider implementations, tensor constructors and their direct operation
callers, then autograd/NN/optimizer/distributed/Python callers. Each leaf
changes the existing names in place and carries value-semantic success and
failure tests. The parent item closes only after all leaves and the full
consumer closure pass.

## Rejected alternatives

- Keeping infallible constructors and adding `try_*` siblings preserves the
  panic boundary and violates the single implementation rule.
- Mapping allocation or transfer failures to zeroed CPU tensors changes device
  ownership and hides the original failure.
- Returning a partially initialized tensor after a failed fill exposes
  uninitialized or stale elements as if construction succeeded.

## Consequences

The public constructor, mutable storage accessors, backend traits, and
overwrite-operation/provider traits are breaking changes and require a major
SemVer review. The migration increases explicit error propagation at callers,
but keeps allocation ownership in the provider and removes the need for host
staging or recovery copies. The core `Result` type remains the backend's
existing typed error, so provider-specific failures stay visible.

## Verification

The leaf gates must include format, clippy with warnings denied, native tests,
doctests, and the backend matrix applicable to each changed provider. Failure
fixtures must assert the returned error and prove that no output is exposed.
The final parent gate includes the complete caller closure, release SemVer
classification, and a search proving no migrated constructor discards a
provider result.
