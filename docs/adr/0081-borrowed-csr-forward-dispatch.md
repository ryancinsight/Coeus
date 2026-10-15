<a id="adr-0081"></a>

# ADR 0081: Borrowed CSR forward products retain backend row dispatch

Status: Accepted

Date: 2026-10-15. Owner: Codex `/root`. Classification: [arch].
Target: 0.13.0. Board: `COEUS-SPARSE-BORROWED-RECOVERY`.

## Context

Preservation draft #531 predates fallible allocation, mutable access and CSR
validation. Recovery #533 first removes copies from the existing unsigned
bridge. Tensor forward products additionally require checked signed borrowing,
physical entry order, separate multiplication/addition and backend scheduling.
Leto ADR 0036 specifies those provider contracts. Ryan Clanton explicitly
approves that decision and its downstream unwind-contract clarification in the
task conversation on 2026-10-15. This ADR records that approved consumer slice;
it does not reopen the original preservation draft.

## Decision

Coeus validates CSR storage/structure and dense shape/layout in the existing
order, prepares a checked signed `StoredOrder` provider product with
`ScalarStrategy`, allocates through fallible `Tensor::alloc_on`, obtains the
fallible initialized mutable slice, and binds it exclusively. The adapter
dispatches provider row tasks through the caller's `Backend::parallel_for`.
The provider owns address decoding, arithmetic and complete row writes.
Coeus owns tensor errors, storage, logical offsets and backend selection.

Clarify the unsafe backend obligation on both return and unwind. Success
visits every in-range index exactly once. Panic permits a subset, at most
once per index, but all admitted invocations must complete before propagation.
No retained callback may run afterward. This is an implementor-contract
change despite the unchanged method signature and requires a pre-1.0 minor
bump with a Breaking migration note.

Audit every implementation: Sequential executes inline; Moirai's indexed
scope catches callback panics and drains admitted tasks before failure
propagation; CUDA's enabled/disabled implementations delegate inline to
Sequential. Other accelerator backends implement `ComputeBackend`, not this
CPU-addressable `Backend` trait. Test-only CountingBackend uses the same
inline loop contract. Real Sequential and threshold-crossing
Moirai panic tests verify completion and at-most-once admission.

## Alternatives

- Reinterpreting signed slices fails negative-index and pointer-width proofs.
- Sorting or coalescing changes stored-order floating-point cancellation.
- Unconditional SIMD changes separate-rounding results through FMA.
- Calling a serial provider bypasses the selected backend's parallelism.
- Adding a second executor trait duplicates the existing scheduling boundary.

## Failure modes and verification

Preserve existing typed validation precedence, including width-dependent
index errors, noncontiguous rejection and readable-storage checks. Provider
preparation/binding errors map explicitly into contextual backend errors.
No `expect` handles a recoverable provider error. Output allocation and mutable
access propagate their original typed failures. The adapter does not expose
uninitialized typed references, partial successful tensors or CSR copies.

The co-located sparse design states initialization and numerical proofs.
Acceptance requires native and borrow-model tests, exact scalar-family and
stored-order regressions, strict diagnostics, API/fresh-consumer checks and
paired performance against the recorded main revision, including RHS width
beyond 256. Until these gates finish, #533 remains a draft. Main merges can
trigger publication; no merge or release is authorized by this implementation.
