# ADR-0051: Route rotate-half through Leto and Hephaestus

- Status: Accepted
- Date: 2026-08-03
- Scope: Coeus rotary embedding, tracked rotate-half execution, CPU Leto
  dispatch, and WGPU/CUDA/ROCm/Metal Hephaestus dispatch
- Change class: `[major] [arch]`
- Board item: `COEUS-ROPE-PROVIDER-001`
- Revision: 2026-08-03 — extend the provider-safe assignment decision to the
  remaining generic unary assignment and structural-gradient accumulation
  sites after an audit found four raw-reference aliases. This additive backend
  capability revision is `[minor] [arch]`.

## Context

Rotary embedding currently constructs `rotate_half([x1, x2]) = [-x2, x1]`
through tracked `split` and `cat`. Both operations require CPU-addressable
storage and materialize values through the CPU Leto adapter. The resulting
trait bound prevents device-resident RoPE execution even though negation and
strided output writes already exist on every accelerator provider.

The operation does not require arbitrary gather or general concatenation.
Both input halves and both output halves are affine layout slices. Leto owns
CPU strided mapping; Hephaestus owns accelerator `NegOp`, `IdentityOp`,
uninitialized allocation, and fixed-rank strided output dispatch.

## Decision

Add one interface-segregated `RotateHalfOps<T>` capability at the Coeus
operation boundary. The method consumes an input storage/layout pair and
returns one fully initialized output buffer with the same contiguous logical
shape. Selection occurs once per operation through the backend type.

Sequential and Moirai implementations allocate one final CPU buffer and use
Leto destination-writing maps: negate the second input-half view into the
first output-half view, then copy the first input-half view into the second
output-half view. WGPU, CUDA, ROCm, and Metal implementations call one generic
`coeus-hephaestus` bridge. That bridge allocates one provider buffer and issues
`NegOp` and `IdentityOp` into disjoint output-half layouts through the selected
provider's monomorphized `ElementwiseOps` implementation.

Add a tracked autograd rotate-half operation. Its backward applies the exact
transpose `R^T = -R` through the same provider-selected primitive. Rotary
embedding calls this tracked operation directly and removes the
CPU-addressable storage bounds. Provider failures propagate through the typed
module error; no backend may download to the host or fall back to another
provider.

Autograd accumulation calls the `ElementwiseOps` assignment capability rather
than fabricating an immutable reference from its mutable destination. CPU
overrides the capability with one borrowed Leto zip traversal, preserving the
allocation when storage is uniquely owned and detaching it through COW when
shared. Accelerator backends use the default distinct device output and
replace the destination with a compact contiguous result after dispatch because
WGPU prohibits one buffer from being bound read-only and read-write in the same
usage scope. The replacement path reads shared source storage without eagerly
detaching it, avoiding a redundant full-device COW copy and ensuring every slot
in the installed allocation is initialized.

Unary assignment follows the same whole-tensor replacement rule. CPU backends
route the destination write through Leto; accelerator backends route it through
Hephaestus. Both write a distinct compact result and replace the destination.
Structural-gradient
accumulation has a different contract because only a sublayout changes: the
default snapshots the source handle, detaches a candidate through device-local
COW, dispatches into the selected sublayout, and installs the candidate only on
success, preserving every untouched parent element. CPU backends override this
with direct Leto mutation.
No implementation may create simultaneous shared and mutable Rust references
to one device-buffer object.

## Alternatives rejected

- General device concatenation or gather: rejected because rotate-half is two
  affine slices and introducing a wider capability has no present requirement.
- Keep tracked `split`/`cat`: rejected because their CPU storage contracts make
  backend selection ineffective for RoPE.
- Add a fused Coeus-local accelerator kernel: rejected because Hephaestus
  already owns the required provider operations and strided dispatch.
- Fall back to Leto after an accelerator error: rejected because it hides a
  provider fault and silently changes device placement.
- Retain the overlapping-reference assignment path: rejected because its
  safety claim depended on backend behavior and WGPU rejects the underlying
  simultaneous read/write binding.

## Verification

Generic Sequential and Moirai tests compare forward and backward values with
the analytical rotation matrix. WGPU and CUDA contracts compare provider output
and gradients with the same oracle; Metal and ROCm contracts compare provider
output. All contracts exercise multiple rows and both feature halves. Residue
scans reject CPU-addressable bounds and host transfers in the migrated closure.

Warning-denied Clippy, focused Nextest, doctests, semver classification,
independent architecture review, and exact-head provider CI gate delivery.
These checks establish dispatch and value semantics, not runtime or memory
improvements; those claims require controlled measurements.

## Revisit trigger

Revisit when a provider offers a single fused rotary kernel whose measured
benefit exceeds the two-dispatch implementation. The `RotateHalfOps` seam
remains stable because provider selection and autograd semantics do not change.
