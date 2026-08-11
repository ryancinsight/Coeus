# ADR-0060: Provider-owned Metal and ROCm bridge

Status: Accepted
Date: 2026-08-11
Board item: `COEUS-HEPHAESTUS-METAL-ROCM-001`

## Context

The Metal and ROCm Coeus crates duplicated the same consumer-owned runtime,
elementwise, reduction, random-initialization, rotate-half, stateful-update,
and cross-entropy routing around provider-specific Hephaestus calls. The
duplication kept vendor operation ownership in Coeus and left the generic
Hephaestus operation seams unused for ordinary elementwise and rank-two
reduction dispatch.

## Decision

Use `HephaestusBackend<P>` as the only accelerator backend implementation in
the Metal and ROCm crates. Each provider declares its Hephaestus operation
bundles in `backend/provider.rs`; generic Coeus dispatch owns ranked
elementwise, scalar-power, axis-reduction, scan, random, rotate-half,
stateful-update, and cross-entropy orchestration. Hephaestus extends the axis
reduction seam with default min/max dispatch and exposes the ROCm elementwise,
axis-reduction, and scan seams in no-feature builds so unavailable devices
remain typed failures rather than feature-gated placeholder paths.

The old `MetalBackend` and `RocmBackend` consumer-owned surfaces are removed.
In-repository callers use `HephaestusBackend<MetalProvider>` or
`HephaestusBackend<RocmProvider>` directly; no public alias or forwarding
adapter preserves the removed names.

## Alternatives rejected

- Keep the vendor operation modules and delegate from them: rejected because
  the consumer would still own duplicated routing and future operation seams
  would drift.
- Keep public backend aliases: rejected because they preserve the removed
  consumer-owned boundary and violate the replacement's no-compatibility-soup
  contract.
- Fall back to CPU or host staging when a provider seam is unavailable:
  rejected because provider failure must remain typed and visible.

## Invariants

- Metal and ROCm kernels remain provider-owned and device-resident.
- Operation selection is static and monomorphized; no trait objects or
  per-element vendor dispatch are introduced.
- Unsupported unary operations return a typed dispatch error.
- Coeus scalar and layout contracts remain unchanged; min/max reduction
  identity and dialect-token requirements are enforced by the Hephaestus
  seam.
- No runtime or memory improvement is claimed without a controlled baseline.

## Verification

- Coeus format, all-targets package check, and warning-denied Clippy for
  `coeus-hephaestus`, `coeus-metal`, and `coeus-rocm`.
- Hephaestus warning-denied Clippy for the changed core and ROCm crates plus
  a no-feature ROCm check.
- Coeus focused native nextest and doctests, with physical-device availability
  reported separately from code failures.
- Residue scans confirm that the deleted Metal/ROCm operation modules and
  removed public backend names have no remaining in-repository callers.
- Hosted provider contracts remain required for exact-head merge evidence;
  local Atlas overlay lock rewriting is reported separately when it prevents
  a locked standalone invocation.
