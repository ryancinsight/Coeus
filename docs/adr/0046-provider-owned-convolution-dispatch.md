# ADR-0046: Route convolution through Leto and Hephaestus

- Status: Accepted
- Date: 2026-07-29
- Board item: `ATLAS-COEUS-DISPATCH-SAFETY-020`
- Change class: `[arch] [major]`

## Context

`ConvOps` exposes six regular and two transposed-convolution operations as
infallible mutations. CPU regular convolution delegates to Leto, but
transposed convolution remains a Coeus-owned host algorithm. CUDA and WGPU
own local device kernels; CUDA additionally downloads unsupported requests,
executes them with `SequentialBackend`, and uploads the result. Backend
selection therefore does not uniquely select the execution provider, and
provider failures cannot cross the public operation seam.

Leto and Hephaestus currently expose no complete regular and transposed
convolution contract. Changing only the Coeus dispatch code would preserve
consumer-owned mathematics under a different module name rather than repair
provider ownership.

## Decision

Leto owns the canonical generic CPU convolution contracts. Hephaestus owns one
generic accelerator convolution contract parameterized by its device API; its
CUDA, WGPU, ROCm, and Metal implementations contain only provider-specific
kernel and submission details. Coeus owns tensor/autograd orchestration and
maps its layouts and buffers directly into the selected provider contract.

The migration proceeds in dependency order:

1. Add shared value-semantic regular and transposed-convolution contracts and
   generic conformance tests to Leto.
2. Add the corresponding Hephaestus accelerator seam, provider
   implementations, and differential tests against Leto.
3. Change all eight Coeus `ConvOps` methods to return
   `Result<(), Self::Error>` and propagate failures through every direct
   workspace consumer.
4. Delete Coeus-owned accelerator convolution kernels, CUDA host fallbacks,
   and the generic host-side transposed-convolution defaults once every
   provider call site has cut over.

Runtime provider selection occurs once at the operation boundary. CPU
selection dispatches to Leto; accelerator selection dispatches to the
corresponding Hephaestus implementation. Unsupported rank, scalar, layout,
parameter, device, compilation, or launch contracts return the selected
backend's typed error. A failed accelerator operation never changes providers.

The provider traits use associated buffer and error types and generic scalar
and structural parameters. Static consumers monomorphize the complete
operation; closed runtime device selection uses enum dispatch at operation
granularity. No per-element dynamic dispatch or compatibility adapter is
introduced.

## Alternatives

- Keep Coeus CUDA/WGPU kernels and only remove the host fallback. Rejected
  because it leaves accelerator mathematics consumer-owned and duplicates the
  provider dimension.
- Add a runtime fallback policy. Rejected because reported backend identity
  would still diverge from execution identity.
- Make only regular convolution fallible. Rejected because the two
  transposed-convolution defaults retain the same hidden provider transition.
- Preserve infallible public methods and record failures out of band.
  Rejected because callers could observe partially written outputs as success.

## Consequences

This is a breaking public-contract correction. Direct callers must handle the
backend result. Provider gaps become compile-time or typed runtime failures
instead of host transfers. Leto and Hephaestus gain the reusable mathematical
ownership required by other consumers; Coeus loses duplicate kernel and
fallback code.

No speed, memory, or binary-size improvement is claimed until matched
measurements compare the complete pre- and post-cutover operation.

## Verification

- Instantiate one generic Leto conformance suite across supported CPU scalars.
- Differentially compare every shipped Hephaestus provider against Leto for
  positive, boundary, invalid-layout, invalid-parameter, and unsupported
  capability cases.
- Assert that a selected accelerator failure returns its typed error without
  host transfer or output mutation.
- Run focused package checks, warning-denied Clippy, Nextest, doctests,
  SemVer checks, and the exact-head Coeus provider matrix.
- Scan Coeus for convolution `SequentialBackend`, `copy_to_host`, fallback,
  and consumer-owned kernel residues before deleting the superseded modules.
