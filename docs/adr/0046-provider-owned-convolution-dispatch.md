# ADR-0046: Route convolution through Leto and Hephaestus

- Status: Accepted
- Date: 2026-07-29
- Board item: `ATLAS-COEUS-DISPATCH-SAFETY-020`
- Change class: `[arch] [major]`
- Revision 2026-07-30: expanded the closure after repository-wide caller and
  implementor audits found the separate 3-D transposed-forward trait and three
  consumer-owned transposed-backward implementations.
- Revision 2026-07-30: completed the provider cutover. One fallible
  const-generic `ConvOps` contract now owns regular and transposed forward and
  backward dispatch for spatial ranks one through three.
- Revision 2026-07-30: consolidated review findings. One generic Hephaestus
  dispatch implementation now owns request mapping for every accelerator, and
  one Leto view module owns borrowed operand conversion for regular and
  transposed operations.
- Revision 2026-07-30: unified regular/transposed forward setup and regular/
  transposed autograd accumulation after the final review, removing the last
  duplicated convolution dispatch and panic-prone gradient indexing.

## Context

Before this decision, `ConvOps` exposed infallible rank-specific mutations,
`ConvTranspose3dOps` carried a separate forward capability, CPU mathematics
lived in Coeus, CUDA and WGPU owned local kernels, CUDA downloaded unsupported
requests for CPU execution, and transposed autograd owned three host-side
backward loops. Backend identity therefore did not determine execution
provider, and provider failures could not cross the public operation seam.

Leto and Hephaestus lacked complete regular and transposed convolution
contracts. A Coeus-only dispatch rename would have retained consumer-owned
mathematics and hidden provider failure.

## Decision

Leto owns the canonical generic CPU convolution contracts. Hephaestus owns one
generic accelerator convolution contract parameterized by its device API; its
CUDA, WGPU, ROCm, and Metal implementations contain only provider-specific
kernel and submission details. Coeus owns tensor/autograd orchestration and
maps its layouts and buffers directly into the selected provider contract.
Within Coeus, one generic `ConvolutionBackend` binding supplies device, buffer,
operation, and typed-error associations to one monomorphized Hephaestus
dispatch core. Vendor backend modules contain no request-layout or bias
construction logic. Leto regular and transposed dispatch share one borrowed
operand-view module.

The migration is implemented in dependency order:

1. Add shared value-semantic regular and transposed-convolution forward and
   backward contracts plus generic conformance tests to Leto.
2. Add the corresponding Hephaestus accelerator seam, provider
   implementations, and differential tests against Leto.
3. Replace the rank-specific capability split with four fallible const-generic
   `ConvOps` methods: regular forward/backward and transposed
   forward/backward. Rank-specific methods are zero-cost default adapters over
   that SSOT and propagate `Self::Error`.
4. Delete Coeus-owned accelerator convolution kernels, CUDA host fallbacks,
   generic host-side transposed-convolution defaults, and autograd
   transposed-backward loops once every provider call site has cut over.

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
- Make only regular convolution fallible. Rejected because the transposed
  forward defaults and host-side backward loops retain the same hidden
  provider transition.
- Preserve infallible public methods and record failures out of band.
  Rejected because callers could observe partially written outputs as success.

## Consequences

This is a breaking public-contract correction. Direct callers must handle the
backend result. Provider gaps become compile-time or typed runtime failures
instead of host transfers. Leto and Hephaestus gain the reusable mathematical
ownership required by other consumers; Coeus loses duplicate kernel and
fallback code.

The temporary `ConvTranspose3dOps` capability split is superseded and removed
because all shipped accelerator providers now expose the complete rank-generic
contract.

No speed, memory, or binary-size improvement is claimed until matched
measurements compare the complete pre- and post-cutover operation.

## Verification

- Leto and Hephaestus own generic conformance and provider differential suites.
- Coeus CPU Nextest covers regular/transposed ranks one through three and exact
  transposed gradients.
- Coeus WGPU Nextest executes the selected Hephaestus provider on-device and
  compares regular/transposed forward/backward values with Leto CPU results.
- Warning-denied all-target Clippy passes for the consolidated Leto,
  Hephaestus, WGPU, CUDA, and operation-contract scope. Final-review
  Leto/Hephaestus/autograd/WGPU Nextest passes 214/214. All 46 executable
  affected-package doctests pass; two pre-existing NN doctests remain ignored.
- `cargo-semver-checks` classifies the fallible contract and removed capability
  seam as a major change. Exact-head provider run `30545333101` passed WGPU
  job `90880014492`, CUDA job `90880014608`, ROCm job `90880014606`, and
  Metal job `90880014508`; required-device ROCm job `90880015294` was skipped
  because no AMD hardware runner was dispatched.
- PR #250 merged the verified provider cutover as `0dfab53e`.
- Residue scans reject convolution `SequentialBackend`, `copy_to_host`,
  fallback, and consumer-owned kernel paths.
