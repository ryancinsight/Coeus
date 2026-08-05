# ADR-0054: Require provider-owned cumulative scans

- Status: Accepted
- Date: 2026-08-04
- Scope: `coeus-ops::ReductionOps` cumulative sum and product methods and all
  current Coeus backend implementations
- Change class: `[major] [arch]`
- Board item: `COEUS-SCAN-DISPATCH-001`

## Context

`ReductionOps` exposed default implementations for `cumsum`, `suffix_sum`,
`cumprod`, and `suffix_prod`. Each default allocated host vectors, copied the
selected backend buffer to host memory, executed a Leto operation, and copied
the result back. The current CPU implementations already route through Leto,
and the current CUDA, WGPU, ROCm, and Metal implementations already provide
native provider dispatch. The defaults therefore leave a reachable generic
host-staging path in the public capability seam even though every shipped
backend has a provider-owned implementation.

## Decision

Make the four cumulative scan methods required operations of
`coeus_ops::ReductionOps`. Delete the generic host-staging helpers. Keep the
single generic operation surface and migrate all in-repository implementors:
Sequential and Moirai continue to use Leto-backed CPU kernels, while CUDA,
WGPU, ROCm, and Metal retain their Hephaestus provider dispatch. Backend
selection remains a type parameter, so each implementation is statically
resolved and monomorphized at the operation boundary.

The CPU-only argmax, argmin, and top-k defaults remain separate. Their trait
bounds already require `CpuBackend`, so they are not available to accelerator
implementations and do not provide an accelerator fallback.

## Alternatives rejected

- Keep the defaults and rely on every shipped backend to override them:
  rejected because a future backend can silently inherit host staging and the
  public seam does not enforce provider ownership.
- Add `CpuBackend` bounds to the cumulative defaults: rejected because the
  generic cumulative operation surface is used for accelerator backends and
  would need a second dispatch seam or backend-specific bounds; the required
  methods express the actual capability directly.
- Add a Coeus-local adapter that downloads unsupported buffers: rejected
  because it recreates the consumer-owned fallback the provider layer is
  intended to remove.

## Verification

The implementation will compile every current `ReductionOps` provider and run
the CPU cumulative value contracts. Static residue scans will confirm that the
Coes cumulative scan path contains no host transfer. Accelerator package and
provider CI remain the runtime authority for Hephaestus dispatch; the local
Atlas overlay is not a valid substitute when its peer Hephaestus checkout is
dirty or does not contain the locked provider APIs. These checks establish
dispatch ownership and value semantics, not runtime or memory improvements.

## Revisit trigger

Revisit if a new backend needs a shared scan planning capability that does not
execute a provider-owned kernel. Such a plan belongs in the upstream provider
contract; it must not restore a Coeus host-staging default.
