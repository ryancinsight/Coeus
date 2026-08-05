# ADR 0056: Provider-owned Lp norms

- Status: Accepted
- Date: 2026-08-05
- Scope: `coeus-ops::{norm_p,norm_p_axis}` and tracked autograd nodes
- Tracking: `docs/backlog.md#coeus-autograd-lp-norm-provider-001`

## Context

`norm_p` and `norm_p_axis` accept generic backends but currently copy the
complete input to host memory, evaluate `powf` in host loops, and upload the
result. The tracked autograd nodes repeat that boundary crossing during
backward and retain no provider-resident derivative intermediates. This makes
accelerator execution depend on CPU-addressable storage and creates
input-sized host allocations.

The provider stack already owns the required scalar-power kernels. Leto
provides `PowfOp<T>` through its generic unary traversal. Hephaestus provides
`PowOp` as a scalar-aware elementwise expression and exposes scalar-strided
dispatch for WGPU, CUDA, ROCm, and Metal.

## Decision

Add a scalar-power operation to the Coeus elementwise capability seam. CPU
implementations call the Leto `PowfOp` traversal. Hephaestus-backed
implementations call the selected device provider's scalar-strided `PowOp`.
The operation is distinct from `CpuUnaryOp`: its exponent is a native `T`
runtime value and must not be encoded through a fixed-width parameter tag.

Implement both norm forward paths as provider-resident compositions:

`abs(x) -> powf(p) -> sum(axis) -> powf(1/p)`

The scalar path reduces the flattened provider view; the axis path preserves
the existing reduced-axis shape convention. Backward saves provider tensors
and composes provider `abs`, scalar power, sign, broadcast, multiplication,
division, and additive accumulation. Zero bases are masked before negative
exponents so the existing zero-norm gradient contract does not create
`0 * infinity` NaNs. No host copy, host payload, CPU fallback, or local
`exp(p * log(x))` approximation is permitted.

This adds required methods to public backend/provider traits and is therefore a
SemVer-major architectural change. In-repository implementations migrate in
the same change; external implementors must provide the scalar-power method.

## Consequences

- CPU, CUDA, WGPU, ROCm, and Metal use one generic norm implementation while
  dispatching to their selected provider at the operation boundary.
- Accelerator norm execution no longer allocates an input-sized host buffer.
- The provider composition may allocate intermediate device tensors. This ADR
  makes no runtime or allocation improvement claim without controlled
  measurements.
- Existing p-value, shape, empty-input, zero-input, and gradient semantics are
  preserved; independent analytical and differential tests are required.

## Verification

- Source scans find no input-sized `copy_to_host`, `CpuAddressableStorage`, or
  saved host vectors in `norm_p`, `norm_p_axis`, or their tracked nodes. The
  scalar `norm_p` API reads only its final one-element result; unrelated
  Frobenius helpers remain outside this ADR.
- CPU tests prove Leto dispatch and analytical forward/backward values for
  contiguous and permuted layouts, including zero inputs and non-unit seeds.
- Backend tests cover scalar-power dispatch and Lp value parity for WGPU,
  CUDA, ROCm, and Metal through the repository's hosted backend matrix.
- `cargo semver-checks` classifies the trait-method additions as the declared
  major break.
