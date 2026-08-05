# ADR-0053: Use minimal module capability bounds

- Status: Accepted
- Date: 2026-08-04
- Scope: `coeus-nn::Module`, binary Coeus kernels, binary autograd, and
  sinusoidal positional encoding
- Change class: `[patch] [arch]`
- Board item: `COEUS-SINUSOIDAL-PROVIDER-001`

## Context

`Module` and the binary/autograd addition path were bounded by the aggregate
`BackendOps` marker. That marker intentionally composes matmul, convolution,
pooling, and unfold/fold capabilities, but sinusoidal encoding only needs
backend allocation/copy during construction, elementwise addition during
forward, and reduction support for broadcast-gradient accumulation. ROCm and
Metal already implement the latter two capabilities but do not implement the
unrelated aggregate operations, so the aggregate bound rejected a valid
module before provider dispatch.

Sinusoidal construction also wrote the table through
`StorageMut::try_as_mut_slice`, which made CPU addressability a hidden
precondition and panicked for device storage. Forward then downloaded the
whole table and re-uploaded the active prefix on every call.

## Decision

Keep `Module`, `Parameter`, and `Sequential` on the storage-independent
`ComputeBackend` seam. Narrow Coeus binary kernels to `ElementwiseOps` and
their tracked binary autograd path to `ElementwiseOps + ReductionOps`, which
is the exact capability closure for broadcasted binary gradients. Operations
whose implementation still requires another capability, such as `remainder`
through `floor`, retain their existing aggregate bound.

Generate the sinusoidal table with native `T: Float` arithmetic in a single
cold construction buffer, upload it once with `Tensor::from_slice_on`, and
use `Tensor::slice` for the active prefix. The provider consumes that view's
layout directly; no forward path downloads or re-uploads positional state.
Backend selection remains static and monomorphized: `SequentialBackend` and
`MoiraiBackend` use their Leto-backed elementwise/reduction implementations,
while accelerator backends use their native Coeus provider implementations.

## Alternatives rejected

- Adding the missing matmul, pooling, convolution, and unfold/fold
  implementations to ROCm and Metal: rejected because it makes unrelated
  capabilities a prerequisite and would create placeholder or duplicated
  provider work.
- Retaining the CPU-addressable construction and host prefix staging: rejected
  because it panics on device storage and violates provider-resident forward
  execution.
- Adding a consumer-local accelerator adapter: rejected because it would
  duplicate the Coeus operation seam instead of routing through Leto or the
  selected Hephaestus provider.

## Verification

The Coeus NN library check and the CPU positional contract cover native-
precision table values, shape validation, forward values, and prefix storage
sharing. ROCm and Metal add compile-time provider contracts instantiating
`SinusoidalEncoding<f32, _>` through their `ElementwiseOps + ReductionOps`
implementations without requiring aggregate `BackendOps`. Exact-head hosted
provider CI remains the runtime authority for hardware dispatch. These checks
establish capability routing and value semantics, not runtime or memory
improvements; those claims require controlled measurements.

## Revisit trigger

Revisit if a provider introduces a dedicated fused positional-encoding kernel
whose measured benefit justifies a new operation capability. The module and
autograd capability seams remain independent of that optional optimization.
