# ADR-0038: Complete Hephaestus activation provider parity

## Status

Accepted; the provider-ownership correction is in progress for the
unparameterized f32 scope.

## Context

Coeus and Leto define unparameterized f32 `Mish`, `MishGrad`, `Elu`, and
`EluGrad` operations. Hephaestus already owns native expression markers for
all four accelerator providers. Coeus ROCm and Metal originally rejected all
four operations. Coeus CUDA and WGPU later implemented ELU with consumer-owned
runtime expressions, bypassing the Hephaestus operation markers.

## Decision

Route ROCm and Metal `Mish`, `MishGrad`, `Elu`, and `EluGrad` through the
existing Hephaestus strided elementwise seam. Route CUDA and WGPU ELU forward
and gradient through the corresponding Hephaestus contiguous and strided
operation markers. Delete the superseded consumer-owned CUDA and WGSL
expressions. A CUDA or WGPU ELU request that cannot satisfy the Hephaestus
dispatch contract returns a typed backend error; it does not enter a local
kernel or CPU capability path. The supported accelerator type remains `f32`;
integer providers retain typed unsupported-operation errors.

The contracts are:

- `Mish(x) = x * tanh(log(1 + exp(x)))`.
- `MishGrad(x) = tanh(softplus(x)) + x * (1 - tanh(softplus(x))²) * sigmoid(x)`.
- `Elu(x) = x` for `x >= 0`, otherwise `exp(x) - 1`.
- `EluGrad(x) = 1` for `x >= 0`, otherwise `exp(x)`.

Backend suites compare forward and gradient values with the Leto CPU oracle
over signed inputs containing negative, zero, and positive branch regions.
The existing device-resident Hephaestus paths remain authoritative; the
increment adds no host staging or temporary backend buffer.

## Alternatives rejected

- Leaving ROCm and Metal unsupported was rejected because the provider markers
  and the WGPU/CUDA/CPU contracts already exist.
- Copying the Hephaestus ROCm or Metal expressions into Coeus was rejected
  because dialect-specific provider expressions belong to Hephaestus.
- Routing missing operations through Leto was rejected because it would hide
  provider capability gaps and violate the device-resident output contract.
- Keeping the CUDA and WGPU ELU expression copies was rejected because the
  Hephaestus markers are the operation SSOT and consumer-owned copies can
  diverge.

## Verification

The correction requires format, package compilation, warning-denied Clippy,
contiguous and strided Leto differential contracts, doctests, and exact-head
CUDA/WGPU/ROCm/Metal provider CI. The prior targeted run `30353984154` proved
value parity but did not prove provider ownership because the consumer-local
expressions computed the same values. No physical-device, runtime-performance,
or resident-memory result is inferred.

## Residual scope

Parameterized activations, f64/reduced/vector contracts, and runtime
performance or resident-memory measurements remain separate work. This ADR
does not claim complete Leto parity for non-unary or parameterized operations.
