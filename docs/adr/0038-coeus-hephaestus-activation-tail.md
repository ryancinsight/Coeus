# ADR 0038: Complete Hephaestus activation provider parity

## Status

Accepted and implemented for the f32 activation-tail and runtime-parameterized
scope.

Revision 2026-07-31: the decision now includes runtime-parameterized
Hardtanh and Threshold forward and gradient operations after Hephaestus added
one provider-owned parameterized unary seam across all four accelerators.

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

Hardtanh, HardtanhGrad, Threshold, and ThresholdGrad use the same ownership
boundary. Coeus decodes each operation's two packed `f32` parameters once and
passes them to Hephaestus as runtime kernel arguments. WGPU, CUDA, ROCm, and
Metal write directly into the caller-provided output buffer. Parameter values
are not pipeline-cache dimensions, and unsupported scalar or layout requests
return typed errors instead of entering the removed Coeus WGPU expressions or
another backend.

The contracts are:

- `Mish(x) = x * tanh(log(1 + exp(x)))`.
- `MishGrad(x) = tanh(softplus(x)) + x * (1 - tanh(softplus(x))²) * sigmoid(x)`.
- `Elu(x) = x` for `x >= 0`, otherwise `exp(x) - 1`.
- `EluGrad(x) = 1` for `x >= 0`, otherwise `exp(x)`.
- `Hardtanh(x; min, max) = clamp(x, min, max)`.
- `HardtanhGrad(x; min, max) = 1` for `min < x < max`, otherwise `0`.
- `Threshold(x; threshold, value) = x` for `x > threshold`, otherwise `value`.
- `ThresholdGrad(x; threshold) = 1` for `x > threshold`, otherwise `0`.

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
- Keeping parameterized WGPU source generation was rejected because it decoded
  the two packed `f32` values as `f64` bit patterns and duplicated the provider
  kernel and cache contract.

## Verification

The original activation-tail increment passed exact-head Coeus run
`30387168252`: CUDA job `90369248008`, WGPU job `90369248023`, ROCm job
`90369247910`, and Metal job `90369248013`. Required-device ROCm job
`90369248641` was skipped because no hosted AMD runner was dispatched.

For the parameterized extension, local locked-source format, compilation, and
warning-denied Clippy pass across `coeus-hephaestus`, WGPU, CUDA, ROCm, and
Metal. The CPU parameter-bit contract, live WGPU differential, WGPU alias
regression, shared provider validation, and affected doctests pass. Initial
hosted run `30648108124` exposed a WGPU alias-error routing regression; the
original failing contract and the parameterized differential pass after the
fix. Corrected exact-head run `30649709774` passes WGPU job `91219683142`, CUDA
job `91219683201`, ROCm job `91219683108`, and Metal job `91219683109`.

The CUDA container lane compiles and selects the differential test but may
skip when no NVIDIA device is present. Workflow-dispatch hardware lanes set
`HEPHAESTUS_CUDA_REQUIRE_DEVICE=1` or `HEPHAESTUS_ROCM_REQUIRE_DEVICE=1` and
fail if acquisition is unavailable; both were skipped on the pull request.
No runtime-performance or resident-memory result is inferred.

## Residual scope

f64/reduced/vector contracts and runtime performance or resident-memory
measurements remain separate work. This ADR does not claim complete Leto
parity for non-unary operations.
