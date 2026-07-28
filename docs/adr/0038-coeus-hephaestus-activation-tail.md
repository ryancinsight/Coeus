# ADR-0038: Complete Hephaestus activation provider parity

## Status

Accepted; implementation and exact-head provider CI are complete for the
unparameterized f32 scope.

## Context

Coeus and Leto define unparameterized f32 `Mish`, `MishGrad`, `Elu`, and
`EluGrad` operations. Hephaestus already owns native expression markers for
all four ROCm and Metal operations, and the WGPU expression generator already
emits their WGSL forms. Coeus ROCm and Metal dispatch rejected all four
operations. Coeus CUDA implemented Mish in both launch forms but rejected ELU.

## Decision

Route ROCm and Metal `Mish`, `MishGrad`, `Elu`, and `EluGrad` through the
existing Hephaestus strided elementwise seam. Add the ELU forward and gradient
expressions to both CUDA contiguous and strided launch tables. Keep WGPU on its
existing provider expression path. The supported accelerator type remains
`f32`; integer providers retain typed unsupported-operation errors.

The contracts are:

- `Mish(x) = x * tanh(log(1 + exp(x)))`.
- `MishGrad(x) = tanh(softplus(x)) + x * (1 - tanh(softplus(x))²) * sigmoid(x)`.
- `Elu(x) = x` for `x >= 0`, otherwise `exp(x) - 1`.
- `EluGrad(x) = 1` for `x >= 0`, otherwise `exp(x)`.

Backend suites compare forward and gradient values with the Leto CPU oracle
over signed inputs containing negative, zero, and positive branch regions.
The existing device-resident strided and launch paths remain authoritative;
the increment adds no host staging or temporary backend buffer.

## Alternatives rejected

- Leaving ROCm and Metal unsupported was rejected because the provider markers
  and the WGPU/CUDA/CPU contracts already exist.
- Copying the Hephaestus ROCm or Metal expressions into Coeus was rejected
  because dialect-specific provider expressions belong to Hephaestus.
- Routing missing operations through Leto was rejected because it would hide
  provider capability gaps and violate the device-resident output contract.
- Adding a second CUDA kernel family was rejected because the existing
  contiguous and strided launch tables are the canonical CUDA dispatch homes.

## Verification

Local format, package compilation, lint, and focused provider contracts cover
the changed dispatch and expression tables. Exact-head Coeus run `30351530489`
passed CUDA job `90249902958`, WGPU job `90249903429`, ROCm job `90249902939`,
and Metal job `90249903016`. Required-device ROCm job `90249904831` was
skipped because no hosted AMD runner was dispatched; no physical-device result
is inferred. The external `recurseml/analysis` status returned its recurring
analyzer error and is not repository-owned verification.

## Residual scope

Parameterized activations, f64/reduced/vector contracts, and runtime
performance or resident-memory measurements remain separate work. This ADR
does not claim complete Leto parity for non-unary or parameterized operations.
