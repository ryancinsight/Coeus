# ADR-0024: Add common Hephaestus activation providers

## Status

Accepted; implementation is complete locally and code-head CI passed. The
documentation-only head requires one final matrix rerun before merge.

## Decision

Extend the Hephaestus operation vocabulary with the common activation forward
and gradient expressions already supported by the Coeus WGPU and CUDA
providers: ReLU, sigmoid, tanh, tanh-approximated GELU, SiLU, and softplus.
ROCm and Metal dispatch these expressions through their native Hephaestus
strided elementwise kernels. The provider implementations are restricted to
`f32`; integer storage keeps the existing typed unsupported-operation result
instead of compiling floating-point formulas for integer shaders.

Hephaestus provides the shared vocabulary at merged master commit `7ac5359`;
Coeus consumes it in commit `28c3cbf`.

The expressions remain dialect-specific constants on the existing
`UnaryExpr<L>` seam. This keeps one operation vocabulary while preserving the
WGSL, CUDA C++, and HIP C++ syntax required by each compiler. The operation
formulas match the existing Coeus CPU/Leto contracts, including the tanh GELU
approximation and its derivative.

## Alternatives rejected

- Copying activation kernels into Coeus ROCm and Metal would fork the shared
  shader-expression and strided traversal logic.
- Routing ROCm or Metal activations through CPU evaluation would hide provider
  capability gaps and violate the device-resident output contract.
- Generalizing the activation formulas to integer scalars would create
  invalid floating-point shader programs; the supported type boundary is
  explicit instead.

## Verification

Hephaestus core expression tests pin representative dialect expressions.
Coeus ROCm and Metal tests compare the native forward and gradient activation
results against the Leto CPU oracle on signed `f32` inputs. The backend-parity
workflow keeps the existing WGPU and CUDA activation contracts in the same
matrix and adds the native ROCm/Metal activation cases to their focused test
filters. Code-head run `30226854005` passed WGPU job `89858362274`, CUDA job
`89858362266`, ROCm job `89858362239`, and Metal job `89858362247`; the
required-device ROCm job `89858362563` skipped without hardware.
