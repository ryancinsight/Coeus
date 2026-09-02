# ADR 0064: Add native unary math providers

## Status

Accepted

Implementation note: implementation is in progress pending the exact-head backend matrix.

## Decision

Route the 19 unparameterized unary math operations already defined by Coeus
and Leto through native Hephaestus strided kernels in the ROCm and Metal
providers:

`Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Log2`, `Log10`, `Exp2`,
`Atanh`, `Asinh`, `Acosh`, `Expm1`, `Log1p`, `Sign`, `Floor`, `Ceil`, `Round`,
and `Trunc`.

The provider dispatch is restricted to f32, using the existing activation-
capable branch. u32 and i32 implementations retain the arithmetic-only
capability boundary and return the typed unsupported-operation error for these
float operations. The operation expressions remain owned by Hephaestus, with
Metal delegating through its WGPU-backed implementation.

Tests use valid input domains per operation and compare native output with the
Leto CPU oracle. The test inputs keep inverse hyperbolic and inverse
trigonometric operations inside their mathematical domains, use positive
values for logarithms, and use values at least one for `acosh`.

## Alternatives rejected

- Keep ROCm and Metal unsupported: rejected because the Coeus/Leto operation
  vocabulary and WGPU/CUDA capability already define the required behavior.
- Copy expressions into the Coeus provider matches: rejected because
  Hephaestus owns dialect syntax and kernel traversal.
- Add a CPU fallback: rejected because it would hide a missing native provider
  capability and break device-resident output semantics.
- Add `erf`, `erfc`, or `lgamma` here: rejected because they do not yet have a
  common native Hephaestus expression across WGPU, CUDA, ROCm, and Metal.

## Verification

The provider unit boundary compiles each new marker through ROCm and Metal.
Integration tests compare every operation with `coeus_leto` across bounded
valid-domain f32 inputs. Exact-head WGPU, CUDA, ROCm, and Metal CI is required
for closure; adapterless provider compilation and physical-device execution
are reported as separate evidence tiers.

## Revisit trigger

Revisit when Coeus requires f64, reduced-precision, vector, `erf`/`erfc`, or
`lgamma` accelerator contracts, or when the Hephaestus unary seam becomes
scalar-parameterized.
