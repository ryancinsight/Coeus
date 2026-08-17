# ADR-0023: Add ranked Coeus Hephaestus elementwise providers

## Status

Accepted

Implementation note: implemented in the native ROCm and Metal elementwise increment.

## Decision

Extend the existing `coeus-hephaestus` integration layer with one fixed-rank,
const-generic elementwise provider seam. The consumer dispatches ranks 1
through 4 and converts dynamic Coeus layouts into left-padded Leto layouts.
ROCm and Metal implement the seam by mapping Coeus operation tags to the
native Hephaestus strided kernels. The supported operation set is Add, Sub,
Mul, Div, Sin, Cos, Exp, Log, Neg, Abs, Sqrt, and Recip.

Broadcasting remains an input-layout concern owned by the Hephaestus strided
kernel. Output zero-stride aliasing is rejected at the Coeus boundary, and
unsupported ranks or provider operations return typed errors. The provider
does not copy computation through the host and does not duplicate the
elementwise algorithm per vendor.

`RankTwoOperand` is replaced by the rank-generic `RankedOperand`; all in-repo
callers are migrated in this change. This is a public-name migration for the
integration crate and is intentionally not bridged by a compatibility alias.

## Alternatives rejected

Separate ROCm and Metal operation trees were rejected because they duplicate
dispatch and allow backend semantics to drift. Retaining a `RankTwoOperand`
alias was rejected because it would preserve a second public mental model for
the rank-generic operand and violate the repository's compatibility-shim
policy.

## Verification

The ROCm and Metal tests compare the native binary and unary operations with
the Leto CPU oracle over contiguous and broadcast inputs at ranks 1 through 4.
The backend-parity workflow runs the existing WGPU and CUDA elementwise
contracts together with the new ROCm and Metal contracts. Exact-head run
`30224422963` passed WGPU `89852207720`, CUDA `89852207699`, ROCm
`89852207677`, and Metal `89852207739`; the required-device ROCm lane
`89852208025` was skipped because the workflow was not manually dispatched.
