# ADR 0079: Float-only mean reduction

Status: Accepted  
Date: 2026-10-04  
Change class: [major] [arch]  
Board item: [COEUS-FLOAT-MEAN-PROVIDER-001](../../backlog.md#coeus-float-mean-provider-001)

## Context

`ReductionOp::Mean` allowed integer tensors to enter a division operation whose
result was truncated. The same tag also forced every provider to expose a mean
kernel for element types for which the operation has no useful arithmetic
meaning. Coeus now consumes Eunomia's float element contract and provider-owned
mean entry points.

## Decision

Remove `ReductionOp::Mean` and expose mean through `ReductionOps::mean`,
`mean`, `mean_axis`, and the corresponding autograd and fused entry points,
all bounded by `FloatElement`. Integer callers convert to a floating element
type before requesting a mean. Provider implementations retain ownership of
the reduction kernel; Coeus does not add a fallback conversion or compatibility
tag.

Mean scaling uses Eunomia's count conversion APIs. Reduced formats must use the
provider's representation-safe accumulation and reciprocal implementation; a
count that overflows the storage format cannot be used as a divisor.

## Alternatives rejected

- Retain the tag and return a truncated integer quotient. This preserves a
  misleading API and silently loses the fractional result.
- Keep the tag but reject integer values at runtime. This leaves invalid
  requests representable and moves a type error to execution.
- Add a Coeus-local conversion or provider adapter. Conversion and accumulation
  contracts belong to the first-party scalar/provider owners and a wrapper
  would duplicate their API.

## Consequences

The change is a public breaking change. Callers migrate from
`ReductionOp::Mean`/generic `reduce` to the float-bound mean methods. The
provider graph must verify forward mean values for reduced formats, including
F16 at counts whose exact reciprocal is representable but whose count is not.
CUDA and other accelerator mean tests require a real device; compile-only and
unavailable-device error-path tests do not establish numerical execution.

## Evidence and overturning conditions

The compile-fail doctests pin the type boundary. Native provider tests must
assert forward values and gradients against a representation-independent oracle.
This decision is revisited if a provider exposes a distinct, documented
integer mean contract or if the scalar contract changes the accumulation and
reciprocal rules.
