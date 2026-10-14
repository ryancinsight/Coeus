# ADR 0079: Floating-point autograd mean contract

Status: Accepted  \
Date: 2026-10-14  \
Change class: [major] [patch]  \
Delivery: Item `COEUS-AUTOGRAD-MEAN-CONTRACT-001`

## Context

At main `93d24974`, `MeanOp` and `coeus_autograd::mean` accept `T: Scalar`,
although provider mean requires `FloatElement`. Integer `[1, 2]` therefore
returns 1 with gradient `[0, 0]`. A binary16 one-hot input of length 65536
has sum 1 but its count converts to infinity, so forward returns zero rather
than the representable mean 2^-16 (bits 0x0100).

PR #494 closed unmerged. PR #523 delivered the GPU generic-reduction
restriction; the current autograd signatures prove that whole-tensor mean
still needs its floating-point contract.

## Decision

Require `T: Scalar + FloatElement` for both the `MeanOp` implementation and
the public `mean` wrapper, matching `mean_axis` and provider mean. Keep sum
dispatch and multiplication in `T`. Form the scale with the locked Eunomia
`FloatElement::from_count_reciprocal(numel)` in forward and backward. This
rounds the integer count's reciprocal directly into the destination format
and avoids count overflow; no tensor values are widened.

For nonempty input, forward computes `sum(x) * round_T(1/N)`, and backward
scales each incoming derivative by `round_T(1/N)`. Empty input retains NaN
(`0 * infinity`). Existing native-precision sum overflow and rounding remain
the reduction contract; this change does not promise widened accumulation.

## Alternatives

- A type bound alone leaves binary16 count overflow unchanged.
- Dividing by `T::from_count(N)` still overflows at large binary16 counts.
- Forwarding to the locked Leto mean inherits its `acc / from_count(N)` path.
- Widening tensor values adds a different accumulation policy outside scope.

## Migration

Integer callers must explicitly choose a floating-point tensor before calling
mean, or use integer sum if that is the desired operation. Generic callers
add `FloatElement` to their element bound. This is a breaking public bound;
the pre-1.0 workspace version advances to 0.13.0, with no release or publish.

## Verification

Native regression tests cover f32, f64, F16, and Bf16 forward/backward with
seed 2, untracked values, strided input, empty input, and all 65536 gradients
of the exact binary16 boundary case. Compile-fail doctests reject integers
through both `mean` and direct `MeanOp` dispatch. Required gates remain
pending until checked on the final PR head. Miri is deferred and unverified.
