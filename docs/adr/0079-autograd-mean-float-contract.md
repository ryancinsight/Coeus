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

Require `T: Scalar + FloatElement + leto_ops::Scalar` for both the `MeanOp`
implementation and the public `mean` wrapper. The floating-point restriction
matches `mean_axis` and provider mean; the Leto operation bound supports the
host-staging path of scalar sum introduced by PR #529. `mse_loss` carries
`Float + FloatElement + leto_ops::Scalar`. Keep sum dispatch and multiplication
in `T`. Form the scale with the locked Eunomia
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
add `FloatElement` and `leto_ops::Scalar` to their element bound. Both traits
are implemented for the supported f32, f64, F16 and Bf16 types. These changes
and the staging-related public bounds ship together in pre-1.0 version 0.13.0;
the intermediate 0.12.1/0.12.2 repair snapshots are not published separately.
A patch release would not cover these breaking generic-bound changes.

## Verification

Native regression tests cover f32, f64, F16, and Bf16 forward/backward with
seed 2, untracked values, strided input, empty input, and all 65536 gradients
of the exact binary16 boundary case. Compile-fail doctests reject integers
through both `mean` and direct `MeanOp` dispatch. All applicable required and release checks must pass on the final PR head
before integration and publication. Miri is deferred and unverified.

NN-layer convolution and pooling gradchecks belong to the downstream NN test
package. Keeping those 15 unchanged dual-scalar cases there removes the
versioned autograd-to-NN development edge, so autograd can package before NN.
Each package retains its own executable test fixtures; no cross-package test
paths or stripped dependencies are used to hide the cycle. The combined test
inventory remains 688 cases, including the same eight existing skips.

## Current provider cohort integration

The correctly versioned current provider contract is Eunomia 0.10, published
from a231e54d; Aequitas 0.3 and Mnemosyne-memory 0.10 have verified registry
receipts. The coordinated graph also requires Hermes 0.9, Leto/Leto-ops 0.46,
Hephaestus 0.22 and Apollo FFT 0.29. Coeus 0.13 remains unpublished, so this
same minor boundary covers its changed public provider identities and generic
bounds. Do not issue an intermediate old-provider compatibility release.

PR #532's normally merged default is integrated without file changes because
its complete implementation already belongs to #528. Separate borrowed-CSR
PR #533 remains with its owner. Preparing dependency requirements preserves
every numerical implementation and regression; the old precise lock stays
until all intended provider defaults resolve. Rerun native, lint, docs, all
15 standard packages, blocking API and exact-head required CI after that lock
transition. Metadata/source-only evidence does not establish final acceptance.

Moirai #623 separately owns compatible runtime 0.9.3 for its private Mnemosyne
0.10 edge. Require 0.9.3 after its reviewed publication; no older runtime graph
or duplicate implementation supplies acceptance. Preserve #622 and Metis #504.
