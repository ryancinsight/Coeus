# ADR 0084: Floating-point autograd mean contract

Status: Accepted  \
Date: 2026-10-14  \
Change class: [major] [patch]  \
Delivery: Item `COEUS-AUTOGRAD-MEAN-CONTRACT-001`

## Context

At reviewed main `93d24974`, `MeanOp` and `coeus_autograd::mean` accept
`T: Scalar` although provider mean requires `FloatElement`. Executing integer
`[1, 2]` returns 1 with gradient `[0, 0]`. A binary16 one-hot input of length
65536 has sum 1 but its count converts to infinity; forward returns zero
rather than the representable mean 2^-16 (bits `0x0100`).

PR #494 closed unmerged and #523 delivered a different GPU reduction bound.
Rechecking exact main `9fc433b032134d79a52b9294116714737c787d47` after
#554/#562 confirms both autograd entry points now require `FloatElement`.
Keep that upstream contract and add integer compile-fail coverage. Its
forward still divides by `T::from_count(N)`, so the F16 overflow remains.
The companion `Float::MIN_POSITIVE` defect also remains: `0x0040` (2^-18)
is subnormal; the provider's canonical `0x0400` (2^-14) is the first normal.

## Decision

Keep `T: Scalar + FloatElement + leto_ops::Scalar` on both `MeanOp` and public
`mean`, matching provider mean and mean-axis's floating domain. The Leto
bound supports scalar sum's host-staging path. `mse_loss` carries the same
floating requirement.

Form the scale with the locked Eunomia
`FloatElement::from_count_reciprocal(numel)` in forward and backward.
For nonempty input, forward computes `sum(x) * round_T(1/N)` and backward
scales each incoming derivative by the same `round_T(1/N)`. Direct rounding
of the reciprocal avoids overflow of N in binary16. Sum and multiplication
stay in native T; no tensor values are widened. Empty input retains NaN
(`0 * infinity`). Native sum overflow and rounding remain the reduction
contract; this change does not introduce widened accumulation.

Use the provider's named `F16::MIN_POSITIVE` constant. Verify its bits,
value, normal classification and immediately preceding subnormal value.

## Alternatives

- A floating bound alone leaves binary16 count overflow unchanged.
- Dividing by `T::from_count(N)` still overflows at large binary16 counts.
- The locked Leto mean also divides by a converted count.
- Widening tensor values changes the accumulation contract outside scope.

## Migration and package graph

Integer callers choose a floating tensor before calling mean, or integer sum
when that is the intended operation. Generic callers add `FloatElement` and
`leto_ops::Scalar`; supported types are f32, f64, F16 and Bf16. Main already
carries those bounds. The unpublished 0.13.0 candidate groups the public
provider identities and generic-bound changes at one pre-1.0 minor boundary;
no intermediate old-provider compatibility release is created.

The 16 NN convolution/pooling/cross-entropy gradchecks live downstream,
removing the versioned autograd-to-NN dev cycle. The newly incoming cross-entropy
consumer first reproduces an unresolved NN import without that dev edge,
then moves unchanged alongside the original 15 checks. Their bodies, fixtures,
nonuniform weighting and tolerances remain unchanged; only the helper follows
main's FloatElement vocabulary. Every library/internal requirement stays
0.13.0. Earlier exact-main #539 ordinary normalized packaging reproduced
453 errors against stale registry 0.12.0 interfaces; the aligned source at
`1d3e62c336ed47be0ff43bf55babdf439bd361f4` compiled all 15 ordinary archives.
This historical package result does not establish current-head acceptance.

## Main reconciliation and ownership

The isolated source continuation merges exact main `9fc433b0`, including
merged #566, into preserved checkpoint `cf75fc0f`. Keep upstream's scalar
vocabulary, cross-host seams, CPU pooling/unfold delegation, typed einsum
errors, gradient coverage and unrelated board entries. The original checkout
and remote draft #528 remain unchanged. Preserve open sparse #533 and other
owners; no unmerged provider candidate is integrated.

All Git provider records and all 231 registry records equal that main.
The provider graph includes Eunomia 0.10 at `15fbb94c`, Leto 0.46 at
`8b4ab407`, Hephaestus 0.22.1 at `b03e9176` and FFT 0.29.1 at `f266bb0b`.
Only this unpublished ADR moves to 0082; main's accepted scalar-vocabulary
ADR owns 0081. Later main `afb9ba9e` (#570–577) changes only the backlog and
book; integrate both without changing validated executable source or graph.

## Verification

The unchanged analytical cases execute against exact main `9fc433b0` before
repair: four pass, two fail, zero skip. F16 length-65536 mean returns `0x0000`
instead of `0x0100`; minimum positive is `0x0040` instead of `0x0400`.
At source `89908e63`, the full suite passes 1539 native tests with WGPU/Metal
devices required and eight existing skips. All 16 downstream NN checks and
all six focused numerical cases pass. The full workspace passes 172 doctests
with two existing ignores, including both integer compile-fail cases; strict
all-target Clippy using the unchanged CI feature selection, warning-denied
docs, formatting and ADR indexing pass. All-feature locked metadata resolves;
actual all-feature compilation on this Mac stops at the provider's explicit
Linux ROCm requirement. The initial sandbox run cannot provide loopback/GPU
access; the unchanged host-access suite passes after normal approval. No test
or timeout is removed. Final release checks remain pending as stated below.

Cases cover tracked/untracked f32/f64/F16/Bf16 values with seed 2, strided
logical input, empty NaN and every one of 65536 gradients of the exact F16
boundary. Both public mean and direct MeanOp integer dispatch are
compile-fail doctests. The constant tests also retain a Bf16 control.
Preserve all upstream checks, test fixtures, tolerances and the existing
30-second slow / 60-second termination policy.

## Remaining integration and release gate

The unchanged added-line cast guard still lacks a sanctioned conversion for
public `const fn CpuUnaryOp::parameter_pair`: its full binary64 domain must
round to the provider's f32 ABI. Native narrowing is const; the locked
Eunomia runtime factory is not. Keep const and all inputs; do not duplicate
the provider kernel or suppress the guard. Existing repairs use implicit Arc
unsizing and const byte decoding for three other moved-main casts; 408 const
decoder boundary pairs preserve the packed domains.

[Eunomia #182](https://github.com/ryancinsight/eunomia/pull/182) provides the
canonical `pub const F32::from_float(f64)` prerequisite. Its local and hosted
validation is historical, and the version-policy decision remains pending:
the new-public-item rule selects 0.11, while an explicit human exception
would permit 0.10.1. Do not choose that exception, consume an unpublished
factory, merge an automatic publisher, or claim a coherent final release.
The separate Atlas #542 / Leto #365 Python linkage repair is merged and
Python 0.46.1 is published; it does not resolve this version decision.

Remote draft #528 remains open at `806366ca`; no known-blocked push is
retried. After the prerequisite, repeat applicable exact-final-head native,
API, standard-package and hosted release checks. Coeus Miri remains deferred
and unverified; required CUDA/ROCm hardware and the full Python wheel matrix
remain unverified. No coverage, baseline, deadline or review is weakened.
