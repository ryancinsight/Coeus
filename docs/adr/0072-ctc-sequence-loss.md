# ADR 0072: Provider-owned CTC sequence loss

Status: Accepted  \
Date: 2026-09-07  \
Change class: [major] [arch]  \
Board item: [COEUS-CTC-SEQUENCE-CONTRACT](../backlog.md#coeus-ctc-sequence-contract)

## Context

At `19e5ac0a`, CTC widens every scalar to `f64`, retains copied log
probabilities and two recurrence tables, and narrows the result. Empty targets
contribute zero loss and no gradient. Forward clamps input lengths while the
backward node retains the originals, so a length beyond the frame dimension
can index beyond its saved state. Existing backward tests check shape and
nonzero entries rather than gradient values.

The prior API requires CPU-addressable storage. Leto owns CPU numerical
operations; its CTC provider lands in PR #177, merge `ba8a879b`. Coeus uses
that recurrence and removes its local implementation.

## Contract

Inputs are log probabilities in `[frames, batch, classes]` order. Targets are
concatenated, with one input length and target length per batch sample. The
blank index and every label must be in range; targets cannot contain blank.
Input lengths cannot exceed the frame dimension. Validate length counts,
checked target totals, layouts, numeric values, and workspace extents before
mutation or allocation driven by those values. Negative infinity denotes zero
probability; NaN, positive values, and positive infinity are invalid active
log probabilities. Padded frames do not contribute.

Let Z be the sum of probabilities of all paths collapsing to a target. The
sample loss is `-ln(Z)`, divided by `max(target_length, 1)` before the batch
mean. An empty target has the all-blank path. With zero frames, the empty
alignment has probability one; a nonempty target has probability zero.
Impossible alignments return infinite forward loss. Their derivative is
undefined and backward returns a typed error before changing the destination.

For independent log-probability inputs, the derivative is negative posterior
occupancy. With one frame and blank probability one-half, an empty target has
loss ln(2) and log-input derivative `[-1, 0]`. Composing with log-softmax gives
`probability - posterior` at logits. This distinguishes the mathematical
log-input derivative from a gradient that already incorporates softmax;
external parity claims must name the boundary being compared.

[Graves et al., section 3.1, equations 2–3 and section 4.2](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
defines path probabilities and the probability/pre-softmax derivatives.

## Decision

Leto owns native-scalar forward and suffix recurrences, validated sequence
metadata, and additive backward evaluation over borrowed views. Its private
state retains forward alpha and suffix beta excluding the current emission,
so posterior evaluation does not require a copied log-probability tensor.
Allocation and arithmetic failures remain typed. Allocation errors retain the
standard library reservation error as their source; the Python boundary
renders that cause into a `MemoryError` without losing the operation context.
Forward and backward use the same binding error conversion. That conversion
lives at the Python crate root because tensor, loss, optimizer, and layer
callers share it; the old neural-network-local module is removed.

Raw scalar log messages lose relative path weights when a large negative log
weight absorbs the logarithm of path multiplicity. The provider retains a
compensated offset and residual in the selected scalar type; recurrence and
posterior normalization use both components. The rounded output loss remains
in that type. Normalization divides by each count rather than multiplying
rounded reciprocals, which can underflow prematurely in reduced precision.
Analytical large-gap and zero-frame cases gate both corrections.

Coeus exposes `CtcOps<T>` with associated provider state. CPU implementations
borrow Leto views over backend storage. Loss and upstream gradients remain in
backend buffers at the seam; the autograd node retains only provider state
and its graph connections. An accelerator implementation is not introduced
without a corresponding provider capability.

The CPU blanket implementation applies to `CpuBackend` and mutable
CPU-addressable storage. It prevents overlapping per-CPU implementations;
CPU specialization therefore belongs in Leto. Other backend families can
implement `CtcOps` independently. Existing `to_leto_view` conversions and
provider layout-error mapping remain the shared conversion boundary.

The shared converter rejects unequal shape and stride counts before padding
or reading either descriptor. Valid singleton strides still normalize to zero,
and leading padding preserves scalar layouts and offsets. This check belongs
at the common conversion boundary because every borrowed provider view relies
on it; a CTC-only check would leave other operations exposed.

## Input boundary

Python arguments and public Rust sequence metadata are untrusted. Invalid
lengths, labels, layout extents, and arithmetic must not cause out-of-bounds
access, unbounded input-driven allocation, or partial gradient writes. The
provider checks metadata and storage, uses checked state extents and fallible
allocation, and preflights every gradient update before mutation. Error
messages identify counts and coordinates without retaining input sequences.
Resource use is proportional to validated frame/target state; this change
does not impose an application-specific maximum sequence length.

## Alternatives

A local special case for empty targets leaves inconsistent lengths and fixed-
precision arithmetic intact. A second recurrence beside the old one creates
competing mathematical owners. Both are rejected. Changing the leaf derivative
to include softmax changes the independent log-input contract; the derivative
is instead documented and tested at both boundaries.

## Migration

`coeus_autograd::ctc_loss` and `coeus_nn::ctc_loss` return
`Result<Var<T, B>, B::Error>`. Callers handle failure with `?` or an explicit
match and require `B: CtcOps<T>`. `CtcLossNode` no longer exposes its old
fixed-precision recurrence fields; construct graphs through `ctc_loss`.
The Python binding retains the same call arguments and converts invalid input
to Python exceptions while releasing the GIL around Rust computation.

## Verification

Independent short-path enumeration checks forward values and posterior
occupancy; seeded analytic cases check log-input and composed logits gradients.
The generic suite covers every supported CPU scalar/backend combination,
empty and repeated targets, mixed batches, padding, impossible paths, and
invalid shapes/lengths/labels. Malformed descriptor tests cover missing and
extra strides for input, loss, upstream, and gradient buffers. Failure tests
preserve every destination lane, including offset padding.
Tolerance bounds follow the recurrence's operation count and scalar epsilon.

Run focused and workspace Clippy, Nextest with committed budgets, doctests,
strict documentation, Python binding tests, and public-surface compatibility
classification against the delivered provider and consumer revisions. The
comparison against `fb0c3150` reports removed public node fields and direct
construction. The fallible return and added capability bound are also breaking
contracts, established by signature comparison; the automated report alone
does not identify those changes. A passing CPU suite establishes no accelerator
execution or performance claim.
