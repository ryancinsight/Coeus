# ADR 0001: Dimension-complete interpolation

- Status: Accepted
- Date: 2026-07-11
- Change class: [major]
- Driver: COEUS-AUTOGRAD-LINEAR-INTERPOLATION-GENERIC-SCALAR; PR #451

## Context

Coeus exposes dimension-generic interpolation for 2-D and 3-D tensors. The
operation was restricted to `f32`, despite the scalar contract supporting
other floating-point representations. Boundary neighbour selection also
converted the extent and adjacent indices through the coordinate scalar,
which can merge distinct indices or round the last valid index out of range
for low-precision scalars.

## Decision

One `linear_interpolation<const D, B, P, T>` operation family owns forward and
reverse mode for `D = 2` and `D = 3`, with `T: Float` shared by image values,
coordinates, weights, outputs, and gradients. The public scalar parameter is
appended after the existing generic parameters. Public gradient and autograd
node types likewise append `T` with an `f32` default, preserving their
existing `f32` type annotations. Explicit function turbofish calls add a
fourth argument; no forwarding overload is retained.

`Replicate` selects neighbouring indices in the integer address space. It
converts the floored finite nonnegative coordinate once, clamps against the
original `usize` extent, and constructs the upper neighbour with integer
arithmetic. This keeps adjacent indices distinct when the scalar cannot
represent them and prevents a rounded extent from admitting an invalid index.
Negative coordinates resolve to the replicated lower border.

This public generic change is [major]. The external migration is documented in
[`interpolation-migration.md`](../book/interpolation-migration.md). The
workspace version changes only through an authorized release.

## Verification

Analytical forward and reverse-mode cases instantiate `f32`, `f64`, `F16`,
and `Bf16` on Sequential and Moirai backends. Dedicated BF16 cases exercise
adjacent indices at 256 and the 260-element extent boundary. Coordinate
gradients use central differences in 2-D and 3-D for `f32` and `f64`; autograd
gradcheck covers image and coordinate inputs separately away from integer
cell boundaries. The analytical coordinate-gradient fixtures use integer
image values, dyadic coordinates, and a `1/8` step that stays within one cell;
all products and central differences are exactly representable in `f32` and
`f64`, so the derivatives compare exactly.

## Revisit trigger

Add another sealed boundary policy only when a consumer requires different
documented semantics. Extend this operation family; do not add a
policy- or dimension-named algorithm.

- Revision (2026-09-28, PR #451): generalize the scalar contract and move
  neighbour indexing to integer space after BF16 boundary tests exposed
  precision loss.
