# ADR 0001 — Dimension-complete interpolation

- Status: Accepted
- Date: 2026-07-11
- Change class: [major]
- Driver: RITK ADR 0004 / MIG-526

## Context

Coeus exposed separate three-dimensional forward and reverse-mode entry
points. RITK's trainable displacement field supports both two and three
dimensions, so consuming those functions would either narrow the field
contract or require a second consumer-owned implementation.

## Decision

One `linear_interpolation<const D, B, P>` operation family owns forward and
reverse mode for `D = 2` and `D = 3`. Sealed `Dimension<D>` implementations
make every other dimension a compile-time error. A sealed `BoundaryPolicy`
selects border semantics at compile time; `Replicate` is the initial
zero-sized policy. Const arrays hold neighbours, indices, and weights so the
per-point corner traversal does not allocate. The prior dimension-specific
functions and gradient type are removed in the same change.

This is a breaking public change and advances the workspace from 0.5.11 to
0.6.0. A compatibility re-export is rejected because it would retain two
names for one operation family and allow new consumers to keep binding to the
incomplete dimensional contract.

## Verification

Exact analytical forward values and image/grid derivatives cover both
dimensions. Independent central differences cover every coordinate axis with
the bound `16 * f32::EPSILON / step`: multilinearity removes truncation error
for a one-axis central difference, while the factor covers rounded arithmetic
in the four contributing evaluations. Typed tests reject unsupported
dimensions and malformed shapes. The same generic suite instantiates
Sequential and Moirai backends, and autograd tests verify both dimensions.
NaN and infinite coordinates are rejected with a typed error before arithmetic.

## Revisit trigger

Add another sealed policy only when a consumer requires different documented
boundary semantics. Extend the same operation family; do not add a policy- or
dimension-named algorithm.
