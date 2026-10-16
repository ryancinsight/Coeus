# ADR 0081: Delete `Scalar::from_f64` in favor of the eunomia element vocabulary

Status: Accepted  \
Date: 2026-10-15  \
Change class: [arch] [major]  \
Board item: [COEUS-SCALAR-VOCAB](../../backlog.md#coeus-scalar-vocab)

## Context

`coeus_core::Scalar` redeclared `from_f64`, duplicating conversion
spelling the eunomia element vocabulary already owns (`zero`/`one`/
`to_f64` had already moved to `NumericElement`). Every `f64`-into-element
conversion in the workspace (~390 sites) resolved through the
redeclaration instead of the single source of truth, and the
`Complex<T>` component conversion had no typed home at all.

Two designs were considered:

1. `Scalar: CastFrom<f64>` with `CastFrom<f64> for Complex<T>` added
   upstream in eunomia. Dead on arrival: eunomia 0.10 retired
   `CastFrom`/`CastTo` (eunomia ADR 0007), whose replacement map
   points `f64`-into-float at `FloatElement::from_f64`.
2. Per-site retargeting: float sites name
   `<T as FloatElement>::from_f64`, int sites name
   `IntegerTarget::from_truncated`. A `Float: FloatElement` supertrait
   was rejected first: `Float` redeclares ~25 `FloatElement` methods,
   so the supertrait makes every `x.exp()`/`x.floor()` on `T: Float`
   ambiguous, and the per-site bounds — though viral — stay explicit.

The sweep's executed design (`16a73a81`) confirmed option 2,
including the `Complex` gap closure: `Complex<T>` calls
`FloatElement::from_f64` on its component `T`, never on itself, so
no new spelling is needed.

## Decision

Delete `Scalar::from_f64` (declaration plus the native, half,
integer, and complex impls) after retargeting every call site:

- float sites (native, half, `Complex` components) use
  `<T as FloatElement>::from_f64`, declared upstream with
  round-to-nearest-even semantics;
- integer sites use `IntegerTarget::from_truncated`, whose body is
  literally `x as Self` — identical values by construction,
  including NaN (`0`) and saturation behavior;
- `Float` gains no `FloatElement` supertrait; generic sites carry
  explicit `+ FloatElement` bounds.

## Consequences

- ~390 sites migrated across the per-crate slices (#549 for
  `from_truncated`, #553–#557 for the float retargets, pool kernels
  via #559's leto delegation) plus this deletion; the workspace
  gate proves values unchanged.
- `Scalar` now declares only its slice-kernel surface
  (`sqrt_val`/`abs_val`/slice folds) over the eunomia supertraits
  (`NumericElement`, `Pod`); backend `Scalar` traits extend rather
  than redeclare the vocabulary.
- Count-typed conversions spelled `from_f64(n as f64)` remain valid
  but spell their intent as float conversion; migrating them to a
  count-typed spelling is separate work (COEUS-COUNT-CONV-SPELLING),
  since `from_count` does not exist yet and its exactness contract
  (exact vs rounded vs rejected above 2^53) is undecided.
