//! Scalar widening, machine epsilon, and tolerance scales.

use coeus_core::{Float, Scalar};

/// Widen a scalar to `f64` for the error analysis.
///
/// Routes through [`NumericElement`](coeus_core::NumericElement), the single
/// vocabulary that survived the `Scalar` dedup; every widening in this
/// module funnels here so the trait is named once.
#[inline]
pub(super) fn widen<T: Scalar>(value: T) -> f64 {
    <T as coeus_core::NumericElement>::to_f64(value)
}

/// Round an `f64` back into the differentiated scalar type.
///
/// The counterpart to [`widen`], via [`FloatElement`](coeus_core::FloatElement).
#[inline]
pub(super) fn narrow<T: Scalar + coeus_core::FloatElement>(value: f64) -> T {
    <T as coeus_core::FloatElement>::from_f64(value)
}

/// Machine epsilon of `T`: the gap between `1` and the next representable
/// value above it.
///
/// Derived from `T`'s own arithmetic rather than read from a constant because
/// [`coeus_core::Float`] does not expose one — it carries `MAX`,
/// `MIN_POSITIVE`, `NAN` and the infinities, but no `EPSILON`. The obvious
/// substitute, `eunomia::RealField::EPSILON`, is unavailable here: `eunomia` is
/// a dev-dependency of this crate, and promoting it to name the constant would
/// put `RealField` into the public bound of [`fn@crate::gradcheck::gradcheck`],
/// excluding the reduced-precision types that implement [`Float`] but not `RealField`.
///
/// Halving a power of two is exact in every binary format until subnormals, and
/// the loop stops many orders of magnitude above them, so the returned value is
/// the exact representable epsilon — not an approximation of it. It converges
/// in one iteration per mantissa bit (53 for `f64`, 11 for `F16`).
pub(super) fn machine_epsilon<T: Float + coeus_core::FloatElement>() -> f64 {
    let one = T::ONE;
    let mut epsilon = T::ONE;
    loop {
        let halved = narrow::<T>(widen(epsilon) * 0.5);
        if one + halved == one {
            return widen(epsilon);
        }
        epsilon = halved;
    }
}

/// Multiplier applied to the `ε^(2/3)` accuracy floor to obtain the default
/// pass tolerance.
///
/// The floor is the *best* attainable accuracy for one perturbed coordinate of
/// an `O(1)`-conditioned function. A real loss reduces over many elements, so
/// its round-off accumulates above that single-coordinate floor; this margin
/// covers that accumulation without approaching the scale of a genuinely wrong
/// gradient, which is typically wrong by an `O(1)` relative amount.
pub(super) const DEFAULT_TOLERANCE_SCALE: f64 = 64.0;

/// Multiplier applied to the accuracy floor to obtain the zero-gradient floor.
///
/// Held below [`DEFAULT_TOLERANCE_SCALE`] so that a gradient large enough to be
/// meaningfully compared is never rejected as trivially zero.
pub(super) const ZERO_FLOOR_SCALE: f64 = 8.0;
