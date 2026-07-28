use super::UnaryOpTag;
use coeus_core::{FloatOps, Scalar};

#[derive(Clone, Copy)]
/// ReLU operation tag.
pub struct Relu;
impl<T: Scalar> UnaryOpTag<T> for Relu {
    const WGSL_TEMPLATE: &'static str = "max(({}), 0.0)";
    #[inline(always)]
    fn apply(x: T) -> T {
        if x > T::zero() { x } else { T::zero() }
    }
}

#[derive(Clone, Copy)]
/// Negation operation tag.
pub struct Neg;
impl<T: Scalar> UnaryOpTag<T> for Neg {
    const WGSL_TEMPLATE: &'static str = "-(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::zero() - x
    }
}

#[derive(Clone, Copy)]
/// Absolute value operation tag.
pub struct Abs;
impl<T: Scalar> UnaryOpTag<T> for Abs {
    const WGSL_TEMPLATE: &'static str = "abs(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.abs_val()
    }
}

#[derive(Clone, Copy)]
/// Square root operation tag.
pub struct Sqrt;
impl<T: Scalar> UnaryOpTag<T> for Sqrt {
    const WGSL_TEMPLATE: &'static str = "sqrt(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        x.sqrt_val()
    }
}

/// Element-wise reciprocal: `1 / x`.
#[derive(Clone, Copy)]
pub struct Recip;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Recip {
    const WGSL_TEMPLATE: &'static str = "(1.0 / ({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        T::one() / x
    }
}

/// Element-wise signum: `-1`, `0`, or `1`.
#[derive(Clone, Copy)]
pub struct Sign;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Sign {
    const WGSL_TEMPLATE: &'static str = "sign(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        if x > T::zero() {
            T::one()
        } else if x < T::zero() {
            T::zero() - T::one()
        } else {
            T::zero()
        }
    }
}

/// Element-wise floor.
#[derive(Clone, Copy)]
pub struct Floor;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Floor {
    const WGSL_TEMPLATE: &'static str = "floor(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).floor())
    }
}

/// Element-wise ceil.
#[derive(Clone, Copy)]
pub struct Ceil;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Ceil {
    const WGSL_TEMPLATE: &'static str = "ceil(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).ceil())
    }
}

/// Element-wise round to nearest integer, ties to even (banker's rounding).
///
/// Matches `torch.round` / IEEE-754 roundTiesToEven; WGSL's `round()` builtin
/// has the same ties-to-even contract.
#[derive(Clone, Copy)]
pub struct Round;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Round {
    const WGSL_TEMPLATE: &'static str = "round(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).round_ties_even())
    }
}

/// Element-wise truncation toward zero.
#[derive(Clone, Copy)]
pub struct Trunc;
impl<T: Scalar + FloatOps> UnaryOpTag<T> for Trunc {
    const WGSL_TEMPLATE: &'static str = "trunc(({}))";
    #[inline(always)]
    fn apply(x: T) -> T {
        <T as Scalar>::from_f64(<T as Scalar>::to_f64(x).trunc())
    }
}
