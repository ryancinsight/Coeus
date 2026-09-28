use crate::dtype::traits::{private, Float, FloatOps, Scalar};
use eunomia::NumericElement;
use eunomia::{Bf16, F16};

/// Computes floor directly from a binary floating-point encoding.
///
/// Non-finite encodings and signed zero retain their bits. For finite values
/// with magnitude below one, the result is positive zero or negative one.
/// Otherwise the fractional significand bits are cleared, and a negative
/// value with any cleared bits set is advanced to the next integer magnitude.
fn floor_bits<const EXPONENT_BITS: u8, const FRACTION_BITS: u8, const EXPONENT_BIAS: u16>(
    bits: u16,
) -> u16 {
    let sign_mask = 1_u16 << (EXPONENT_BITS + FRACTION_BITS);
    let sign = bits & sign_mask;
    let exponent_mask = (1_u16 << EXPONENT_BITS) - 1;
    let exponent = (bits >> FRACTION_BITS) & exponent_mask;

    if exponent == exponent_mask {
        return bits;
    }

    let magnitude = bits & !sign_mask;
    if magnitude == 0 {
        return bits;
    }

    if exponent < EXPONENT_BIAS {
        return if sign == 0 {
            0
        } else {
            sign | (EXPONENT_BIAS << FRACTION_BITS)
        };
    }

    let integer_exponent = exponent - EXPONENT_BIAS;
    if integer_exponent >= u16::from(FRACTION_BITS) {
        return bits;
    }

    let fractional_bits = u16::from(FRACTION_BITS) - integer_exponent;
    let fractional_mask = (1_u16 << fractional_bits) - 1;
    if magnitude & fractional_mask == 0 {
        return bits;
    }

    let truncated = bits & !fractional_mask;
    if sign == 0 {
        truncated
    } else {
        truncated + (1_u16 << fractional_bits)
    }
}

macro_rules! impl_scalar_float_half {
    ($t:ty, $max:expr, $min_pos:expr, $exponent_bits:literal, $fraction_bits:literal, $exponent_bias:literal) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline(always)]
            fn zero() -> Self {
                Self::ZERO
            }
            #[inline(always)]
            fn one() -> Self {
                Self::ONE
            }
            #[inline(always)]
            fn to_f64(self) -> f64 {
                <Self as NumericElement>::to_f64(self)
            }
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                <Self as eunomia::FloatElement>::from_f64(v)
            }
            #[inline(always)]
            fn from_usize(v: usize) -> Self {
                <Self as eunomia::FloatElement>::from_f64(v as f64)
            }
            #[inline(always)]
            fn sqrt_val(self) -> Self {
                <Self as NumericElement>::sqrt(self)
            }
            #[inline(always)]
            fn abs_val(self) -> Self {
                <Self as NumericElement>::abs(self)
            }
            #[inline(always)]
            fn total_add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline(always)]
            fn total_mul(self, rhs: Self) -> Self {
                self * rhs
            }
        }
        impl FloatOps for $t {
            #[inline(always)]
            fn exp_op(self) -> Self {
                <Self as eunomia::FloatElement>::exp(self)
            }
            #[inline(always)]
            fn exp2_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.exp2())
            }
            #[inline(always)]
            fn log_op(self) -> Self {
                <Self as eunomia::FloatElement>::ln(self)
            }
            #[inline(always)]
            fn tanh_op(self) -> Self {
                <Self as eunomia::FloatElement>::tanh(self)
            }
            #[inline(always)]
            fn sin_op(self) -> Self {
                <Self as eunomia::FloatElement>::sin(self)
            }
            #[inline(always)]
            fn cos_op(self) -> Self {
                <Self as eunomia::FloatElement>::cos(self)
            }
            #[inline(always)]
            fn erf_op(self) -> Self {
                <Self as eunomia::FloatElement>::erf(self)
            }
            #[inline(always)]
            fn erfc_op(self) -> Self {
                <Self as eunomia::FloatElement>::erfc(self)
            }
            #[inline(always)]
            fn lgamma_op(self) -> Self {
                <Self as eunomia::FloatElement>::lgamma(self)
            }
            #[inline(always)]
            fn tan_op(self) -> Self {
                <Self as eunomia::FloatElement>::tan(self)
            }
            #[inline(always)]
            fn asin_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.asin())
            }
            #[inline(always)]
            fn acos_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.acos())
            }
            #[inline(always)]
            fn atan_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.atan())
            }
            #[inline(always)]
            fn sinh_op(self) -> Self {
                <Self as eunomia::FloatElement>::sinh(self)
            }
            #[inline(always)]
            fn cosh_op(self) -> Self {
                <Self as eunomia::FloatElement>::cosh(self)
            }
            #[inline(always)]
            fn log2_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log2())
            }
            #[inline(always)]
            fn log10_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log10())
            }
            #[inline(always)]
            fn atanh_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.atanh())
            }
            #[inline(always)]
            fn asinh_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.asinh())
            }
            #[inline(always)]
            fn acosh_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.acosh())
            }
            #[inline(always)]
            fn expm1_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.exp_m1())
            }
            #[inline(always)]
            fn log1p_op(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.ln_1p())
            }
            #[inline(always)]
            fn gelu_op(self) -> Self {
                let half = Self::from_f64(0.5);
                let one = Self::one();
                let inv_sqrt_two = Self::from_f64(core::f64::consts::FRAC_1_SQRT_2);
                half * self * (one + (self * inv_sqrt_two).erf_op())
            }
            #[inline(always)]
            fn sigmoid_op(self) -> Self {
                let x_f = <Self as NumericElement>::to_f64(self);
                let res = 1.0 / (1.0 + (-x_f).exp());
                Self::from_f64(res)
            }
        }
        impl Float for $t {
            const MAX: Self = $max;
            const MIN_POSITIVE: Self = $min_pos;
            const NAN: Self = Self::NAN;
            const NEG_INFINITY: Self = Self::NEG_INFINITY;
            const INFINITY: Self = Self::INFINITY;
            #[inline(always)]
            fn floor(self) -> Self {
                Self::from_bits(
                    floor_bits::<$exponent_bits, $fraction_bits, $exponent_bias>(self.to_bits()),
                )
            }
            #[inline(always)]
            fn ceil(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.ceil())
            }
            #[inline(always)]
            fn round(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.round())
            }
            #[inline(always)]
            fn trunc(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.trunc())
            }
            #[inline(always)]
            fn fract(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.fract())
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <Self as NumericElement>::abs(self)
            }
            #[inline(always)]
            fn signum(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.signum())
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <Self as NumericElement>::sqrt(self)
            }
            #[inline(always)]
            fn exp(self) -> Self {
                <Self as eunomia::FloatElement>::exp(self)
            }
            #[inline(always)]
            fn exp2(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.exp2())
            }
            #[inline(always)]
            fn ln(self) -> Self {
                <Self as eunomia::FloatElement>::ln(self)
            }
            #[inline(always)]
            fn log2(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log2())
            }
            #[inline(always)]
            fn log10(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.log10())
            }
            #[inline(always)]
            fn sin(self) -> Self {
                <Self as eunomia::FloatElement>::sin(self)
            }
            #[inline(always)]
            fn cos(self) -> Self {
                <Self as eunomia::FloatElement>::cos(self)
            }
            #[inline(always)]
            fn tan(self) -> Self {
                <Self as eunomia::FloatElement>::tan(self)
            }
            #[inline(always)]
            fn asin(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.asin())
            }
            #[inline(always)]
            fn acos(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.acos())
            }
            #[inline(always)]
            fn atan(self) -> Self {
                let v = <Self as NumericElement>::to_f64(self);
                <Self as eunomia::FloatElement>::from_f64(v.atan())
            }
            #[inline(always)]
            fn sinh(self) -> Self {
                <Self as eunomia::FloatElement>::sinh(self)
            }
            #[inline(always)]
            fn cosh(self) -> Self {
                <Self as eunomia::FloatElement>::cosh(self)
            }
            #[inline(always)]
            fn tanh(self) -> Self {
                <Self as eunomia::FloatElement>::tanh(self)
            }
            #[inline(always)]
            fn powf(self, n: Self) -> Self {
                <Self as eunomia::FloatElement>::powf(self, n)
            }
            #[inline(always)]
            fn powi(self, exp: i32) -> Self {
                <Self as eunomia::FloatElement>::powi(self, exp)
            }
            #[inline(always)]
            fn is_integer(self) -> bool {
                let f = <Self as NumericElement>::to_f64(self);
                f.is_finite() && f == f.trunc()
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                <Self as NumericElement>::is_nan(self)
            }
            #[inline(always)]
            fn is_infinite(self) -> bool {
                let f = <Self as NumericElement>::to_f64(self);
                f.is_infinite()
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                <Self as NumericElement>::is_finite(self)
            }
        }
    };
}

// F16: largest finite = 65504.0, smallest positive normal = 2^-14
impl_scalar_float_half!(F16, F16(0x7BFF), F16(0x0040), 5, 10, 15);
// Bf16: largest finite ≈ 3.3895e38, smallest positive normal = 2^-126
impl_scalar_float_half!(Bf16, Bf16(0x7F7F), Bf16(0x0080), 8, 7, 127);

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_floor_matches_f32_reference<T: Float>(
        from_bits: fn(u16) -> T,
        to_bits: fn(T) -> u16,
        to_f32: fn(T) -> f32,
        from_f32: fn(f32) -> T,
    ) {
        for bits in u16::MIN..=u16::MAX {
            let value = from_bits(bits);
            let widened = to_f32(value);
            let expected = if widened.is_finite() {
                to_bits(from_f32(widened.floor()))
            } else {
                bits
            };

            assert_eq!(
                to_bits(<T as Float>::floor(value)),
                expected,
                "floor mismatch for input bits {bits:#06x}"
            );
        }
    }

    #[test]
    fn half_float_floor_matches_reference_for_all_encodings() {
        assert_floor_matches_f32_reference::<F16>(
            F16::from_bits,
            F16::to_bits,
            F16::to_f32,
            F16::from_f32,
        );
        assert_floor_matches_f32_reference::<Bf16>(
            Bf16::from_bits,
            Bf16::to_bits,
            Bf16::to_f32,
            Bf16::from_f32,
        );
    }

    fn assert_floor_special_values<T: Float>(positive_subnormal: T, negative_subnormal: T) {
        let to_f64 = <T as Scalar>::to_f64;
        let positive_zero = <T as Scalar>::from_f64(0.0);
        let negative_zero = <T as Scalar>::from_f64(-0.0);

        assert_eq!(
            to_f64(<T as Float>::floor(positive_zero)).to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            to_f64(<T as Float>::floor(negative_zero)).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            to_f64(<T as Float>::floor(<T as Float>::INFINITY)),
            f64::INFINITY
        );
        assert_eq!(
            to_f64(<T as Float>::floor(<T as Float>::NEG_INFINITY)),
            f64::NEG_INFINITY
        );
        assert!(<T as Float>::is_nan(<T as Float>::floor(<T as Float>::NAN)));
        assert_eq!(to_f64(<T as Float>::floor(positive_subnormal)), 0.0_f64);
        assert_eq!(to_f64(<T as Float>::floor(negative_subnormal)), -1.0_f64);
    }

    #[test]
    fn floor_preserves_its_special_value_contract() {
        assert_floor_special_values::<F16>(F16::from_bits(0x0001), F16::from_bits(0x8001));
        assert_floor_special_values::<Bf16>(Bf16::from_bits(0x0001), Bf16::from_bits(0x8001));
        assert_floor_special_values::<f32>(
            f32::from_bits(0x0000_0001),
            f32::from_bits(0x8000_0001),
        );
        assert_floor_special_values::<f64>(
            f64::from_bits(0x0000_0000_0000_0001),
            f64::from_bits(0x8000_0000_0000_0001),
        );
    }
}
