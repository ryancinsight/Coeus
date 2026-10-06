use eunomia::NumericElement;
// ── Integer implementations ──
// Scalar + Int for i8, i16, i32, i64, u8, u16, u32, u64.

use super::traits::{private, Int, Scalar};

macro_rules! impl_scalar_int_signed {
    ($t:ty) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            // `Scalar` keeps only what no provider owns; identities live on
            // `NumericElement` and are used directly at call sites.
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as Self
            }
            #[inline(always)]
            fn total_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }
            #[inline(always)]
            fn total_mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
        }
        impl Int for $t {
            #[inline(always)]
            fn count_zeros(self) -> u32 {
                self.count_zeros()
            }
            #[inline(always)]
            fn leading_zeros(self) -> u32 {
                self.leading_zeros()
            }
            #[inline(always)]
            fn trailing_zeros(self) -> u32 {
                self.trailing_zeros()
            }
            #[inline(always)]
            fn rotate_left(self, n: u32) -> Self {
                self.rotate_left(n)
            }
            #[inline(always)]
            fn rotate_right(self, n: u32) -> Self {
                self.rotate_right(n)
            }
            #[inline(always)]
            fn pow(self, exp: u32) -> Self {
                self.pow(exp)
            }
        }
    };
}

macro_rules! impl_scalar_int_unsigned {
    ($t:ty) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            // `Scalar` keeps only what no provider owns; identities live on
            // `NumericElement` and are used directly at call sites.
            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as Self
            }
            #[inline(always)]
            fn total_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }
            #[inline(always)]
            fn total_mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
        }
        impl Int for $t {
            #[inline(always)]
            fn count_zeros(self) -> u32 {
                self.count_zeros()
            }
            #[inline(always)]
            fn leading_zeros(self) -> u32 {
                self.leading_zeros()
            }
            #[inline(always)]
            fn trailing_zeros(self) -> u32 {
                self.trailing_zeros()
            }
            #[inline(always)]
            fn rotate_left(self, n: u32) -> Self {
                self.rotate_left(n)
            }
            #[inline(always)]
            fn rotate_right(self, n: u32) -> Self {
                self.rotate_right(n)
            }
            #[inline(always)]
            fn pow(self, exp: u32) -> Self {
                self.pow(exp)
            }
        }
    };
}

impl_scalar_int_signed!(i8);
impl_scalar_int_signed!(i16);
impl_scalar_int_signed!(i32);
impl_scalar_int_signed!(i64);
impl_scalar_int_unsigned!(u8);
impl_scalar_int_unsigned!(u16);
impl_scalar_int_unsigned!(u32);
impl_scalar_int_unsigned!(u64);

macro_rules! impl_cpu_unary_dispatch_int {
    ($t:ty) => {
        impl $crate::dtype::CpuUnaryDispatch for $t {
            #[inline(always)]
            fn eval_unary(op: $crate::dtype::CpuUnaryOp, x: Self) -> Self {
                use $crate::dtype::CpuUnaryOp;
                match op {
                    CpuUnaryOp::Relu => {
                        if x > <Self as NumericElement>::ZERO {
                            x
                        } else {
                            <Self as NumericElement>::ZERO
                        }
                    }
                    CpuUnaryOp::ReluGrad => {
                        if x > <Self as NumericElement>::ZERO {
                            <Self as NumericElement>::ONE
                        } else {
                            <Self as NumericElement>::ZERO
                        }
                    }
                    CpuUnaryOp::Neg => <Self as NumericElement>::ZERO - x,
                    CpuUnaryOp::Abs => x.abs(),
                    CpuUnaryOp::Sqrt => x.sqrt(),
                    CpuUnaryOp::SigmoidGrad => x * (<Self as NumericElement>::ONE - x),
                    CpuUnaryOp::TanhGrad => <Self as NumericElement>::ONE - x * x,
                    CpuUnaryOp::LeakyRelu(slope_bits) => {
                        let slope = Self::from_f64(f64::from_bits(slope_bits));
                        if x >= <Self as NumericElement>::ZERO {
                            x
                        } else {
                            slope * x
                        }
                    }
                    // PReLU / LeakyReLU gradient oracle: dx = 1 if x > 0 else slope.
                    // Matches PyTorch's contract which returns slope (not 1) at x = 0.
                    CpuUnaryOp::LeakyReluGrad(slope_bits) => {
                        let slope = Self::from_f64(f64::from_bits(slope_bits));
                        if x > <Self as NumericElement>::ZERO {
                            <Self as NumericElement>::ONE
                        } else {
                            slope
                        }
                    }
                    CpuUnaryOp::Recip => {
                        if x == <Self as NumericElement>::ZERO {
                            <Self as NumericElement>::ZERO
                        } else {
                            <Self as NumericElement>::ONE / x
                        }
                    }
                    CpuUnaryOp::Sign => {
                        if x > <Self as NumericElement>::ZERO {
                            <Self as NumericElement>::ONE
                        } else if x < <Self as NumericElement>::ZERO {
                            // Two's complement: !ZERO = -1 for signed types.
                            // This branch is unreachable for unsigned types (x >= 0 always).
                            !<Self as NumericElement>::ZERO
                        } else {
                            <Self as NumericElement>::ZERO
                        }
                    }
                    // Floor/ceil/round/trunc are identity for integers.
                    CpuUnaryOp::Floor
                    | CpuUnaryOp::Ceil
                    | CpuUnaryOp::Round
                    | CpuUnaryOp::Trunc => x,
                    _ => panic!("Float unary operation not supported for integer types"),
                }
            }
        }
    };
}

impl_cpu_unary_dispatch_int!(i8);
impl_cpu_unary_dispatch_int!(i16);
impl_cpu_unary_dispatch_int!(i32);
impl_cpu_unary_dispatch_int!(i64);
impl_cpu_unary_dispatch_int!(u8);
impl_cpu_unary_dispatch_int!(u16);
impl_cpu_unary_dispatch_int!(u32);
impl_cpu_unary_dispatch_int!(u64);
