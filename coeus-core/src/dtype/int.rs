// ── Integer implementations ──
// Scalar + Int for i8, i16, i32, i64, u8, u16, u32, u64.

use super::traits::{private, Scalar, Int, FloatOps};

macro_rules! impl_scalar_int_signed {
    ($t:ty) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 { self as f64 }
            #[inline(always)]
            fn from_f64(v: f64) -> Self { v as Self }
            #[inline(always)]
            fn sqrt_val(self) -> Self { (self.to_f64().sqrt()) as Self }
            #[inline(always)]
            fn abs_val(self) -> Self { self.abs() }
        }
        impl FloatOps for $t {
            fn exp_op(self) -> Self { panic!("exp not supported on integer types") }
            fn log_op(self) -> Self { panic!("log not supported on integer types") }
            fn tanh_op(self) -> Self { panic!("tanh not supported on integer types") }
            fn sin_op(self) -> Self { panic!("sin not supported on integer types") }
            fn cos_op(self) -> Self { panic!("cos not supported on integer types") }
            fn gelu_op(self) -> Self { panic!("gelu not supported on integer types") }
            fn sigmoid_op(self) -> Self { panic!("sigmoid not supported on integer types") }
        }
        impl Int for $t {
            #[inline(always)] fn count_ones(self) -> u32 { self.count_ones() }
            #[inline(always)] fn count_zeros(self) -> u32 { self.count_zeros() }
            #[inline(always)] fn leading_zeros(self) -> u32 { self.leading_zeros() }
            #[inline(always)] fn trailing_zeros(self) -> u32 { self.trailing_zeros() }
            #[inline(always)] fn rotate_left(self, n: u32) -> Self { self.rotate_left(n) }
            #[inline(always)] fn rotate_right(self, n: u32) -> Self { self.rotate_right(n) }
            #[inline(always)] fn pow(self, exp: u32) -> Self { self.pow(exp) }
            #[inline(always)] fn abs(self) -> Self { self.abs() }
        }
    };
}

macro_rules! impl_scalar_int_unsigned {
    ($t:ty) => {
        impl private::Sealed for $t {}
        impl Scalar for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 { self as f64 }
            #[inline(always)]
            fn from_f64(v: f64) -> Self { v as Self }
            #[inline(always)]
            fn sqrt_val(self) -> Self { (self.to_f64().sqrt()) as Self }
            #[inline(always)]
            fn abs_val(self) -> Self { self }
        }
        impl FloatOps for $t {
            fn exp_op(self) -> Self { panic!("exp not supported on integer types") }
            fn log_op(self) -> Self { panic!("log not supported on integer types") }
            fn tanh_op(self) -> Self { panic!("tanh not supported on integer types") }
            fn sin_op(self) -> Self { panic!("sin not supported on integer types") }
            fn cos_op(self) -> Self { panic!("cos not supported on integer types") }
            fn gelu_op(self) -> Self { panic!("gelu not supported on integer types") }
            fn sigmoid_op(self) -> Self { panic!("sigmoid not supported on integer types") }
        }
        impl Int for $t {
            #[inline(always)] fn count_ones(self) -> u32 { self.count_ones() }
            #[inline(always)] fn count_zeros(self) -> u32 { self.count_zeros() }
            #[inline(always)] fn leading_zeros(self) -> u32 { self.leading_zeros() }
            #[inline(always)] fn trailing_zeros(self) -> u32 { self.trailing_zeros() }
            #[inline(always)] fn rotate_left(self, n: u32) -> Self { self.rotate_left(n) }
            #[inline(always)] fn rotate_right(self, n: u32) -> Self { self.rotate_right(n) }
            #[inline(always)] fn pow(self, exp: u32) -> Self { self.pow(exp) }
            #[inline(always)] fn abs(self) -> Self { self }
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
