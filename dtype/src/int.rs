//! Integer data types - Complete implementation
//!
//! Provides all signed and unsigned integer types with full trait support.

use core::fmt;
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num_traits::{Num, NumCast, One, Signed, Zero};

use crate::traits::{DataType, IntExt};
use crate::Dtype;

/// Macro to implement complete integer type
macro_rules! impl_int_dtype {
    ($name:ident, $inner:ty, $dtype:expr, signed) => {
        /// Signed integer type wrapper
        #[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            /// Create a new value
            #[must_use]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            /// Get the inner value
            #[must_use]
            pub const fn get(self) -> $inner {
                self.0
            }

            /// Minimum value for this type
            #[must_use]
            pub const fn min_value() -> Self {
                Self(<$inner>::MIN)
            }

            /// Maximum value for this type
            #[must_use]
            pub const fn max_value() -> Self {
                Self(<$inner>::MAX)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl Zero for $name {
            fn zero() -> Self {
                Self(0)
            }

            fn is_zero(&self) -> bool {
                self.0 == 0
            }
        }

        impl One for $name {
            fn one() -> Self {
                Self(1)
            }
        }

        impl Num for $name {
            type FromStrRadixErr = <$inner as Num>::FromStrRadixErr;

            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$inner>::from_str_radix(str, radix).map(Self)
            }
        }

        impl NumCast for $name {
            fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
                <$inner as NumCast>::from(n).map(Self)
            }
        }

        impl num_traits::ToPrimitive for $name {
            fn to_i64(&self) -> Option<i64> {
                self.0.to_i64()
            }

            fn to_u64(&self) -> Option<u64> {
                self.0.to_u64()
            }
        }

        impl Add for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0.wrapping_add(rhs.0))
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0.wrapping_sub(rhs.0))
            }
        }

        impl Mul for $name {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self::Output {
                Self(self.0.wrapping_mul(rhs.0))
            }
        }

        impl Div for $name {
            type Output = Self;
            fn div(self, rhs: Self) -> Self::Output {
                Self(self.0 / rhs.0)
            }
        }

        impl Rem for $name {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self::Output {
                Self(self.0 % rhs.0)
            }
        }

        impl Neg for $name {
            type Output = Self;
            fn neg(self) -> Self::Output {
                Self(self.0.wrapping_neg())
            }
        }

        impl Signed for $name {
            fn abs(&self) -> Self {
                Self(self.0.wrapping_abs())
            }

            fn abs_sub(&self, other: &Self) -> Self {
                if self.0 > other.0 {
                    Self(self.0 - other.0)
                } else {
                    Self::zero()
                }
            }

            fn signum(&self) -> Self {
                Self(self.0.signum())
            }

            fn is_positive(&self) -> bool {
                self.0.is_positive()
            }

            fn is_negative(&self) -> bool {
                self.0.is_negative()
            }
        }

        impl DataType for $name {
            fn dtype() -> Dtype {
                $dtype
            }
        }

        impl IntExt for $name {
            fn checked_add(self, rhs: Self) -> Option<Self> {
                self.0.checked_add(rhs.0).map(Self)
            }

            fn checked_sub(self, rhs: Self) -> Option<Self> {
                self.0.checked_sub(rhs.0).map(Self)
            }

            fn checked_mul(self, rhs: Self) -> Option<Self> {
                self.0.checked_mul(rhs.0).map(Self)
            }

            fn checked_div(self, rhs: Self) -> Option<Self> {
                self.0.checked_div(rhs.0).map(Self)
            }

            fn checked_rem(self, rhs: Self) -> Option<Self> {
                self.0.checked_rem(rhs.0).map(Self)
            }

            fn checked_neg(self) -> Option<Self> {
                self.0.checked_neg().map(Self)
            }

            fn checked_abs(self) -> Option<Self> {
                self.0.checked_abs().map(Self)
            }

            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }

            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }

            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }

            fn bitnot(self) -> Self {
                Self(!self.0)
            }

            fn checked_shl(self, rhs: u32) -> Option<Self> {
                self.0.checked_shl(rhs).map(Self)
            }

            fn checked_shr(self, rhs: u32) -> Option<Self> {
                self.0.checked_shr(rhs).map(Self)
            }

            fn leading_zeros(self) -> u32 {
                self.0.leading_zeros()
            }

            fn trailing_zeros(self) -> u32 {
                self.0.trailing_zeros()
            }

            fn count_ones(self) -> u32 {
                self.0.count_ones()
            }

            fn count_zeros(self) -> u32 {
                self.0.count_zeros()
            }
        }
    };

    ($name:ident, $inner:ty, $dtype:expr, unsigned) => {
        /// Unsigned integer type wrapper
        #[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
        #[repr(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            /// Create a new value
            #[must_use]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            /// Get the inner value
            #[must_use]
            pub const fn get(self) -> $inner {
                self.0
            }

            /// Minimum value for this type
            #[must_use]
            pub const fn min_value() -> Self {
                Self(<$inner>::MIN)
            }

            /// Maximum value for this type
            #[must_use]
            pub const fn max_value() -> Self {
                Self(<$inner>::MAX)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl Zero for $name {
            fn zero() -> Self {
                Self(0)
            }

            fn is_zero(&self) -> bool {
                self.0 == 0
            }
        }

        impl One for $name {
            fn one() -> Self {
                Self(1)
            }
        }

        impl Num for $name {
            type FromStrRadixErr = <$inner as Num>::FromStrRadixErr;

            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$inner>::from_str_radix(str, radix).map(Self)
            }
        }

        impl NumCast for $name {
            fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
                <$inner as NumCast>::from(n).map(Self)
            }
        }

        impl num_traits::ToPrimitive for $name {
            fn to_i64(&self) -> Option<i64> {
                self.0.to_i64()
            }

            fn to_u64(&self) -> Option<u64> {
                self.0.to_u64()
            }
        }

        impl Add for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self::Output {
                Self(self.0.wrapping_add(rhs.0))
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self::Output {
                Self(self.0.wrapping_sub(rhs.0))
            }
        }

        impl Mul for $name {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self::Output {
                Self(self.0.wrapping_mul(rhs.0))
            }
        }

        impl Div for $name {
            type Output = Self;
            fn div(self, rhs: Self) -> Self::Output {
                Self(self.0 / rhs.0)
            }
        }

        impl Rem for $name {
            type Output = Self;
            fn rem(self, rhs: Self) -> Self::Output {
                Self(self.0 % rhs.0)
            }
        }

        impl DataType for $name {
            fn dtype() -> Dtype {
                $dtype
            }
        }

        impl IntExt for $name {
            fn checked_add(self, rhs: Self) -> Option<Self> {
                self.0.checked_add(rhs.0).map(Self)
            }

            fn checked_sub(self, rhs: Self) -> Option<Self> {
                self.0.checked_sub(rhs.0).map(Self)
            }

            fn checked_mul(self, rhs: Self) -> Option<Self> {
                self.0.checked_mul(rhs.0).map(Self)
            }

            fn checked_div(self, rhs: Self) -> Option<Self> {
                self.0.checked_div(rhs.0).map(Self)
            }

            fn checked_rem(self, rhs: Self) -> Option<Self> {
                self.0.checked_rem(rhs.0).map(Self)
            }

            fn checked_neg(self) -> Option<Self> {
                if self.0 == 0 {
                    Some(Self(0))
                } else {
                    None
                }
            }

            fn checked_abs(self) -> Option<Self> {
                Some(self)
            }

            fn bitand(self, rhs: Self) -> Self {
                Self(self.0 & rhs.0)
            }

            fn bitor(self, rhs: Self) -> Self {
                Self(self.0 | rhs.0)
            }

            fn bitxor(self, rhs: Self) -> Self {
                Self(self.0 ^ rhs.0)
            }

            fn bitnot(self) -> Self {
                Self(!self.0)
            }

            fn checked_shl(self, rhs: u32) -> Option<Self> {
                self.0.checked_shl(rhs).map(Self)
            }

            fn checked_shr(self, rhs: u32) -> Option<Self> {
                self.0.checked_shr(rhs).map(Self)
            }

            fn leading_zeros(self) -> u32 {
                self.0.leading_zeros()
            }

            fn trailing_zeros(self) -> u32 {
                self.0.trailing_zeros()
            }

            fn count_ones(self) -> u32 {
                self.0.count_ones()
            }

            fn count_zeros(self) -> u32 {
                self.0.count_zeros()
            }
        }
    };
}

// Implement all integer types
impl_int_dtype!(Int8, i8, Dtype::Int8, signed);
impl_int_dtype!(Int16, i16, Dtype::Int16, signed);
impl_int_dtype!(Int32, i32, Dtype::Int32, signed);
impl_int_dtype!(Int64, i64, Dtype::Int64, signed);
impl_int_dtype!(UInt8, u8, Dtype::UInt8, unsigned);
impl_int_dtype!(UInt16, u16, Dtype::UInt16, unsigned);
impl_int_dtype!(UInt32, u32, Dtype::UInt32, unsigned);
impl_int_dtype!(UInt64, u64, Dtype::UInt64, unsigned);

// Re-exports
pub use self::{Int16 as I16, Int32 as I32, Int64 as I64, Int8 as I8};
pub use self::{UInt16 as U16, UInt32 as U32, UInt64 as U64, UInt8 as U8};
