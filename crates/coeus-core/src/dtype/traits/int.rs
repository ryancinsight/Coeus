//! Integer extension trait.

use super::Scalar;

/// Integer extension trait.
///
/// Provides bitwise operations and integer-specific math.
/// Implemented for i8, i16, i32, i64, u8, u16, u32, u64.
pub trait Int: Scalar {
    /// Count of set bits (popcount).
    fn count_ones(self) -> u32;
    /// Count of unset bits.
    fn count_zeros(self) -> u32;
    /// Count of leading zero bits.
    fn leading_zeros(self) -> u32;
    /// Count of trailing zero bits.
    fn trailing_zeros(self) -> u32;
    /// Bitwise rotate left by `n` positions.
    fn rotate_left(self, n: u32) -> Self;
    /// Bitwise rotate right by `n` positions.
    fn rotate_right(self, n: u32) -> Self;
    /// Integer power: self^exp.
    fn pow(self, exp: u32) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
}
