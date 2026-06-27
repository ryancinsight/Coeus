//! Non-ZST tag types for LeakyReLU (carry a runtime slope parameter).

use coeus_core::Scalar;

/// LeakyRelu tag — NOT a ZST; carries slope encoded as `f64::to_bits()`.
///
/// Cannot implement `UnaryOpTag` because `WGSL_TEMPLATE` must be `&'static str`
/// and the slope is a runtime value. Handled explicitly in fuse evaluator.
#[derive(Clone, Copy)]
pub struct LeakyReluTag {
    /// `f64::to_bits(slope)` — negative-region slope.
    pub slope_bits: u64,
}

impl LeakyReluTag {
    /// Create a new tag with the given negative-region slope.
    #[inline]
    pub fn new(slope: f64) -> Self {
        Self {
            slope_bits: f64::to_bits(slope),
        }
    }

    /// Decode the stored slope back to `f64`.
    #[inline]
    pub fn slope(&self) -> f64 {
        f64::from_bits(self.slope_bits)
    }

    /// Apply LeakyReLU: `x >= 0 ? x : slope * x`.
    #[inline(always)]
    pub fn apply<T: Scalar>(&self, x: T) -> T {
        let slope = T::from_f64(self.slope());
        if x >= T::zero() {
            x
        } else {
            slope * x
        }
    }
}

/// LeakyRelu gradient tag — NOT a ZST; carries slope encoded as `f64::to_bits()`.
///
/// Same static-string constraint prevents `UnaryOpTag` implementation.
/// Handled explicitly in fuse evaluator.
#[derive(Clone, Copy)]
pub struct LeakyReluGradTag {
    /// `f64::to_bits(slope)` — negative-region slope.
    pub slope_bits: u64,
}

impl LeakyReluGradTag {
    /// Create a new gradient tag with the given negative-region slope.
    #[inline]
    pub fn new(slope: f64) -> Self {
        Self {
            slope_bits: f64::to_bits(slope),
        }
    }

    /// Decode the stored slope back to `f64`.
    #[inline]
    pub fn slope(&self) -> f64 {
        f64::from_bits(self.slope_bits)
    }

    /// Apply LeakyReLU gradient: `x >= 0 ? 1 : slope`.
    #[inline(always)]
    pub fn apply<T: Scalar>(&self, x: T) -> T {
        let slope = T::from_f64(self.slope());
        if x >= T::zero() {
            T::one()
        } else {
            slope
        }
    }
}
