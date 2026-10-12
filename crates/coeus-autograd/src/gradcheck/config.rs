//! Tuning for a gradcheck run.

use super::precision::DEFAULT_TOLERANCE_SCALE;

/// Tuning for a [`fn@crate::gradcheck::gradcheck`] run.
///
/// [`Default`] derives every value from the scalar type's machine epsilon; see
/// the module documentation. Construct with [`GradcheckConfig::default`] and
/// override only what a specific operation's conditioning requires.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct GradcheckConfig {
    /// Multiplier on the `ε^(2/3)` accuracy floor that sets the pass tolerance.
    ///
    /// Raise it only for an operation whose forward is measurably worse
    /// conditioned than `O(1)`, and record why at the call site — a tolerance
    /// widened to make a failing check pass is the defect it was meant to
    /// catch.
    pub tolerance_scale: f64,
}

impl Default for GradcheckConfig {
    fn default() -> Self {
        Self {
            tolerance_scale: DEFAULT_TOLERANCE_SCALE,
        }
    }
}
