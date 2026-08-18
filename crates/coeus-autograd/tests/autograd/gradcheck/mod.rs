//! Finite-difference verification of the hand-written backward passes.
//!
//! Every check here runs through [`coeus_autograd::gradcheck`], which
//! reconstructs the gradient from forward evaluations alone. That makes it
//! independent of both the backward implementation *and* of any closed form a
//! test author might derive by hand — the two error sources a hand-derived
//! expected-value test cannot separate.
//!
//! # Conventions
//!
//! **Non-uniform output weighting.** `gradcheck` requires a scalar loss, and
//! the obvious `sum(op(x))` is the wrong reduction for any op whose rows sum to
//! a constant — `sum(softmax(x))` is identically `1`, so its gradient is exactly
//! zero and the comparison is vacuous. [`weighted`] reduces through a fixed
//! non-uniform tensor instead, which probes real Jacobian entries. The helper
//! rejects the vacuous case rather than passing it silently.
//!
//! **Inputs off the kinks.** A central difference straddles the point it is
//! evaluated at, so at a non-differentiable point it returns the *average* of
//! the one-sided derivatives and disagrees with any correct backward by an
//! `O(1)` amount. `relu`/`abs` at `0`, `max`/`min`/`sort`/`topk` at ties,
//! hinge losses at the margin, `clamp` at a bound and `sqrt` at `0` are all such
//! points. Inputs come from [`Sampler`], whose values are an irrational
//! rotation of the unit interval: the sequence is deterministic, reproducible,
//! and provably never lands on the rational kink locations those ops place at
//! `0`, at a margin, or at an exact tie. A seeded random generator would land on
//! one eventually and produce a test that fails once in a hundred runs, which is
//! worse than no test.
//!
//! **Derived tolerances.** The pass bound is `ε^(2/3)`-derived inside the
//! helper; see its module documentation for the step-size and accuracy-floor
//! derivation. A check that needs a wider bound passes an explicit
//! [`GradcheckConfig`](coeus_autograd::GradcheckConfig) and states the
//! conditioning that justifies it at the call site.

mod activation;
mod attention;
mod core_ops;
mod losses;
mod normalization;
mod reduction;
mod shape;

use coeus_autograd::{mul, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

/// The scalar and backend every check in this module differentiates.
///
/// `f64` is the strongest setting for a finite-difference oracle: its accuracy
/// floor is `ε^(2/3) ≈ 3.7e-11`, five orders of magnitude below `f32`'s
/// `2.4e-5`, so a wrong gradient has nowhere to hide inside the tolerance. The
/// backward implementations under test are generic over `T: Float`, and the
/// `f32` instantiation is exercised by the crate's parity suites.
pub type T64 = Tensor<f64, MoiraiBackend>;

/// Reciprocal of the golden ratio: the rotation step of [`Sampler`].
///
/// `1/φ = 0.6180339887…` is irrational, so the additive recurrence
/// `u_{i+1} = frac(u_i + 1/φ)` never repeats and never returns exactly `0`,
/// `1/2`, or any other rational the ops under test place a kink at. It is also
/// the rotation with the best low-discrepancy behaviour, so a handful of
/// samples still spreads across the interval instead of clustering — the
/// property a kink-avoiding *and* well-conditioned fixture needs.
const GOLDEN_RATIO_INVERSE: f64 = 0.618_033_988_749_894_9;

/// Deterministic irregular sample sequence over an open interval.
///
/// Each call to [`Sampler::values`] restarts from the configured phase, so a
/// fixture built twice with the same parameters is bit-identical — the
/// finite-difference comparison re-evaluates the forward `2·numel` times and
/// every one of them must see the same input.
pub struct Sampler {
    /// Starting phase of the rotation, in `(0, 1)`.
    phase: f64,
    /// Inclusive-exclusive open interval the samples are mapped into.
    range: (f64, f64),
}

impl Sampler {
    /// Samples spread over `(low, high)`, starting at `phase`.
    ///
    /// `phase` distinguishes one fixture from another; distinct phases give
    /// operands that are not translates of each other, so a backward that
    /// happens to be correct only on symmetric or repeated inputs still fails.
    pub const fn new(phase: f64, low: f64, high: f64) -> Self {
        Self {
            phase,
            range: (low, high),
        }
    }

    /// Values in roughly `[-0.9, 0.9]`, straddling zero without reaching it.
    ///
    /// The default for an op that is smooth across the origin.
    pub const fn signed(phase: f64) -> Self {
        Self::new(phase, -0.9, 0.9)
    }

    /// Strictly positive values in `(0.2, 1.8)`.
    ///
    /// For `log`, `sqrt`, `norm_p` and the losses whose domain excludes zero:
    /// the lower bound keeps the input away from the singularity at `0`, where
    /// the derivative diverges and no finite-difference bound holds.
    pub const fn positive(phase: f64) -> Self {
        Self::new(phase, 0.2, 1.8)
    }

    /// Values strictly inside `(0.1, 0.9)`, for probabilities.
    ///
    /// Bounded away from both `0` and `1` so `log(p)` and `log(1 - p)` stay
    /// well conditioned.
    pub const fn probability(phase: f64) -> Self {
        Self::new(phase, 0.1, 0.9)
    }

    /// `count` samples of the configured sequence.
    pub fn values(&self, count: usize) -> Vec<f64> {
        let (low, high) = self.range;
        (0..count)
            .map(|index| {
                let steps = f64::from(u32::try_from(index).expect("fixture sizes are small"));
                let unit = GOLDEN_RATIO_INVERSE.mul_add(steps, self.phase).fract();
                (high - low).mul_add(unit, low)
            })
            .collect()
    }

    /// A tensor of `shape` filled with the configured sequence.
    pub fn tensor(&self, shape: &[usize]) -> T64 {
        let count = shape.iter().product();
        T64::from_slice_on(shape.to_vec(), &self.values(count), &MoiraiBackend::new())
    }

    /// A non-differentiated [`Var`] of `shape`, for constants a closure captures.
    pub fn constant(&self, shape: &[usize]) -> Var<f64, MoiraiBackend> {
        Var::new(self.tensor(shape), false)
    }
}

/// A tensor of `shape` from the default signed sequence at `phase`.
pub fn tensor(shape: &[usize], phase: f64) -> T64 {
    Sampler::signed(phase).tensor(shape)
}

/// A fixed non-uniform weighting of an op's output.
///
/// Held constant across every perturbed evaluation so the loss stays a pure
/// function of the differentiated inputs, and non-uniform so that reductions
/// over a Jacobian with constant row sums do not cancel to zero.
pub fn weighting(shape: &[usize]) -> Var<f64, MoiraiBackend> {
    Sampler::new(0.311, -1.3, 1.7).constant(shape)
}

/// Reduce an op's output to a scalar through a fixed non-uniform weighting.
///
/// The canonical loss for these checks: `Σ w ⊙ y`, whose output gradient is `w`
/// rather than the all-ones a bare `sum` would supply. That distinction is what
/// keeps a Jacobian with constant row sums (softmax, the normalizations) from
/// cancelling to an identically-zero comparison.
pub fn weighted(
    output: &Var<f64, MoiraiBackend>,
    w: &Var<f64, MoiraiBackend>,
) -> Var<f64, MoiraiBackend> {
    sum(&mul(output, w))
}
