//! Finite-difference checks through the owning NN layers.
//!
//! These Module consumers live downstream of autograd, keeping the published
//! autograd/NN package graph acyclic. All f32/f64 cases retain their original
//! fixtures, non-uniform weighting and gradcheck-derived tolerances.

mod conv;
mod pooling;

use coeus_autograd::{mul, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_core::{CpuAddressableStorage, Float, FloatElement};
use coeus_tensor::Tensor;

/// Bound shared by every scalar type this module instantiates checks at.
///
/// `MoiraiBackend` is the CPU reference backend every check in this module
/// differentiates against; accelerator backends are covered by the
/// differential parity suites (`standards`: Differential Verification),
/// which compare a fixed scalar type across backends rather than a fixed
/// backend across scalar types.
///
/// `CpuAddressableStorage` is required by [`fn@coeus_autograd::gradcheck`]
/// itself to read perturbed values back off the backend; naming it here once
/// lets a generic check function state a single bound instead of repeating
/// gradcheck's own where-clause at every call site.
pub trait GradcheckScalar:
    Float + FloatElement + leto_ops::Scalar + coeus_leto::RealScalar
where
    MoiraiBackend: coeus_ops::BackendOps<Self>,
    <MoiraiBackend as coeus_core::ComputeBackend>::DeviceBuffer<Self>: CpuAddressableStorage<Self>,
{
}

impl<T> GradcheckScalar for T
where
    T: Float + FloatElement + leto_ops::Scalar + coeus_leto::RealScalar,
    MoiraiBackend: coeus_ops::BackendOps<T>,
    <MoiraiBackend as coeus_core::ComputeBackend>::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
}

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

    /// A tensor of `shape` filled with the configured sequence, at scalar `T`.
    pub fn tensor<T: GradcheckScalar>(&self, shape: &[usize]) -> Tensor<T, MoiraiBackend> {
        let count = shape.iter().product();
        let values: Vec<T> = self
            .values(count)
            .into_iter()
            .map(<T as FloatElement>::from_f64)
            .collect();
        Tensor::from_slice_on(shape.to_vec(), &values, &MoiraiBackend::new())
            .expect("invariant: test backend operation succeeds")
    }

    /// A non-differentiated [`Var`] of `shape`, for constants a closure captures.
    pub fn constant<T: GradcheckScalar>(&self, shape: &[usize]) -> Var<T, MoiraiBackend> {
        Var::new(self.tensor(shape), false).expect("invariant: test backend operation succeeds")
    }
}

/// A tensor of `shape` from the default signed sequence at `phase`.
pub fn tensor<T: GradcheckScalar>(shape: &[usize], phase: f64) -> Tensor<T, MoiraiBackend> {
    Sampler::signed(phase).tensor(shape)
}

/// A fixed non-uniform weighting of an op's output.
///
/// Held constant across every perturbed evaluation so the loss stays a pure
/// function of the differentiated inputs, and non-uniform so that reductions
/// over a Jacobian with constant row sums do not cancel to zero.
pub fn weighting<T: GradcheckScalar>(shape: &[usize]) -> Var<T, MoiraiBackend> {
    Sampler::new(0.311, -1.3, 1.7).constant(shape)
}

/// Reduce an op's output to a scalar through a fixed non-uniform weighting.
///
/// The canonical loss for these checks: `Σ w ⊙ y`, whose output gradient is `w`
/// rather than the all-ones a bare `sum` would supply. That distinction is what
/// keeps a Jacobian with constant row sums (softmax, the normalizations) from
/// cancelling to an identically-zero comparison.
pub fn weighted<T: GradcheckScalar>(
    output: &Var<T, MoiraiBackend>,
    w: &Var<T, MoiraiBackend>,
) -> Var<T, MoiraiBackend> {
    sum(&mul(output, w).expect("invariant: test operation succeeds"))
        .expect("invariant: test operation succeeds")
}
