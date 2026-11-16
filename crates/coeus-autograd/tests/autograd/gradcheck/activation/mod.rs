//! Finite-difference checks for the element-wise activation derivatives.
//!
//! Each op here supplies its own `backward_from_*` formula through the generic
//! unary node, so every one is a separate hand-derived claim even though they
//! share a node body.
//!
//! Each check runs at both `f64` and `f32` (see the module documentation for
//! what each scalar width verifies).
//!
//! # Domains and kinks
//!
//! Two input constraints govern the fixtures:
//!
//! * **Domain.** `log`, `sqrt` and `recip` are undefined or singular at `0`;
//!   `asin`, `acos` and `atanh` need `(-1, 1)`; `acosh` needs `x > 1`. Sampling
//!   outside a domain produces `NaN`, and sampling near a singularity produces a
//!   true derivative so large that no finite-difference bound holds — neither
//!   is a defect in the backward. Each check states the interval it samples.
//!
//! * **Kinks.** The rectifier family is non-differentiable at a known point:
//!   `0` for `relu`/`elu`/`selu`/`celu`/`abs`/`prelu`, `±1` for `hardtanh`,
//!   `±λ` for the shrinkage ops, the threshold for `threshold`, the bounds for
//!   `clamp`. A central difference across such a point averages the two
//!   one-sided slopes and disagrees with any correct backward. The fixtures put
//!   the samples in the interior of the branches; the step is `≈ 6.1e-6`, so an
//!   interval whose endpoints clear the kink by `0.1` cannot be crossed.
//!
//! The checks live in leaf modules by activation family: `smooth` for the ops
//! that are smooth across the origin, `elementwise` for the elementary
//! functions, and `branch` for the kinked rectifier family.

mod branch;
mod elementwise;
mod smooth;

use super::{weighted, weighting, GradcheckScalar, Sampler};
use coeus_autograd::{gradcheck, Var};
use coeus_core::{BackendError, MoiraiBackend};

/// Shape every element-wise check uses.
const SHAPE: [usize; 2] = [3, 4];

/// Run `op` on a fixture drawn from `sampler` and require agreement.
///
/// The non-uniform weighting is what makes the reduction non-vacuous; see the
/// module documentation of the parent module.
fn check<T: GradcheckScalar>(
    sampler: &Sampler,
    op: impl Fn(&Var<T, MoiraiBackend>) -> Result<Var<T, MoiraiBackend>, BackendError>,
) {
    let x = sampler.tensor::<T>(&SHAPE);
    let w = weighting::<T>(&SHAPE);
    gradcheck(&[x], |v| {
        let output = op(&v[0]).expect("invariant: gradcheck backend operation succeeds");
        weighted(&output, &w)
    })
    .expect("activation backward must match central differences");
}

/// Samples in `(-0.9, 0.9)`, for ops smooth across the origin.
fn smooth(phase: f64) -> Sampler {
    Sampler::signed(phase)
}

/// Samples in `(0.2, 1.8)`, for ops singular or undefined at `0`.
fn positive(phase: f64) -> Sampler {
    Sampler::positive(phase)
}

/// Samples in `(-0.85, 0.85)`, strictly inside the unit interval.
///
/// For `asin`, `acos` and `atanh`, whose derivatives diverge at `±1`; `0.15`
/// of clearance keeps the derivative below `4` and the second derivative
/// bounded, so the `O(1)`-conditioning assumption behind the derived tolerance
/// holds.
fn unit_interior(phase: f64) -> Sampler {
    Sampler::new(phase, -0.85, 0.85)
}
