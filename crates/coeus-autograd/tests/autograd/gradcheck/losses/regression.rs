//! Finite-difference checks for the residual-based regression losses.

use super::{tensor, GradcheckScalar, PAIR_SHAPE};
use coeus_autograd::{gradcheck, huber_loss, l1_loss, smooth_l1_loss, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

/// Residuals `pred - target` for the piecewise regression losses.
///
/// Chosen so that every `|r|` clears each break by a wide margin:
///
/// | break            | closest `|r|` | clearance |
/// |------------------|---------------|-----------|
/// | `0` (l1 kink)    | `0.22`        | `0.22`    |
/// | `0.5` (`beta`)   | `0.31`        | `0.19`    |
/// | `0.7` (`delta`)  | `0.83`        | `0.13`    |
///
/// Both branches of each piecewise loss are exercised: `0.22` and `0.31` fall
/// in the quadratic region of `smooth_l1(beta = 0.5)`, the rest in the linear
/// one; `0.22`, `0.31` in the quadratic region of `huber(delta = 0.7)`, the rest
/// linear. A test that only reached one branch would leave the other's
/// derivative unverified.
const RESIDUALS: [f64; 6] = [0.83, -0.31, 1.27, -0.94, 0.22, -1.55];

/// `beta` for `smooth_l1_loss`; see [`RESIDUALS`] for the clearance.
const SMOOTH_L1_BETA: f64 = 0.5;

/// `delta` for `huber_loss`; see [`RESIDUALS`] for the clearance.
const HUBER_DELTA: f64 = 0.7;

/// A constant target placed exactly `residuals` below `pred`.
///
/// Building the target from the prediction — rather than sampling both — is
/// what makes the residual at the evaluation point a known quantity, and
/// therefore what makes the clearance from each kink a derivation rather than a
/// hope.
fn target_below<T: GradcheckScalar>(
    pred: &Tensor<T, MoiraiBackend>,
    residuals: &[f64],
) -> Var<T, MoiraiBackend> {
    let values: Vec<T> = pred
        .as_slice()
        .iter()
        .zip(residuals)
        .map(|(&p, &r)| p - <T as coeus_core::FloatElement>::from_f64(r))
        .collect();
    Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on(
            pred.shape().to_vec(),
            &values,
            &MoiraiBackend::new(),
        )
        .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds")
}

fn l1_loss_case<T: GradcheckScalar>() {
    // d/dpred mean|pred - target| = sign(pred - target)/N. The residuals carry
    // both signs, so a backward that dropped the sign — or applied `abs` to the
    // gradient — disagrees on the negative entries.
    let pred = tensor::<T>(&PAIR_SHAPE, 0.19);
    let target = target_below(&pred, &RESIDUALS);

    gradcheck(&[pred], |v| {
        l1_loss(&v[0], &target).expect("invariant: test operation succeeds")
    })
    .expect("l1_loss backward must match central differences");
}

#[test]
fn l1_loss_backward_matches_finite_differences() {
    l1_loss_case::<f64>();
    l1_loss_case::<f32>();
}

fn smooth_l1_loss_case<T: GradcheckScalar>() {
    // Exercises both branches: r/beta inside the quadratic region, sign(r)
    // outside it. A backward that used the wrong branch boundary, or forgot the
    // 1/beta scaling of the quadratic arm, fails on the entries that straddle.
    let pred = tensor::<T>(&PAIR_SHAPE, 0.43);
    let target = target_below(&pred, &RESIDUALS);

    gradcheck(&[pred], |v| {
        smooth_l1_loss(
            &v[0],
            &target,
            <T as coeus_core::FloatElement>::from_f64(SMOOTH_L1_BETA),
        )
        .expect("invariant: test operation succeeds")
    })
    .expect("smooth_l1_loss backward must match central differences");
}

#[test]
fn smooth_l1_loss_backward_matches_finite_differences() {
    smooth_l1_loss_case::<f64>();
    smooth_l1_loss_case::<f32>();
}

fn huber_loss_case<T: GradcheckScalar>() {
    // Huber differs from smooth_l1 by a factor of delta on the quadratic arm;
    // conflating the two is the classic implementation error, and it shows up
    // only on the entries inside |r| < delta.
    let pred = tensor::<T>(&PAIR_SHAPE, 0.61);
    let target = target_below(&pred, &RESIDUALS);

    gradcheck(&[pred], |v| {
        huber_loss(
            &v[0],
            &target,
            <T as coeus_core::FloatElement>::from_f64(HUBER_DELTA),
        )
        .expect("huber_loss shapes agree")
    })
    .expect("huber_loss backward must match central differences");
}

#[test]
fn huber_loss_backward_matches_finite_differences() {
    huber_loss_case::<f64>();
    huber_loss_case::<f32>();
}
