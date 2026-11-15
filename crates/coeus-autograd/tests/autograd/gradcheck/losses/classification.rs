//! Finite-difference checks for the log-probability classification losses.

use super::{tensor, GradcheckScalar, Sampler, PAIR_SHAPE};
use coeus_autograd::{
    bce_with_logits, binary_cross_entropy, gradcheck, kl_divergence, nll_loss, poisson_nll,
    soft_margin, Var,
};
use coeus_core::MoiraiBackend;
use coeus_nn::cross_entropy_loss;
use coeus_tensor::Tensor;

/// Logits for [`bce_with_logits_backward_matches_finite_differences`].
///
/// The forward is written in the numerically stable `max(z,0) - z·y +
/// log(1+exp(-|z|))` form, which is smooth despite containing `max` and `abs` —
/// the two branches agree in value and slope at `z = 0`. These logits still
/// carry both signs and stay clear of `0`, so each branch is evaluated in its
/// own interior and a branch-selection error cannot hide behind the agreement
/// at the seam.
const LOGITS: [f64; 6] = [1.35, -0.82, 0.47, -1.91, 2.06, -0.55];

/// Class labels in `{-1, +1}` for the margin losses.
const SIGNS: [f64; 6] = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

fn binary_cross_entropy_case<T: GradcheckScalar>() {
    // d/dp mean(-[y log p + (1-y) log(1-p)]) = (p - y)/(p(1-p)N). Both p and y
    // stay inside (0.1, 0.9), so neither log term approaches its singularity and
    // the derived tolerance holds.
    let pred = Sampler::probability(0.29).tensor::<T>(&PAIR_SHAPE);
    let target = Sampler::probability(0.73).constant::<T>(&PAIR_SHAPE);

    gradcheck(&[pred], |v| {
        binary_cross_entropy(
            &v[0],
            &target,
            <T as coeus_core::FloatElement>::from_f64(1e-12),
        )
        .expect("invariant: test operation succeeds")
    })
    .expect("binary_cross_entropy backward must match central differences");
}

#[test]
fn binary_cross_entropy_backward_matches_finite_differences() {
    binary_cross_entropy_case::<f64>();
    binary_cross_entropy_case::<f32>();
}

fn bce_with_logits_case<T: GradcheckScalar>() {
    // d/dz mean(softplus(z) - z·y) = (sigmoid(z) - y)/N.
    let logit_values: Vec<T> = LOGITS
        .into_iter()
        .map(<T as coeus_core::FloatElement>::from_f64)
        .collect();
    let logits = Tensor::<T, MoiraiBackend>::from_slice_on(
        PAIR_SHAPE.to_vec(),
        &logit_values,
        &MoiraiBackend::new(),
    )
    .expect("invariant: test backend operation succeeds");
    let target = Sampler::probability(0.37).constant::<T>(&PAIR_SHAPE);

    gradcheck(&[logits], |v| {
        bce_with_logits(&v[0], &target).expect("invariant: test operation succeeds")
    })
    .expect("bce_with_logits backward must match central differences");
}

#[test]
fn bce_with_logits_backward_matches_finite_differences() {
    bce_with_logits_case::<f64>();
    bce_with_logits_case::<f32>();
}

fn kl_divergence_case<T: GradcheckScalar>() {
    // mean(target·(log target - input)) is linear in `input`, so the gradient is
    // the constant -target/N. Linear is not trivial here: the check catches a
    // missing 1/N, a sign error, or a gradient that accidentally depends on
    // `input`.
    let log_q = Sampler::new(0.23, -2.5, -0.3).tensor::<T>(&PAIR_SHAPE);
    let p = Sampler::positive(0.59).constant::<T>(&PAIR_SHAPE);

    gradcheck(&[log_q], |v| {
        kl_divergence(&v[0], &p).expect("invariant: test operation succeeds")
    })
    .expect("kl_divergence backward must match central differences");
}

#[test]
fn kl_divergence_backward_matches_finite_differences() {
    kl_divergence_case::<f64>();
    kl_divergence_case::<f32>();
}

fn poisson_nll_case<T: GradcheckScalar>() {
    // mean(exp(input) - target·input); d/dinput = (exp(input) - target)/N. The
    // log-rate stays in (-1.2, 1.2) so exp() is O(1) and the loss magnitude does
    // not inflate the derived tolerance.
    let log_rate = Sampler::new(0.31, -1.2, 1.2).tensor::<T>(&PAIR_SHAPE);
    let counts = Sampler::positive(0.67).constant::<T>(&PAIR_SHAPE);

    gradcheck(&[log_rate], |v| {
        poisson_nll(&v[0], &counts).expect("invariant: test operation succeeds")
    })
    .expect("poisson_nll backward must match central differences");
}

#[test]
fn poisson_nll_backward_matches_finite_differences() {
    poisson_nll_case::<f64>();
    poisson_nll_case::<f32>();
}

fn soft_margin_case<T: GradcheckScalar>() {
    // mean log(1 + exp(-y·x)) is smooth everywhere; d/dx = -y·sigmoid(-y·x)/N.
    // Both label signs appear, so a backward that dropped the `-y` factor fails
    // on the negative half.
    let input = tensor::<T>(&PAIR_SHAPE, 0.47);
    let sign_values: Vec<T> = SIGNS
        .into_iter()
        .map(<T as coeus_core::FloatElement>::from_f64)
        .collect();
    let labels = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on(
            PAIR_SHAPE.to_vec(),
            &sign_values,
            &MoiraiBackend::new(),
        )
        .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    gradcheck(&[input], |v| {
        soft_margin(&v[0], &labels).expect("invariant: test operation succeeds")
    })
    .expect("soft_margin backward must match central differences");
}

#[test]
fn soft_margin_backward_matches_finite_differences() {
    soft_margin_case::<f64>();
    soft_margin_case::<f32>();
}

fn nll_loss_case<T: GradcheckScalar>() {
    // -mean(log_probs[i, target_i]) is linear in log_probs, so the gradient is a
    // scaled one-hot mask. The check verifies the mask lands on the right
    // column: a transposed or off-by-one index produces a gradient in the wrong
    // place, which finite differences localise exactly.
    let log_probs = Sampler::new(0.17, -2.4, -0.4).tensor::<T>(&[3, 4]);
    let targets = [2usize, 0, 3];

    gradcheck(&[log_probs], |v| {
        nll_loss(&v[0], &targets).expect("invariant: test operation succeeds")
    })
    .expect("nll_loss backward must match central differences");
}

#[test]
fn nll_loss_backward_matches_finite_differences() {
    nll_loss_case::<f64>();
    nll_loss_case::<f32>();
}

fn cross_entropy_case<T: GradcheckScalar>()
where
    MoiraiBackend: coeus_ops::CrossEntropyOps<T>,
{
    // Softmax-plus-log is smooth everywhere, so any finite logits with
    // in-range targets give a well-posed central difference. The targets
    // mirror `nll_loss_case` so the two classification losses share one
    // fixture shape; the check goes through the `coeus_nn` forward wrapper
    // so provider forward, target preparation, and graph assembly are all
    // inside the differentiated closure.
    let logits = tensor::<T>(&[3, 4], 0.29);
    let targets = [2usize, 0, 3];

    gradcheck(&[logits], |v| {
        cross_entropy_loss(&v[0], &targets)
            .expect("invariant: in-range targets satisfy the forward contract")
    })
    .expect("cross_entropy_loss backward must match central differences");
}

#[test]
fn cross_entropy_backward_matches_finite_differences() {
    cross_entropy_case::<f64>();
    cross_entropy_case::<f32>();
}
