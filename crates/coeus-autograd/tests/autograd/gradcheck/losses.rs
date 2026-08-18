//! Finite-difference checks for the loss backward passes.
//!
//! # Placing the inputs off the kinks
//!
//! Most of these losses are piecewise: `l1` and `smooth_l1` break where the
//! residual is `0` or `±beta`, the margin family breaks where the hinge
//! argument crosses `0`. A central difference straddles its evaluation point, so
//! at a break it averages the two one-sided derivatives and disagrees with any
//! correct backward — the check would report a defect that is not there.
//!
//! The residual-based losses therefore take an explicitly chosen residual
//! vector rather than a sampled one, so the distance from every break is a
//! stated number rather than an accident of the generator. The finite-difference
//! step is `ε^(1/3)·max(|x|,1) ≈ 6.1e-6` in `f64`, so a residual `0.1` away from
//! a break is four orders of magnitude clear of it: no perturbation can cross.

use super::{tensor, weighted, weighting, Sampler, T64};
use coeus_autograd::{
    bce_with_logits, binary_cross_entropy, cosine_embedding_loss, cosine_similarity, gradcheck,
    huber_loss, kl_divergence, l1_loss, margin_ranking_loss, multi_label_margin_loss, multi_margin,
    nll_loss, pairwise_distance, poisson_nll, smooth_l1_loss, soft_margin, Var,
};
use coeus_core::MoiraiBackend;

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

/// Shape shared by the element-wise regression losses.
const PAIR_SHAPE: [usize; 2] = [2, 3];

/// A constant target placed exactly `residuals` below `pred`.
///
/// Building the target from the prediction — rather than sampling both — is
/// what makes the residual at the evaluation point a known quantity, and
/// therefore what makes the clearance from each kink a derivation rather than a
/// hope.
fn target_below(pred: &T64, residuals: &[f64]) -> Var<f64, MoiraiBackend> {
    let values: Vec<f64> = pred
        .as_slice()
        .iter()
        .zip(residuals)
        .map(|(&p, &r)| p - r)
        .collect();
    Var::new(
        T64::from_slice_on(pred.shape().to_vec(), &values, &MoiraiBackend::new()),
        false,
    )
}

#[test]
fn l1_loss_backward_matches_finite_differences() {
    // d/dpred mean|pred - target| = sign(pred - target)/N. The residuals carry
    // both signs, so a backward that dropped the sign — or applied `abs` to the
    // gradient — disagrees on the negative entries.
    let pred = tensor(&PAIR_SHAPE, 0.19);
    let target = target_below(&pred, &RESIDUALS);

    gradcheck(&[pred], |v| l1_loss(&v[0], &target))
        .expect("l1_loss backward must match central differences");
}

#[test]
fn smooth_l1_loss_backward_matches_finite_differences() {
    // Exercises both branches: r/beta inside the quadratic region, sign(r)
    // outside it. A backward that used the wrong branch boundary, or forgot the
    // 1/beta scaling of the quadratic arm, fails on the entries that straddle.
    let pred = tensor(&PAIR_SHAPE, 0.43);
    let target = target_below(&pred, &RESIDUALS);

    gradcheck(&[pred], |v| smooth_l1_loss(&v[0], &target, SMOOTH_L1_BETA))
        .expect("smooth_l1_loss backward must match central differences");
}

#[test]
fn huber_loss_backward_matches_finite_differences() {
    // Huber differs from smooth_l1 by a factor of delta on the quadratic arm;
    // conflating the two is the classic implementation error, and it shows up
    // only on the entries inside |r| < delta.
    let pred = tensor(&PAIR_SHAPE, 0.61);
    let target = target_below(&pred, &RESIDUALS);

    gradcheck(&[pred], |v| {
        huber_loss(&v[0], &target, HUBER_DELTA).expect("huber_loss shapes agree")
    })
    .expect("huber_loss backward must match central differences");
}

#[test]
fn binary_cross_entropy_backward_matches_finite_differences() {
    // d/dp mean(-[y log p + (1-y) log(1-p)]) = (p - y)/(p(1-p)N). Both p and y
    // stay inside (0.1, 0.9), so neither log term approaches its singularity and
    // the derived tolerance holds.
    let pred = Sampler::probability(0.29).tensor(&PAIR_SHAPE);
    let target = Sampler::probability(0.73).constant(&PAIR_SHAPE);

    gradcheck(&[pred], |v| binary_cross_entropy(&v[0], &target, 1e-12))
        .expect("binary_cross_entropy backward must match central differences");
}

/// Logits for [`bce_with_logits_backward_matches_finite_differences`].
///
/// The forward is written in the numerically stable `max(z,0) - z·y +
/// log(1+exp(-|z|))` form, which is smooth despite containing `max` and `abs` —
/// the two branches agree in value and slope at `z = 0`. These logits still
/// carry both signs and stay clear of `0`, so each branch is evaluated in its
/// own interior and a branch-selection error cannot hide behind the agreement
/// at the seam.
const LOGITS: [f64; 6] = [1.35, -0.82, 0.47, -1.91, 2.06, -0.55];

#[test]
fn bce_with_logits_backward_matches_finite_differences() {
    // d/dz mean(softplus(z) - z·y) = (sigmoid(z) - y)/N.
    let logits = T64::from_slice_on(PAIR_SHAPE.to_vec(), &LOGITS, &MoiraiBackend::new());
    let target = Sampler::probability(0.37).constant(&PAIR_SHAPE);

    gradcheck(&[logits], |v| bce_with_logits(&v[0], &target))
        .expect("bce_with_logits backward must match central differences");
}

#[test]
fn kl_divergence_backward_matches_finite_differences() {
    // mean(target·(log target - input)) is linear in `input`, so the gradient is
    // the constant -target/N. Linear is not trivial here: the check catches a
    // missing 1/N, a sign error, or a gradient that accidentally depends on
    // `input`.
    let log_q = Sampler::new(0.23, -2.5, -0.3).tensor(&PAIR_SHAPE);
    let p = Sampler::positive(0.59).constant(&PAIR_SHAPE);

    gradcheck(&[log_q], |v| kl_divergence(&v[0], &p))
        .expect("kl_divergence backward must match central differences");
}

#[test]
fn poisson_nll_backward_matches_finite_differences() {
    // mean(exp(input) - target·input); d/dinput = (exp(input) - target)/N. The
    // log-rate stays in (-1.2, 1.2) so exp() is O(1) and the loss magnitude does
    // not inflate the derived tolerance.
    let log_rate = Sampler::new(0.31, -1.2, 1.2).tensor(&PAIR_SHAPE);
    let counts = Sampler::positive(0.67).constant(&PAIR_SHAPE);

    gradcheck(&[log_rate], |v| poisson_nll(&v[0], &counts))
        .expect("poisson_nll backward must match central differences");
}

/// Class labels in `{-1, +1}` for the margin losses.
const SIGNS: [f64; 6] = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0];

#[test]
fn soft_margin_backward_matches_finite_differences() {
    // mean log(1 + exp(-y·x)) is smooth everywhere; d/dx = -y·sigmoid(-y·x)/N.
    // Both label signs appear, so a backward that dropped the `-y` factor fails
    // on the negative half.
    let input = tensor(&PAIR_SHAPE, 0.47);
    let labels = Var::new(
        T64::from_slice_on(PAIR_SHAPE.to_vec(), &SIGNS, &MoiraiBackend::new()),
        false,
    );

    gradcheck(&[input], |v| soft_margin(&v[0], &labels))
        .expect("soft_margin backward must match central differences");
}

#[test]
fn nll_loss_backward_matches_finite_differences() {
    // -mean(log_probs[i, target_i]) is linear in log_probs, so the gradient is a
    // scaled one-hot mask. The check verifies the mask lands on the right
    // column: a transposed or off-by-one index produces a gradient in the wrong
    // place, which finite differences localise exactly.
    let log_probs = Sampler::new(0.17, -2.4, -0.4).tensor(&[3, 4]);
    let targets = [2usize, 0, 3];

    gradcheck(&[log_probs], |v| nll_loss(&v[0], &targets))
        .expect("nll_loss backward must match central differences");
}

/// Hinge arguments for [`margin_ranking_loss_backward_matches_finite_differences`].
///
/// The loss is `mean(max(0, -y·(a - b) + margin))`, which breaks where the
/// hinge argument is `0`. `a - b` is fixed by construction below and the labels
/// are [`SIGNS`], so with `margin = 0.3` the hinge arguments are
/// `-y·(a-b) + 0.3`: comfortably active for some rows and comfortably inactive
/// for others, none within `0.2` of the break. Both the active and the inactive
/// branch are therefore verified.
const RANKING_GAPS: [f64; 4] = [0.9, 0.9, -1.4, -1.4];

/// Labels paired with [`RANKING_GAPS`].
const RANKING_LABELS: [f64; 4] = [1.0, -1.0, 1.0, -1.0];

/// Margin for the ranking loss; see [`RANKING_GAPS`].
const RANKING_MARGIN: f64 = 0.3;

#[test]
fn margin_ranking_loss_backward_matches_finite_differences() {
    // Hinge arguments are -y(a-b)+0.3 = {-0.6, 1.2, 1.7, -1.1}: two active, two
    // inactive, each at least 0.6 from the break. Both inputs are
    // differentiated, so the check covers the -y/N and +y/N rules together and
    // a swapped pair surfaces as a sign mismatch.
    let backend = MoiraiBackend::new();
    let b = Sampler::signed(0.41).tensor(&[4]);
    let a_values: Vec<f64> = b
        .as_slice()
        .iter()
        .zip(&RANKING_GAPS)
        .map(|(&bv, &gap)| bv + gap)
        .collect();
    let a = T64::from_slice_on([4], &a_values, &backend);

    gradcheck(&[a, b], |v| {
        margin_ranking_loss(&v[0], &v[1], &RANKING_LABELS, RANKING_MARGIN)
    })
    .expect("margin_ranking_loss backward must match central differences");
}

/// Scores for [`multi_margin_backward_matches_finite_differences_at_p_one`].
///
/// With `margin = 0.5` the hinge arguments are `m_j = margin - x[y] + x[j]`:
///
/// | row | target | `m_j`                  | nearest break |
/// |-----|--------|------------------------|---------------|
/// | 0   | `0`    | `[0.5, 0.3, -1.5]`     | `0.3`         |
/// | 1   | `1`    | `[-1.6, 0.5, 0.2]`     | `0.2`         |
///
/// Each row has an active and an inactive non-target hinge, so both sides of
/// the rectifier are exercised, and the closest approach to a break is `0.2` —
/// four orders of magnitude beyond the finite-difference step.
const MULTI_MARGIN_SCORES: [f64; 6] = [1.4, 1.2, -0.6, -0.2, 1.9, 1.6];

#[test]
fn multi_margin_backward_matches_finite_differences_at_p_one() {
    // p = 1 makes each per-pair term a bare hinge, so the loss is piecewise
    // linear and every gradient component comes from the active set alone.
    //
    // N = 2 here deliberately. Every pre-existing test of this op used a single
    // row, which is the one batch size at which its per-row target selection
    // and its row-wise margin subtraction both happen to be shape-correct; see
    // the comments at those two sites.
    let scores = T64::from_slice_on([2, 3], &MULTI_MARGIN_SCORES, &MoiraiBackend::new());
    let targets = [0usize, 1];

    gradcheck(&[scores], |v| multi_margin(&v[0], &targets, 1.0, 0.5))
        .expect("multi_margin p=1 backward must match central differences");
}

/// Scores for [`multi_label_margin_backward_matches_finite_differences`].
///
/// The loss sums `max(0, 1 - (x[t] - x[j]))` over valid targets `t` and
/// `j != t`. Row 0 has the single target `{0}` and row 1 the pair `{1, 3}`, so
/// the hinge arguments are
///
/// | row | `t` | `1 - (x[t] - x[j])` over `j != t` | nearest break |
/// |-----|-----|-----------------------------------|---------------|
/// | 0   | `0` | `[-0.2, 0.3, -1.0]`               | `0.2`         |
/// | 1   | `1` | `[-0.1, -1.1, 0.25]`              | `0.1`         |
/// | 1   | `3` | `[0.65, 1.75, -0.35]`             | `0.35`        |
///
/// Every row mixes active and inactive hinges and the closest approach to a
/// break is `0.1`. An earlier fixture here placed `1 - (x[1] - x[3])` at
/// exactly `0`: the analytic gradient counted that hinge as fully active while
/// the central difference, straddling the break, saw half of it — a clean
/// factor-of-two disagreement that was a defect in the fixture, not in the op.
const MULTI_LABEL_SCORES: [f64; 8] = [0.9, -0.3, 0.2, -1.1, 0.4, 1.5, -0.6, 0.75];

#[test]
fn multi_label_margin_backward_matches_finite_differences() {
    // The -1 padding must be skipped: a backward that treated it as class 0
    // would put gradient on the wrong column. Row 1 carries two valid targets,
    // so the per-row gather is exercised on more than one column, and N = 2
    // covers the batch dimension this op's other tests do not.
    let scores = T64::from_slice_on([2, 4], &MULTI_LABEL_SCORES, &MoiraiBackend::new());
    let targets = [0isize, -1, -1, -1, 1, 3, -1, -1];

    gradcheck(&[scores], |v| multi_label_margin_loss(&v[0], &targets))
        .expect("multi_label_margin_loss backward must match central differences");
}

#[test]
fn cosine_similarity_backward_matches_finite_differences() {
    // <a,b>/(||a||·||b||) is smooth away from zero rows; both operands are
    // sampled strictly positive-magnitude, so neither norm approaches the eps
    // clamp and the full norm-derivative path — not the clamped constant one —
    // is the branch under test.
    let x1 = Sampler::new(0.13, 0.3, 1.6).tensor(&[3, 4]);
    let x2 = Sampler::new(0.79, -1.6, -0.3).tensor(&[3, 4]);
    let w = weighting(&[3]);

    gradcheck(&[x1, x2], |v| {
        weighted(&cosine_similarity(&v[0], &v[1], 1, 1e-8), &w)
    })
    .expect("cosine_similarity backward must match central differences");
}

#[test]
fn cosine_embedding_loss_backward_matches_finite_differences() {
    // y = +1 rows contribute 1 - cos, y = -1 rows contribute max(0, cos -
    // margin). With margin = -0.5 and the operands below anti-aligned, the
    // y = -1 hinge is firmly active, so both the smooth and the hinge branch are
    // covered without either sitting on its break.
    let x1 = Sampler::new(0.13, 0.3, 1.6).tensor(&[3, 4]);
    let x2 = Sampler::new(0.79, -1.6, -0.3).tensor(&[3, 4]);
    let labels = [1.0f64, -1.0, 1.0];

    gradcheck(&[x1, x2], |v| {
        cosine_embedding_loss(&v[0], &v[1], &labels, -0.5)
    })
    .expect("cosine_embedding_loss backward must match central differences");
}

#[test]
fn pairwise_distance_backward_matches_finite_differences() {
    // p = 2 keeps |x1 - x2 + eps|^p smooth through zero, so this checks the
    // s^(1/p - 1) chain factor without a kink in the way. The operands are drawn
    // from disjoint intervals, so every difference is bounded away from zero and
    // the row sums stay well clear of the origin where the p-norm is
    // non-differentiable.
    let x1 = Sampler::new(0.23, 0.4, 1.7).tensor(&[3, 4]);
    let x2 = Sampler::new(0.61, -1.7, -0.4).tensor(&[3, 4]);
    let w = weighting(&[3]);

    gradcheck(&[x1, x2], |v| {
        weighted(&pairwise_distance(&v[0], &v[1], 2.0, 1e-6), &w)
    })
    .expect("pairwise_distance p=2 backward must match central differences");
}

#[test]
fn pairwise_distance_backward_matches_finite_differences_at_p_three() {
    // p = 3 exercises the general |d|^(p-1)·sign(d) path rather than the p = 2
    // special case, where the sign factor cancels and an implementation can be
    // wrong without the p = 2 check noticing.
    let x1 = Sampler::new(0.37, 0.4, 1.7).tensor(&[2, 3]);
    let x2 = Sampler::new(0.83, -1.7, -0.4).tensor(&[2, 3]);
    let w = weighting(&[2]);

    gradcheck(&[x1, x2], |v| {
        weighted(&pairwise_distance(&v[0], &v[1], 3.0, 1e-6), &w)
    })
    .expect("pairwise_distance p=3 backward must match central differences");
}
