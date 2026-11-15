//! Finite-difference checks for the margin-based ranking losses.

use super::{GradcheckScalar, Sampler};
use coeus_autograd::{gradcheck, margin_ranking_loss, multi_label_margin_loss, multi_margin};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

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

fn margin_ranking_loss_case<T: GradcheckScalar>() {
    // Hinge arguments are -y(a-b)+0.3 = {-0.6, 1.2, 1.7, -1.1}: two active, two
    // inactive, each at least 0.6 from the break. Both inputs are
    // differentiated, so the check covers the -y/N and +y/N rules together and
    // a swapped pair surfaces as a sign mismatch.
    let backend = MoiraiBackend::new();
    let b = Sampler::signed(0.41).tensor::<T>(&[4]);
    let a_values: Vec<T> = b
        .as_slice()
        .iter()
        .zip(&RANKING_GAPS)
        .map(|(&bv, &gap)| bv + <T as coeus_core::FloatElement>::from_f64(gap))
        .collect();
    let a = Tensor::<T, MoiraiBackend>::from_slice_on([4], &a_values, &backend)
        .expect("invariant: test backend operation succeeds");
    let labels: Vec<T> = RANKING_LABELS
        .into_iter()
        .map(<T as coeus_core::FloatElement>::from_f64)
        .collect();

    gradcheck(&[a, b], |v| {
        margin_ranking_loss(
            &v[0],
            &v[1],
            &labels,
            <T as coeus_core::FloatElement>::from_f64(RANKING_MARGIN),
        )
        .expect("invariant: test operation succeeds")
    })
    .expect("margin_ranking_loss backward must match central differences");
}

#[test]
fn margin_ranking_loss_backward_matches_finite_differences() {
    margin_ranking_loss_case::<f64>();
    margin_ranking_loss_case::<f32>();
}

fn multi_margin_case<T: GradcheckScalar>() {
    // p = 1 makes each per-pair term a bare hinge, so the loss is piecewise
    // linear and every gradient component comes from the active set alone.
    //
    // N = 2 here deliberately. Every pre-existing test of this op used a single
    // row, which is the one batch size at which its per-row target selection
    // and its row-wise margin subtraction both happen to be shape-correct; see
    // the comments at those two sites.
    let score_values: Vec<T> = MULTI_MARGIN_SCORES
        .into_iter()
        .map(<T as coeus_core::FloatElement>::from_f64)
        .collect();
    let scores =
        Tensor::<T, MoiraiBackend>::from_slice_on([2, 3], &score_values, &MoiraiBackend::new())
            .expect("invariant: test backend operation succeeds");
    let targets = [0usize, 1];

    gradcheck(&[scores], |v| {
        multi_margin(
            &v[0],
            &targets,
            <T as coeus_core::FloatElement>::from_f64(1.0),
            <T as coeus_core::FloatElement>::from_f64(0.5),
        )
        .expect("invariant: test operation succeeds")
    })
    .expect("multi_margin p=1 backward must match central differences");
}

#[test]
fn multi_margin_backward_matches_finite_differences_at_p_one() {
    multi_margin_case::<f64>();
    multi_margin_case::<f32>();
}

fn multi_label_margin_case<T: GradcheckScalar>() {
    // The -1 padding must be skipped: a backward that treated it as class 0
    // would put gradient on the wrong column. Row 1 carries two valid targets,
    // so the per-row gather is exercised on more than one column, and N = 2
    // covers the batch dimension this op's other tests do not.
    let score_values: Vec<T> = MULTI_LABEL_SCORES
        .into_iter()
        .map(<T as coeus_core::FloatElement>::from_f64)
        .collect();
    let scores =
        Tensor::<T, MoiraiBackend>::from_slice_on([2, 4], &score_values, &MoiraiBackend::new())
            .expect("invariant: test backend operation succeeds");
    let targets = [0isize, -1, -1, -1, 1, 3, -1, -1];

    gradcheck(&[scores], |v| {
        multi_label_margin_loss(&v[0], &targets).expect("invariant: test operation succeeds")
    })
    .expect("multi_label_margin_loss backward must match central differences");
}

#[test]
fn multi_label_margin_backward_matches_finite_differences() {
    multi_label_margin_case::<f64>();
    multi_label_margin_case::<f32>();
}
