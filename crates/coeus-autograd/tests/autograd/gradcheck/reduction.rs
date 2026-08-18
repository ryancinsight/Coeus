//! Finite-difference checks for the reduction backward passes.
//!
//! # Ties
//!
//! `max_axis`, `min_axis`, `sort` and `topk` are non-differentiable at a tie:
//! the selected index jumps, and a central difference straddling the crossing
//! reports a slope that belongs to neither branch. [`Sampler`] is an irrational
//! rotation of the interval, so its values are pairwise distinct by
//! construction — no two samples in a fixture of this size come within the
//! `≈ 6.1e-6` finite-difference step of each other, and no perturbation can
//! reorder them. This is the property a seeded random fixture would only have
//! with high probability, which is the difference between a test and a flake.

use super::{tensor, weighted, weighting, Sampler};
use coeus_autograd::{
    cumprod, cumsum, gradcheck, log_sum_exp, max_axis, mean, mean_axis, min_axis, norm, norm_p,
    norm_p_axis, prod, sort, std_dev, std_dev_axis, sum, sum_axis, topk, var, var_axis,
};

/// Shape shared by the axis reductions.
const SHAPE: [usize; 2] = [3, 4];

#[test]
fn sum_backward_matches_finite_differences() {
    // The gradient is all-ones, which a broken backward reproduces easily —
    // but a wrong *shape* or a missing broadcast does not.
    let x = tensor(&SHAPE, 0.11);
    gradcheck(&[x], |v| sum(&v[0])).expect("sum backward must match central differences");
}

#[test]
fn mean_backward_matches_finite_differences() {
    // 1/N everywhere; the check is that N is the element count and not an axis
    // length.
    let x = tensor(&SHAPE, 0.13);
    gradcheck(&[x], |v| mean(&v[0])).expect("mean backward must match central differences");
}

#[test]
fn sum_axis_backward_matches_finite_differences() {
    let x = tensor(&SHAPE, 0.17);
    let w = weighting(&[3, 1]);
    gradcheck(&[x], |v| weighted(&sum_axis(&v[0], 1), &w))
        .expect("sum_axis backward must match central differences");
}

#[test]
fn mean_axis_backward_matches_finite_differences() {
    // 1/L over the reduced axis; reducing axis 0 rather than the last one
    // catches a backward that assumed the trailing axis.
    let x = tensor(&SHAPE, 0.19);
    let w = weighting(&[1, 4]);
    gradcheck(&[x], |v| weighted(&mean_axis(&v[0], 0), &w))
        .expect("mean_axis backward must match central differences");
}

#[test]
fn norm_backward_matches_finite_differences() {
    // d||x||₂/dx = x/||x||₂. Sampled away from the origin, where the L2 norm is
    // not differentiable.
    let x = Sampler::positive(0.23).tensor(&SHAPE);
    gradcheck(&[x], |v| norm(&v[0])).expect("norm backward must match central differences");
}

#[test]
fn norm_p_backward_matches_finite_differences_at_p_four() {
    // The general rule is |x|^(p-1)·sign(x)·s^(1/p - 1). p = 4 keeps it smooth
    // through zero while still exercising the exponent arithmetic that p = 2
    // simplifies away.
    let x = tensor(&SHAPE, 0.29);
    gradcheck(&[x], |v| norm_p(&v[0], 4.0))
        .expect("norm_p p=4 backward must match central differences");
}

#[test]
fn norm_p_axis_backward_matches_finite_differences() {
    // Same rule applied per row rather than globally; a backward that reduced
    // over the wrong axis produces a gradient of the right shape and the wrong
    // values.
    let x = Sampler::positive(0.31).tensor(&SHAPE);
    let w = weighting(&[3, 1]);
    gradcheck(&[x], |v| weighted(&norm_p_axis(&v[0], 3.0, 1), &w))
        .expect("norm_p_axis backward must match central differences");
}

#[test]
fn log_sum_exp_backward_matches_finite_differences() {
    // d/dx logsumexp(x) = softmax(x): the gradient of the reduction is the
    // normalized exponential, and a backward that forgot the normalisation is
    // off by the partition function.
    let x = tensor(&SHAPE, 0.37);
    let w = weighting(&[3, 1]);
    gradcheck(&[x], |v| weighted(&log_sum_exp(&v[0], 1), &w))
        .expect("log_sum_exp backward must match central differences");
}

#[test]
fn prod_backward_matches_finite_differences() {
    // d(Πx)/dx_i = Π_{j≠i} x_j. Implemented as total/x_i, which is exact only
    // while no element is zero — the sampler's interval excludes it, and the
    // magnitudes stay near 1 so the product does not underflow.
    let x = Sampler::new(0.41, 0.5, 1.6).tensor(&[6]);
    gradcheck(&[x], |v| prod(&v[0])).expect("prod backward must match central differences");
}

#[test]
fn variance_backward_matches_finite_differences() {
    // The mean is itself a function of every element, so the gradient is
    // 2(x - x̄)/N rather than 2x/N. Unbiased and biased differ by N/(N-1);
    // both are checked.
    let x = tensor(&[6], 0.43);
    gradcheck(std::slice::from_ref(&x), |v| var(&v[0], false))
        .expect("biased variance backward must match central differences");
    gradcheck(&[x], |v| var(&v[0], true))
        .expect("unbiased variance backward must match central differences");
}

#[test]
fn std_dev_backward_matches_finite_differences() {
    // One more chain step than the variance: the 1/(2σ) factor is the part an
    // implementation omits when it reuses the variance backward directly.
    let x = tensor(&[6], 0.47);
    gradcheck(&[x], |v| std_dev(&v[0], true))
        .expect("std_dev backward must match central differences");
}

#[test]
fn var_axis_backward_matches_finite_differences() {
    let x = tensor(&SHAPE, 0.53);
    let w = weighting(&[3, 1]);
    gradcheck(&[x], |v| weighted(&var_axis(&v[0], 1, true), &w))
        .expect("var_axis backward must match central differences");
}

#[test]
fn std_dev_axis_backward_matches_finite_differences() {
    let x = tensor(&SHAPE, 0.59);
    let w = weighting(&[3, 1]);
    gradcheck(&[x], |v| weighted(&std_dev_axis(&v[0], 1, true), &w))
        .expect("std_dev_axis backward must match central differences");
}

#[test]
fn max_axis_backward_matches_finite_differences() {
    // The gradient is a one-hot mask on the arg-max. Samples are pairwise
    // distinct, so the arg-max is stable under the perturbation and the mask is
    // well defined; a backward that routed gradient to every element, or to the
    // wrong index, disagrees immediately.
    let x = tensor(&SHAPE, 0.61);
    let w = weighting(&[3, 1]);
    gradcheck(&[x], |v| weighted(&max_axis(&v[0], 1), &w))
        .expect("max_axis backward must match central differences");
}

#[test]
fn min_axis_backward_matches_finite_differences() {
    let x = tensor(&SHAPE, 0.67);
    let w = weighting(&[3, 1]);
    gradcheck(&[x], |v| weighted(&min_axis(&v[0], 1), &w))
        .expect("min_axis backward must match central differences");
}

#[test]
fn sort_backward_matches_finite_differences() {
    // Sorting is a permutation, so its backward is the inverse permutation. The
    // non-uniform weighting is essential: with a uniform one the loss is
    // permutation-invariant and every arrangement scores the same, which would
    // let an incorrect inverse pass.
    let x = tensor(&SHAPE, 0.71);
    let w = weighting(&SHAPE);
    gradcheck(&[x], |v| weighted(&sort(&v[0], 1, false).0, &w))
        .expect("sort backward must match central differences");
}

#[test]
fn topk_backward_matches_finite_differences() {
    // A partial permutation: the k selected positions receive gradient and the
    // rest receive none. Distinct samples keep the selection stable.
    let x = tensor(&SHAPE, 0.73);
    let w = weighting(&[3, 2]);
    gradcheck(&[x], |v| weighted(&topk(&v[0], 2, 1, true).0, &w))
        .expect("topk backward must match central differences");
}

#[test]
fn cumsum_backward_matches_finite_differences() {
    // The gradient is a reversed cumulative sum — the transpose of the forward's
    // lower-triangular matrix. A backward that reused the forward direction
    // produces the upper-triangular result instead, which agrees only on the
    // first and last elements.
    let x = tensor(&SHAPE, 0.79);
    let w = weighting(&SHAPE);
    gradcheck(&[x], |v| weighted(&cumsum(&v[0], 1), &w))
        .expect("cumsum backward must match central differences");
}

#[test]
fn cumprod_backward_matches_finite_differences() {
    // Each output depends on every earlier element, so the gradient is a sum
    // over suffixes of running products. Samples are bounded away from zero,
    // where the division-based form of that rule breaks down.
    let x = Sampler::new(0.83, 0.5, 1.6).tensor(&[2, 4]);
    let w = weighting(&[2, 4]);
    gradcheck(&[x], |v| weighted(&cumprod(&v[0], 1), &w))
        .expect("cumprod backward must match central differences");
}
