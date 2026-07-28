// ── Tracked variance / standard deviation ──
//
// Composed from existing tracked ops (`mean`/`mean_axis`, `sub`, `mul`,
// `sum`/`sum_axis`, `scalar_div`, `sqrt`, `reshape`) so gradients flow through
// the DAG automatically without a bespoke backward node — the same pattern as
// `log_sum_exp`. Mirrors the untracked `coeus_ops::reduction::variance`
// surface (two-pass `E[(x − μ)²]`, Bessel-corrected when `unbiased`), and the
// pair functions (`var_mean`, `var_mean_axis`) are the SSOT: singleton and
// std variants delegate so the mean subgraph is never duplicated.
//
// Gradient (by composition): for `v = Σ(x_i − μ)² / (n − c)` the μ-terms
// cancel (`Σ(x_j − μ) = 0`), so `dv/dx_i = 2(x_i − μ)/(n − c)` — the autograd
// graph reproduces this exactly; the parity tests pin it against PyTorch.

use crate::var::Var;
use coeus_core::Float;

/// Tracked variance and mean over all elements.
///
/// Returns `(variance, mean)`, both `[1]`-shaped. `unbiased = true` divides
/// by `N − 1` (PyTorch default); `false` divides by `N` (population).
///
/// # Panics
/// Panics if `a` is empty, or if `unbiased` and `a` has a single element
/// (variance denominator would be zero).
#[must_use]
pub fn var_mean<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    unbiased: bool,
) -> Result<(Var<T, B>, Var<T, B>), B::Error> {
    let n = a.tensor.numel();
    assert!(n > 0, "var_mean: empty tensor has no variance");
    let denom = n - usize::from(unbiased);
    assert!(
        denom > 0,
        "var_mean: unbiased variance of a single element divides by zero"
    );

    let mu = crate::ops::arithmetic::mean(a)?;
    // Flatten so the deviation broadcasts against the scalar mean uniformly.
    let flat = crate::ops::shape::reshape(a, vec![n])?;
    let dev = crate::ops::arithmetic::sub(&flat, &mu)?;
    let sq = crate::ops::arithmetic::mul(&dev, &dev)?;
    let ssum = crate::ops::arithmetic::sum(&sq)?;
    let v = crate::ops::arithmetic::scalar_div(&ssum, T::from_f64(denom as f64))?;
    Ok((v, mu))
}

/// Tracked variance over all elements (`torch.var`). See [`var_mean`].
#[must_use]
#[inline]
pub fn var<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    unbiased: bool,
) -> Result<Var<T, B>, B::Error> {
    Ok(var_mean(a, unbiased)?.0)
}

/// Tracked standard deviation and mean over all elements (`torch.std_mean`).
#[must_use]
#[inline]
pub fn std_mean<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    unbiased: bool,
) -> Result<(Var<T, B>, Var<T, B>), B::Error> {
    let (v, mu) = var_mean(a, unbiased)?;
    Ok((crate::ops::activation::sqrt(&v)?, mu))
}

/// Tracked standard deviation over all elements (`torch.std`).
#[must_use]
#[inline]
pub fn std_dev<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    unbiased: bool,
) -> Result<Var<T, B>, B::Error> {
    Ok(std_mean(a, unbiased)?.0)
}

/// Tracked variance and mean along `axis` (`torch.var_mean(dim=axis)`).
///
/// Both outputs keep `axis` as a size-1 dimension (the tracked reduction
/// convention, matching `keepdim=True`), so they broadcast against the input.
///
/// # Panics
/// Panics if `axis` is out of range, or if `unbiased` and the axis extent is
/// 1 (denominator would be zero).
#[must_use]
pub fn var_mean_axis<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
    unbiased: bool,
) -> Result<(Var<T, B>, Var<T, B>), B::Error> {
    let shape = a.tensor.shape();
    assert!(
        axis < shape.len(),
        "var_mean_axis: axis {axis} out of range for rank {}",
        shape.len()
    );
    let extent = shape[axis];
    let denom = extent - usize::from(unbiased);
    assert!(
        denom > 0,
        "var_mean_axis: unbiased variance along axis {axis} of extent {extent} divides by zero"
    );

    let mu = crate::ops::arithmetic::mean_axis(a, axis)?;
    let dev = crate::ops::arithmetic::sub(a, &mu)?;
    let sq = crate::ops::arithmetic::mul(&dev, &dev)?;
    let ssum = crate::ops::arithmetic::sum_axis(&sq, axis)?;
    let v = crate::ops::arithmetic::scalar_div(&ssum, T::from_f64(denom as f64))?;
    Ok((v, mu))
}

/// Tracked variance along `axis` (`torch.var(dim=axis, keepdim=True)`).
#[must_use]
#[inline]
pub fn var_axis<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
    unbiased: bool,
) -> Result<Var<T, B>, B::Error> {
    Ok(var_mean_axis(a, axis, unbiased)?.0)
}

/// Tracked standard deviation and mean along `axis` (`torch.std_mean(dim=axis)`).
#[must_use]
#[inline]
pub fn std_mean_axis<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
    unbiased: bool,
) -> Result<(Var<T, B>, Var<T, B>), B::Error> {
    let (v, mu) = var_mean_axis(a, axis, unbiased)?;
    Ok((crate::ops::activation::sqrt(&v)?, mu))
}

/// Tracked standard deviation along `axis` (`torch.std(dim=axis, keepdim=True)`).
#[must_use]
#[inline]
pub fn std_dev_axis<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    a: &Var<T, B>,
    axis: usize,
    unbiased: bool,
) -> Result<Var<T, B>, B::Error> {
    Ok(std_mean_axis(a, axis, unbiased)?.0)
}
