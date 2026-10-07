//! Margin-based losses.
//!
//! Hinge and margin ranking objectives over scores: class-index margins,
//! binary soft-margin, pairwise ranking, and hinge embedding.

use coeus_autograd::Var;
use coeus_core::{
    ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, FloatElement,
};
use coeus_tensor::Tensor;

/// Multi-class margin loss (PyTorch `MultiMarginLoss`, `reduction="mean"`).
/// x: `[N, C]` scores, targets: `[N]` class indices, p >= 1, margin.
/// Computes `mean_i (1/C) sum_{j != y_i} max(0, margin - x[i,y_i] + x[i,j])^p`.
#[inline]
pub fn multi_margin<
    T: Float + FloatElement + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    x: &Var<T, B>,
    targets: &[usize],
    p: T,
    margin: T,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::multi_margin(x, targets, p, margin)
}

/// Multi-label margin loss (PyTorch `MultiLabelMarginLoss` with `reduction="mean"`).
///
/// `x`: shape `(N, C)` scores, `target`: shape `(N, C)` where
/// `target[i][j] >= 0` are valid class indices and `-1` means ignore padding.
/// Computes `mean_i sum_{t: target[i][t] >= 0} sum_{j != t} max(0, 1 - (x[i][t] - x[i][j]))`.
#[inline]
pub fn multi_label_margin_loss<
    T: Float + FloatElement + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    x: &Var<T, B>,
    target: &[isize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::multi_label_margin_loss(x, target)
}

/// Soft-margin (logistic) loss. input: `[..]`, target: `[..]` in `{-1, +1}`.
/// Computes `mean(log(1 + exp(-target * input)))` (PyTorch `SoftMarginLoss`).
#[inline]
pub fn soft_margin<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::soft_margin(input, target)
}

/// Margin ranking loss.
///
/// `target` contains `+1` or `-1` labels. Computes
/// `mean(max(0, -target * (input1 - input2) + margin))`.
#[inline]
pub fn margin_ranking_loss<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    input1: &Var<T, B>,
    input2: &Var<T, B>,
    target: &[T],
    margin: T,
) -> Var<T, B> {
    coeus_autograd::margin_ranking_loss(input1, input2, target, margin)
}

/// Hinge embedding loss (PyTorch `HingeEmbeddingLoss` with `reduction="mean"`).
///
/// `target` contains `+1` or `-1`. For `y_i == 1`, computes `mean(max(0, margin - x_i))`;
/// for `y_i == -1`, computes `mean(max(0, -x_i))`.
///
/// Composed from `where_cond`, `relu`, `neg`, and `scalar_sub` — no dedicated autograd node.
#[inline]
pub fn hinge_embedding_loss<T, B>(x: &Var<T, B>, target: &[T], margin: T) -> Var<T, B>
where
    T: Float + coeus_leto::RealScalar,
    B: ComputeBackend + Default + coeus_ops::BackendOps<T>,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        target.len(),
        x.tensor.numel(),
        "target length must match input length"
    );
    let backend = B::default();
    let zero = T::zero();

    let mask_data: Vec<T> = target
        .iter()
        .map(|&y| if y > zero { T::one() } else { zero })
        .collect();
    let mask_tensor = Tensor::from_slice_on(x.tensor.shape(), &mask_data, &backend);
    let mask_var = Var::new(mask_tensor, false);

    // PyTorch HingeEmbeddingLoss: target = +1 → loss = x (identity, no clamp);
    // target = -1 → loss = max(0, margin - x). `mask` is 1 where target > 0, so
    // `where_cond` selects the identity branch there and the hinge branch (the
    // `-1` case) otherwise.
    let hinge = coeus_autograd::relu(&coeus_autograd::neg(&coeus_autograd::scalar_sub(x, margin)));
    let selected = coeus_autograd::where_cond(&mask_var, x, &hinge);
    coeus_autograd::mean(&selected)
}
