//! Finite-difference checks for ops the parent module did not yet cover:
//! CTC, dropout, sparse matmul (CSR and COO), `transpose_2d`, `index_put`,
//! `rotate_half`, and `linear_interpolation`.
//!
//! Every check here runs at both `f64` and `f32` except
//! [`linear_interpolation_backward_matches_finite_differences`]: that op's
//! signature is pinned to `f32` end to end (`Var<f32, B>` on both the image
//! and the grid, in both `coeus-autograd` and `coeus-ops`), so there is no
//! `f32`-monomorphization gap for a wider check to close — the `f32` run is
//! this op's only instantiation, not a narrowed one.

use super::{tensor, weighted, weighting, GradcheckScalar, Sampler};
use coeus_autograd::{
    ctc_loss, dropout, gradcheck, index_put, linear_interpolation, rotate_half, sparse_matmul,
    sparse_matmul_coo, staggered_divergence, staggered_gradient, transpose_2d, Var,
};
use coeus_core::MoiraiBackend;
use coeus_ops::{Axis, Replicate};
use coeus_tensor::Tensor;

fn transpose_2d_case<T: GradcheckScalar>() {
    let x = tensor::<T>(&[3, 4], 0.11);
    let w = weighting::<T>(&[4, 3]);
    gradcheck(&[x], |v| weighted(&transpose_2d(&v[0]), &w))
        .expect("transpose_2d backward must match central differences");
}

#[test]
fn transpose_2d_backward_matches_finite_differences() {
    transpose_2d_case::<f64>();
    transpose_2d_case::<f32>();
}

fn rotate_half_case<T: GradcheckScalar>()
where
    MoiraiBackend: coeus_ops::RotateHalfOps<T>,
{
    // The final extent must be even (`rotate_half`'s own contract); [2, 4]
    // splits each row into two halves of width 2.
    let x = tensor::<T>(&[2, 4], 0.13);
    let w = weighting::<T>(&[2, 4]);
    gradcheck(&[x], |v| {
        weighted(
            &rotate_half(&v[0]).expect("invariant: even final extent satisfies the contract"),
            &w,
        )
    })
    .expect("rotate_half backward must match central differences");
}

#[test]
fn rotate_half_backward_matches_finite_differences() {
    rotate_half_case::<f64>();
    rotate_half_case::<f32>();
}

fn index_put_case<T: GradcheckScalar>() {
    // Position 1 is overwritten (not accumulated), so its original gradient
    // path must be cut while `values`' gradient lands there instead; position
    // 3 is untouched and must pass its input gradient straight through.
    let backend = MoiraiBackend::new();
    let x = tensor::<T>(&[4], 0.17);
    let index_values: Vec<T> = [1.0, 2.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let indices = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on([2], &index_values, &backend),
        false,
    );
    let values = tensor::<T>(&[2], 0.19);
    let w = weighting::<T>(&[4]);
    gradcheck(&[x, values], |v| {
        weighted(&index_put(&v[0], &indices, &v[1], false), &w)
    })
    .expect("index_put backward must match central differences");
}

#[test]
fn index_put_backward_matches_finite_differences() {
    index_put_case::<f64>();
    index_put_case::<f32>();
}

fn dropout_case<T: GradcheckScalar>() {
    // `dropout`'s mask is a deterministic function of `seed` alone (a fresh
    // `Xorshift64::new(seed)` every call), so gradcheck's repeated forward
    // evaluations see the same mask and the finite difference is well posed —
    // an RNG that carried state across calls would make the forward
    // non-reproducible and the comparison meaningless.
    let x = tensor::<T>(&[8], 0.23);
    let w = weighting::<T>(&[8]);
    gradcheck(&[x], |v| weighted(&dropout(&v[0], 0.3, true, 7), &w))
        .expect("dropout backward must match central differences");
}

#[test]
fn dropout_backward_matches_finite_differences() {
    dropout_case::<f64>();
    dropout_case::<f32>();
}

fn sparse_matmul_case<T: GradcheckScalar>()
where
    MoiraiBackend: coeus_core::Backend,
{
    // Only the nonzero values are differentiated; the CSR structure (column
    // indices, row offsets) is fixed data, exactly as `x.indices()` is fixed
    // for `index_select` elsewhere in this module.
    let backend = MoiraiBackend::new();
    let a_data: Vec<T> = [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let a_dense = Tensor::<T, MoiraiBackend>::from_slice_on(vec![3, 4], &a_data, &backend);
    let csr = coeus_ops::dense_to_csr(&a_dense, &backend);
    let a_values = csr.values().clone();
    let col_indices = csr.col_indices().clone();
    let row_offsets = csr.row_offsets().clone();

    let b = tensor::<T>(&[4, 2], 0.29);
    let w = weighting::<T>(&[3, 2]);

    gradcheck(&[a_values, b], |v| {
        weighted(
            &sparse_matmul(
                &v[0],
                &col_indices,
                &row_offsets,
                coeus_core::Shape::from(vec![3, 4]),
                &v[1],
            ),
            &w,
        )
    })
    .expect("sparse_matmul backward must match central differences");
}

#[test]
fn sparse_matmul_backward_matches_finite_differences() {
    sparse_matmul_case::<f64>();
    sparse_matmul_case::<f32>();
}

fn sparse_matmul_coo_case<T: GradcheckScalar>()
where
    MoiraiBackend: coeus_core::Backend,
{
    let backend = MoiraiBackend::new();
    let a_data: Vec<T> = [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 0.0, 5.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let a_dense = Tensor::<T, MoiraiBackend>::from_slice_on(vec![3, 4], &a_data, &backend);
    let coo = coeus_ops::dense_to_coo(&a_dense, &backend);
    let a_values = coo.values().clone();
    let indices = coo.indices().clone();
    let shape = coo.shape().clone();

    let b = tensor::<T>(&[4, 2], 0.31);
    let w = weighting::<T>(&[3, 2]);

    gradcheck(&[a_values, b], |v| {
        weighted(
            &sparse_matmul_coo(&v[0], &indices, shape.clone(), &v[1]),
            &w,
        )
    })
    .expect("sparse_matmul_coo backward must match central differences");
}

#[test]
fn sparse_matmul_coo_backward_matches_finite_differences() {
    sparse_matmul_coo_case::<f64>();
    sparse_matmul_coo_case::<f32>();
}

fn ctc_case<T: GradcheckScalar>()
where
    MoiraiBackend: coeus_ops::CtcOps<T>,
{
    // log_probs must already be normalized per frame (log_softmax's output),
    // so the check differentiates through a real forward chain rather than a
    // hand-crafted "log-probability" fixture that no forward pass produces.
    use coeus_autograd::log_softmax;

    const FRAMES: usize = 4;
    const BATCH: usize = 1;
    const CLASSES: usize = 3;
    let blank = 0usize;
    let targets = [1usize, 2];
    let input_lengths = [FRAMES];
    let target_lengths = [targets.len()];

    let logits = tensor::<T>(&[FRAMES, BATCH, CLASSES], 0.37);
    let w = weighting::<T>(&[1]);

    gradcheck(&[logits], |v| {
        let log_probs = log_softmax(&v[0], 2);
        let loss = ctc_loss(&log_probs, &targets, &input_lengths, &target_lengths, blank)
            .expect("invariant: valid ctc fixture completes forward");
        weighted(&loss, &w)
    })
    .expect("ctc_loss backward must match central differences");
}

#[test]
fn ctc_backward_matches_finite_differences() {
    ctc_case::<f64>();
    ctc_case::<f32>();
}

fn staggered_case<T: GradcheckScalar>(axis: Axis, divergence: bool)
where
    MoiraiBackend: coeus_ops::StaggeredPairOps<T>,
{
    // Order 4 (half-order 2) exercises multi-tap reflection; every axis
    // extent clears the stencil depth without relying on thin-grid folding.
    let x = tensor::<T>(&[5, 4, 6], 0.23);
    let w = weighting::<T>(&[5, 4, 6]);
    let spacing = [
        <T as coeus_core::Scalar>::from_f64(0.5),
        <T as coeus_core::Scalar>::from_f64(1.0),
        <T as coeus_core::Scalar>::from_f64(2.0),
    ];
    gradcheck(&[x], |v| {
        let swept = if divergence {
            staggered_divergence(&v[0], 4, spacing, axis)
                .expect("invariant: valid staggered fixture completes forward")
        } else {
            staggered_gradient(&v[0], 4, spacing, axis)
                .expect("invariant: valid staggered fixture completes forward")
        };
        weighted(&swept, &w)
    })
    .expect("staggered backward must match central differences");
}

#[test]
fn staggered_backward_matches_finite_differences() {
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        for divergence in [false, true] {
            staggered_case::<f64>(axis, divergence);
            staggered_case::<f32>(axis, divergence);
        }
    }
}

fn linear_interpolation_case() {
    // Grid coordinates are chosen away from every integer pixel boundary —
    // `linear_interpolation`'s `Replicate` policy has a kink wherever a
    // coordinate crosses an integer, exactly like `floor` in the shape ops
    // above — so the central difference lands inside one bilinear cell on
    // both sides of the perturbation.
    let backend = MoiraiBackend;
    let image_values: Vec<f32> = Sampler::signed(0.41)
        .values(16)
        .into_iter()
        .map(|value| value as f32)
        .collect();
    let image = Tensor::from_slice_on([1, 1, 4, 4], &image_values, &backend);
    let grid = Tensor::from_slice_on([1, 2, 2, 1], &[0.37_f32, 2.63, 1.19, 1.81], &backend);
    // Only the image is differentiated here: the grid's own gradient is
    // covered by the analytical checks in `autograd_ops::interpolation`, and
    // gradcheck perturbs every input it is given, so mixing a coordinate
    // input into the same call would let a perturbation cross the kink this
    // fixture was built to avoid.
    let w = weighting::<f32>(&[1, 1, 2, 1]);

    gradcheck(&[image], |v| {
        let sampled =
            linear_interpolation::<2, _, _>(&v[0], &Var::new(grid.clone(), false), Replicate)
                .expect("invariant: valid interpolation fixture completes forward");
        weighted(&sampled, &w)
    })
    .expect("linear_interpolation backward must match central differences");
}

#[test]
fn linear_interpolation_backward_matches_finite_differences() {
    linear_interpolation_case();
}
