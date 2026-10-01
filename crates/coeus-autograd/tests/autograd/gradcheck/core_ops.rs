//! Finite-difference checks for the backward paths ritk's registration stack
//! depends on: matmul, softmax, layernorm and gather.
//!
//! Each check runs at both `f64` and `f32` (see the module documentation for
//! what each scalar width verifies).

use super::{tensor, weighted, weighting, GradcheckScalar, Sampler};
use coeus_autograd::{gather, gradcheck, layernorm, matmul, mul, reshape, softmax, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

fn matmul_case<T: GradcheckScalar>() {
    // [2,3] × [3,4] → [2,4]. Both operands are differentiated, so this covers
    // dA = dC·Bᵀ and dB = Aᵀ·dC in one check; a transposed or swapped operand
    // in either rule shows up as a mismatch.
    let a = tensor::<T>(&[2, 3], 0.17);
    let b = tensor::<T>(&[3, 4], 0.53);
    let w = weighting::<T>(&[2, 4]);

    gradcheck(&[a, b], |v| {
        weighted(
            &matmul(&v[0], &v[1]).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("matmul backward must match central differences");
}

#[test]
fn matmul_backward_matches_finite_differences() {
    matmul_case::<f64>();
    matmul_case::<f32>();
}

fn softmax_case<T: GradcheckScalar>() {
    // The softmax Jacobian J = diag(y) - y·yᵀ has rows summing to zero, which
    // is exactly why a uniform loss weighting yields no signal. The non-uniform
    // weighting keeps the off-diagonal -y_i·y_j terms in the comparison.
    let x = tensor::<T>(&[3, 5], 0.29);
    let w = weighting::<T>(&[3, 5]);

    gradcheck(&[x], |v| {
        weighted(
            &softmax(&v[0], 1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("softmax backward must match central differences");
}

#[test]
fn softmax_backward_matches_finite_differences() {
    softmax_case::<f64>();
    softmax_case::<f32>();
}

fn softmax_negative_dim_case<T: GradcheckScalar>() {
    // `softmax` takes an isize dim with negative indexing; -1 must normalise to
    // the last axis and produce the same verified gradient.
    let x = tensor::<T>(&[2, 4], 0.71);
    let w = weighting::<T>(&[2, 4]);

    gradcheck(&[x], |v| {
        weighted(
            &softmax(&v[0], -1).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("softmax over dim -1 must match central differences");
}

#[test]
fn softmax_backward_matches_finite_differences_on_negative_dim() {
    softmax_negative_dim_case::<f64>();
    softmax_negative_dim_case::<f32>();
}

fn layernorm_case<T: GradcheckScalar>() {
    // LayerNorm's backward is the one most easily got wrong by hand: the
    // gradient must carry both correction terms that arise because the mean and
    // the variance are themselves functions of every element of the row.
    // Input, weight and bias are all differentiated.
    const ROWS: usize = 3;
    const WIDTH: usize = 4;
    const EPS: f64 = 1e-5;

    let x = tensor::<T>(&[ROWS, WIDTH], 0.13);
    let weight = Sampler::new(0.41, 0.5, 1.5).tensor::<T>(&[WIDTH]);
    let bias = tensor::<T>(&[WIDTH], 0.67);
    let w = weighting::<T>(&[ROWS, WIDTH]);

    gradcheck(&[x, weight, bias], |v| {
        let backend = MoiraiBackend::new();
        let flattened = reshape(&v[0], [ROWS, WIDTH]).expect("invariant: test operation succeeds");

        // `layernorm` attaches a node to an already-computed forward, so the
        // closure reproduces the statistics the node saves.
        let mean = coeus_ops::mean_axis(&flattened.tensor, 1, &backend).expect("row mean");
        let centered = coeus_ops::sub(&flattened.tensor, &mean, &backend)
            .expect("invariant: test operation succeeds");
        let centered_squared = coeus_ops::mul(&centered, &centered, &backend)
            .expect("invariant: test operation succeeds");
        let mut deviation =
            coeus_ops::mean_axis(&centered_squared, 1, &backend).expect("row variance");
        let epsilon = Tensor::<T, MoiraiBackend>::full_on(
            [1],
            <T as coeus_core::Scalar>::from_f64(EPS),
            &backend,
        )
        .expect("invariant: test backend operation succeeds");
        coeus_ops::add_assign(&mut deviation, &epsilon, &backend).expect("variance + eps");
        coeus_ops::sqrt_assign(&mut deviation, &backend).expect("stddev");

        let mut istdev = Tensor::<T, MoiraiBackend>::ones_on([ROWS, 1], &backend)
            .expect("invariant: test backend operation succeeds");
        coeus_ops::div_assign(&mut istdev, &deviation, &backend).expect("inverse stddev");
        let x_hat = coeus_ops::mul(&centered, &istdev, &backend)
            .expect("invariant: test operation succeeds");

        let weight_row = v[1].tensor.reshape([1, WIDTH]);
        let bias_row = v[2].tensor.reshape([1, WIDTH]);
        let mut output = coeus_ops::mul(&x_hat, &weight_row, &backend)
            .expect("invariant: test operation succeeds");
        coeus_ops::add_assign(&mut output, &bias_row, &backend).expect("affine shift");

        let normalized = layernorm(
            &flattened,
            &v[1],
            &v[2],
            output,
            x_hat,
            istdev,
            Tensor::<T, MoiraiBackend>::full_on(
                [1],
                <T as coeus_core::Scalar>::from_f64(WIDTH as f64),
                &backend,
            )
            .expect("invariant: test backend operation succeeds"),
        )
        .expect("invariant: test operation succeeds");
        weighted(&normalized, &w)
    })
    .expect("layernorm backward must match central differences");
}

#[test]
fn layernorm_backward_matches_finite_differences() {
    layernorm_case::<f64>();
    layernorm_case::<f32>();
}

fn gather_repeated_indices_case<T: GradcheckScalar>() {
    // gather's backward is a scatter-add, so a repeated index must *accumulate*
    // rather than overwrite. Column 1 is selected twice in row 0 and column 3
    // twice in row 1; an overwriting backward under-counts those entries and
    // the finite difference catches it.
    let backend = MoiraiBackend::new();
    let x = tensor::<T>(&[2, 4], 0.23);
    let index_values: Vec<T> = [1.0, 1.0, 2.0, 3.0, 0.0, 3.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let index = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on([2, 3], &index_values, &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let w = weighting::<T>(&[2, 3]);

    gradcheck(&[x], |v| {
        weighted(
            &gather(&v[0], 1, &index).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("gather backward must match central differences");
}

#[test]
fn gather_backward_matches_finite_differences_with_repeated_indices() {
    gather_repeated_indices_case::<f64>();
    gather_repeated_indices_case::<f32>();
}

fn gather_unselected_columns_case<T: GradcheckScalar>() {
    // Complement to the check above: an index set that never names column 2
    // must leave that column's gradient exactly zero. The non-uniform weighting
    // keeps the selected columns non-zero, so the guard does not fire and the
    // zero is a real result rather than a vacuous one.
    let backend = MoiraiBackend::new();
    let x = tensor::<T>(&[2, 4], 0.37);
    let index_values: Vec<T> = [0.0, 1.0, 3.0, 0.0]
        .into_iter()
        .map(<T as coeus_core::Scalar>::from_f64)
        .collect();
    let index = Var::new(
        Tensor::<T, MoiraiBackend>::from_slice_on([2, 2], &index_values, &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let w = weighting::<T>(&[2, 2]);

    gradcheck(std::slice::from_ref(&x), |v| {
        weighted(
            &gather(&v[0], 1, &index).expect("invariant: test operation succeeds"),
            &w,
        )
    })
    .expect("gather backward must match central differences");

    let tracked = Var::new(x, true).expect("invariant: test backend operation succeeds");
    sum(&mul(
        &gather(&tracked, 1, &index).expect("invariant: test operation succeeds"),
        &w,
    )
    .expect("invariant: test operation succeeds"))
    .expect("invariant: test operation succeeds")
    .backward()
    .expect("invariant: valid autograd fixture completes backward");
    let grad = tracked.grad().expect("input must receive a gradient");
    let slice = grad.as_slice();
    let zero = <T as coeus_core::Scalar>::from_f64(0.0);
    assert_eq!(slice[2], zero, "row 0 column 2 was never selected");
    assert_eq!(slice[6], zero, "row 1 column 2 was never selected");
}

#[test]
fn gather_backward_leaves_unselected_columns_at_zero() {
    gather_unselected_columns_case::<f64>();
    gather_unselected_columns_case::<f32>();
}
