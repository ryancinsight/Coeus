//! Finite-difference verification for the backward paths ritk's registration
//! stack depends on.
//!
//! Every check here runs through [`coeus_autograd::gradcheck`], which
//! reconstructs the gradient from forward evaluations alone. That makes it
//! independent of both the backward implementation *and* of any closed form a
//! test author might derive by hand — the two error sources a hand-derived
//! expected-value test cannot separate.
//!
//! Two conventions apply throughout:
//!
//! * **Non-uniform output weighting.** `gradcheck` requires a scalar loss, and
//!   the obvious `sum(op(x))` is the wrong reduction for any op whose rows sum
//!   to a constant — `sum(softmax(x))` is identically `1`, so its gradient is
//!   exactly zero and the comparison is vacuous. Weighting the output by a
//!   fixed non-uniform tensor probes real Jacobian entries. The helper rejects
//!   the vacuous case rather than passing it silently.
//! * **Deterministic irregular inputs.** Values are generated from a fixed
//!   affine sequence rather than round numbers, so a gradient that is only
//!   correct on symmetric or repeated inputs still fails.

use coeus_autograd::{gather, gradcheck, layernorm, matmul, mul, reshape, softmax, sum, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

type T64 = Tensor<f64, MoiraiBackend>;

/// Deterministic irregular values in roughly `[-1, 1]`, distinct per index.
fn ramp(count: usize, offset: f64, slope: f64) -> Vec<f64> {
    (0..count)
        .map(|i| {
            let x = slope.mul_add(i as f64, offset);
            // Fold into [-1, 1] without repeating a period over small counts.
            (x % 1.7) - 0.85
        })
        .collect()
}

fn tensor(shape: &[usize], offset: f64, slope: f64) -> T64 {
    let count = shape.iter().product();
    T64::from_slice_on(
        shape.to_vec(),
        &ramp(count, offset, slope),
        &MoiraiBackend::new(),
    )
}

/// A fixed non-uniform weighting of the op output, held constant across every
/// perturbed evaluation so the loss stays a pure function of the inputs.
fn weighting(shape: &[usize]) -> Var<f64, MoiraiBackend> {
    Var::new(tensor(shape, 0.31, 0.23), false)
}

#[test]
fn matmul_backward_matches_finite_differences() {
    // [2,3] × [3,4] → [2,4]. Both operands are differentiated, so this covers
    // dA = dC·Bᵀ and dB = Aᵀ·dC in one check; a transposed or swapped operand
    // in either rule shows up as a mismatch.
    let a = tensor(&[2, 3], -0.4, 0.37);
    let b = tensor(&[3, 4], 0.15, -0.29);
    let w = weighting(&[2, 4]);

    gradcheck(&[a, b], |v| sum(&mul(&matmul(&v[0], &v[1]), &w)))
        .expect("matmul backward must match central differences");
}

#[test]
fn softmax_backward_matches_finite_differences() {
    // The softmax Jacobian J = diag(y) - y·yᵀ has rows summing to zero, which
    // is exactly why a uniform loss weighting yields no signal. The non-uniform
    // weighting below keeps the off-diagonal -y_i·y_j terms in the comparison.
    let x = tensor(&[3, 5], -0.6, 0.41);
    let w = weighting(&[3, 5]);

    gradcheck(&[x], |v| sum(&mul(&softmax(&v[0], 1), &w)))
        .expect("softmax backward must match central differences");
}

#[test]
fn softmax_backward_matches_finite_differences_on_negative_dim() {
    // `softmax` takes an isize dim with negative indexing; -1 must normalise to
    // the last axis and produce the same verified gradient.
    let x = tensor(&[2, 4], 0.22, -0.33);
    let w = weighting(&[2, 4]);

    gradcheck(&[x], |v| sum(&mul(&softmax(&v[0], -1), &w)))
        .expect("softmax over dim -1 must match central differences");
}

#[test]
fn layernorm_backward_matches_finite_differences() {
    // LayerNorm's backward is the one most easily got wrong by hand: the
    // gradient must carry both correction terms that arise because the mean and
    // the variance are themselves functions of every element of the row.
    // Input, weight and bias are all differentiated.
    const ROWS: usize = 3;
    const WIDTH: usize = 4;
    const EPS: f64 = 1e-5;

    let x = tensor(&[ROWS, WIDTH], -0.5, 0.43);
    let weight = tensor(&[WIDTH], 0.7, 0.19);
    let bias = tensor(&[WIDTH], -0.2, 0.11);
    let w = weighting(&[ROWS, WIDTH]);

    gradcheck(&[x, weight, bias], |v| {
        let backend = MoiraiBackend::new();
        let flattened = reshape(&v[0], [ROWS, WIDTH]);

        // `layernorm` attaches a node to an already-computed forward, so the
        // closure reproduces the statistics the node saves.
        let mean = coeus_ops::mean_axis(&flattened.tensor, 1, &backend).expect("row mean");
        let centered = coeus_ops::sub(&flattened.tensor, &mean, &backend);
        let centered_squared = coeus_ops::mul(&centered, &centered, &backend);
        let mut deviation =
            coeus_ops::mean_axis(&centered_squared, 1, &backend).expect("row variance");
        let epsilon = T64::full_on([1], EPS, &backend);
        coeus_ops::add_assign(&mut deviation, &epsilon, &backend).expect("variance + eps");
        coeus_ops::sqrt_assign(&mut deviation, &backend).expect("stddev");

        let mut istdev = T64::ones_on([ROWS, 1], &backend);
        coeus_ops::div_assign(&mut istdev, &deviation, &backend).expect("inverse stddev");
        let x_hat = coeus_ops::mul(&centered, &istdev, &backend);

        let weight_row = v[1].tensor.reshape([1, WIDTH]);
        let bias_row = v[2].tensor.reshape([1, WIDTH]);
        let mut output = coeus_ops::mul(&x_hat, &weight_row, &backend);
        coeus_ops::add_assign(&mut output, &bias_row, &backend).expect("affine shift");

        let normalized = layernorm(
            &flattened,
            &v[1],
            &v[2],
            output,
            x_hat,
            istdev,
            T64::full_on([1], WIDTH as f64, &backend),
        );
        sum(&mul(&normalized, &w))
    })
    .expect("layernorm backward must match central differences");
}

#[test]
fn gather_backward_matches_finite_differences_with_repeated_indices() {
    // gather's backward is a scatter-add, so a repeated index must *accumulate*
    // rather than overwrite. Column 1 is selected twice in row 0 and column 3
    // twice in row 1; an overwriting backward under-counts those entries and
    // the finite difference catches it.
    let backend = MoiraiBackend::new();
    let x = tensor(&[2, 4], -0.45, 0.31);
    let index = Var::new(
        T64::from_slice_on([2, 3], &[1.0, 1.0, 2.0, 3.0, 0.0, 3.0], &backend),
        false,
    );
    let w = weighting(&[2, 3]);

    gradcheck(&[x], |v| sum(&mul(&gather(&v[0], 1, &index), &w)))
        .expect("gather backward must match central differences");
}

#[test]
fn gather_backward_leaves_unselected_columns_at_zero() {
    // Complement to the check above: an index set that never names column 2
    // must leave that column's gradient exactly zero. The non-uniform weighting
    // keeps the selected columns non-zero, so the guard does not fire and the
    // zero is a real result rather than a vacuous one.
    let backend = MoiraiBackend::new();
    let x = tensor(&[2, 4], 0.13, 0.27);
    let index = Var::new(
        T64::from_slice_on([2, 2], &[0.0, 1.0, 3.0, 0.0], &backend),
        false,
    );
    let w = weighting(&[2, 2]);

    gradcheck(std::slice::from_ref(&x), |v| {
        sum(&mul(&gather(&v[0], 1, &index), &w))
    })
    .expect("gather backward must match central differences");

    let tracked = Var::new(x, true);
    sum(&mul(&gather(&tracked, 1, &index), &w))
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = tracked.grad().expect("input must receive a gradient");
    let slice = grad.as_slice();
    assert_eq!(slice[2], 0.0, "row 0 column 2 was never selected");
    assert_eq!(slice[6], 0.0, "row 1 column 2 was never selected");
}
