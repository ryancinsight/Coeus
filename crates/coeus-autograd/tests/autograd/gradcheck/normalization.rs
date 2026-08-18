//! Finite-difference checks for the normalization backward passes.
//!
//! These are the backward passes most easily got wrong by hand. In each of
//! them the normalizing statistic is itself a function of every element being
//! normalized, so the gradient carries correction terms beyond the obvious
//! `dy·γ/σ`: drop them and the result still looks plausible, still trains after
//! a fashion, and still passes any test that only checks shapes or finiteness.
//! A finite difference of the real forward is what separates the correct
//! gradient from the plausible one.
//!
//! `rmsnorm` and the `batchnorm` family attach their node to an
//! already-computed forward, so each closure reproduces the forward exactly as
//! the owning layer in `coeus-nn` computes it. That duplication is the price of
//! checking the node from outside the layer; the alternative — trusting the
//! layer to build the saved tensors correctly — would put the thing under test
//! on both sides of the comparison.

use super::{tensor, weighted, weighting, Sampler, T64};
use coeus_autograd::{batchnorm1d, gradcheck, rmsnorm, BatchNormArgs};
use coeus_core::MoiraiBackend;

/// Epsilon added inside the normalizing square root.
///
/// `1e-5` is the framework default. It is far above the `f64` accuracy floor,
/// so it does not interact with the finite-difference tolerance; its only role
/// here is to keep the derivative of the square root bounded.
const EPS: f64 = 1e-5;

#[test]
fn rmsnorm_backward_matches_finite_differences() {
    // RMSNorm normalizes by r = sqrt(mean(x²) + eps) without centering, so
    //
    //   dL/dx_i = (1/r)·[ (dy·w)_i - x̂_i·mean_j((dy·w)_j·x̂_j) ]
    //
    // The second term exists only because r depends on every element of the
    // row. A backward that returned just (dy·w)/r — the term you get by
    // treating r as a constant — passes shape checks and is wrong by an O(1)
    // amount that this check localises per element. Input and weight are both
    // differentiated.
    const ROWS: usize = 3;
    const WIDTH: usize = 4;

    let x = tensor(&[ROWS, WIDTH], 0.27);
    let weight = Sampler::new(0.53, 0.4, 1.6).tensor(&[WIDTH]);
    let w = weighting(&[ROWS, WIDTH]);

    gradcheck(&[x, weight], |v| {
        let backend = MoiraiBackend::new();

        let x_sq = coeus_ops::mul(&v[0].tensor, &v[0].tensor, &backend);
        let mut rms = coeus_ops::mean_axis(&x_sq, 1, &backend).expect("row mean square");
        let epsilon = T64::full_on([1], EPS, &backend);
        coeus_ops::add_assign(&mut rms, &epsilon, &backend).expect("mean square + eps");
        coeus_ops::sqrt_assign(&mut rms, &backend).expect("root mean square");

        let x_hat = coeus_ops::div(&v[0].tensor, &rms, &backend);
        let weight_row = v[1].tensor.reshape([1, WIDTH]);
        let out_tensor = coeus_ops::mul(&x_hat, &weight_row, &backend);

        weighted(&rmsnorm(&v[0], &v[1], out_tensor, x_hat, rms), &w)
    })
    .expect("rmsnorm backward must match central differences");
}

#[test]
fn batchnorm1d_backward_matches_finite_differences() {
    // BatchNorm in training mode normalizes by statistics taken over the batch,
    // so element (n, c, l) influences the output of *every* sample in channel c.
    // The gradient therefore carries the mean and variance correction terms
    //
    //   dL/dx = (γ·istd/m)·[ m·dy - Σdy - x̂·Σ(dy·x̂) ]
    //
    // and dropping either sum leaves a backward that is correct only when the
    // batch happens to be centred — which a fixture of round numbers often is,
    // and which this irregular fixture is not. Input, weight and bias are all
    // differentiated.
    const N: usize = 3;
    const C: usize = 2;
    const L: usize = 2;
    const M: usize = N * L;

    let x = tensor(&[N, C, L], 0.19);
    let weight = Sampler::new(0.61, 0.4, 1.6).tensor(&[C]);
    let bias = tensor(&[C], 0.83);
    let w = weighting(&[N, C, L]);

    gradcheck(&[x, weight, bias], |v| {
        let backend = MoiraiBackend::new();

        // [N, C, L] → [N, L, C] → [M, C], matching the layer's own view.
        let nlc = v[0].tensor.permute(&[0, 2, 1]).to_contiguous_on(&backend);
        let flat = nlc.reshape([M, C]);

        let mean = coeus_ops::mean_axis(&flat, 0, &backend).expect("per-channel mean");
        let xmu = coeus_ops::sub(&flat, &mean, &backend);
        let xmu_sq = coeus_ops::mul(&xmu, &xmu, &backend);
        let variance = coeus_ops::mean_axis(&xmu_sq, 0, &backend).expect("per-channel variance");

        let mut stdev = variance;
        let epsilon = T64::full_on([1], EPS, &backend);
        coeus_ops::add_assign(&mut stdev, &epsilon, &backend).expect("variance + eps");
        coeus_ops::sqrt_assign(&mut stdev, &backend).expect("standard deviation");

        let mut istdev = T64::ones_on([1, C], &backend);
        coeus_ops::div_assign(&mut istdev, &stdev, &backend).expect("inverse standard deviation");

        let x_hat = coeus_ops::mul(&xmu, &istdev, &backend);
        let weight_row = v[1].tensor.reshape([1, C]);
        let bias_row = v[2].tensor.reshape([1, C]);
        let mut y_flat = coeus_ops::mul(&x_hat, &weight_row, &backend);
        coeus_ops::add_assign(&mut y_flat, &bias_row, &backend).expect("affine shift");

        let out_tensor = y_flat
            .reshape([N, L, C])
            .permute(&[0, 2, 1])
            .to_contiguous_on(&backend);

        let normalized = batchnorm1d(
            &v[0],
            &v[1],
            &v[2],
            BatchNormArgs {
                out_tensor,
                x_hat,
                xmu,
                istdev,
                m_const: T64::full_on([1], M as f64, &backend),
                minus_half: T64::full_on([1], -0.5, &backend),
                two_const: T64::full_on([1], 2.0, &backend),
                n: N,
                c: C,
                spatial: [L, 1, 1],
                m: M,
            },
        );
        weighted(&normalized, &w)
    })
    .expect("batchnorm1d backward must match central differences");
}
