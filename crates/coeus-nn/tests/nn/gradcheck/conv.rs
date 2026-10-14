//! Finite-difference checks for convolution, transposed convolution, and
//! unfold/fold, tested through their `coeus-nn` `Module` layers (see
//! `pooling.rs` for why: reproducing output-shape arithmetic here would put
//! the same computation on both sides of the comparison).
//!
//! `Conv`/`ConvTranspose` hold their weight as a `Var` field rather than a
//! function argument, so each case rebuilds the layer inside the gradcheck
//! closure from the tracked `weight` input (`Conv::from_vars` for `Conv`,
//! direct struct construction for `ConvTranspose` — its fields are `pub` and
//! its `new`/`with_params` draw a random initial weight via
//! `RandomInitOps`, which gradcheck's fixed, irrational-sequence inputs must
//! bypass). Both input and weight are differentiated. Convolution and
//! transposed convolution are linear in both operands (no kink); unfold/fold
//! are linear rearrangements (no kink, no weight).

use super::{tensor, weighted, weighting, GradcheckScalar};
use coeus_autograd::gradcheck;
use coeus_nn::{Conv1d, Conv3d, ConvTranspose, Fold1d, Fold2d, Module, Unfold1d, Unfold2d};
use coeus_ops::conv_transpose::{
    conv_transpose1d_output_len, conv_transpose2d_output_dims, conv_transpose3d_output_dims,
};

fn conv1d_case<T: GradcheckScalar>() {
    // [N=1, Cin=2, L=6], kernel 3 stride 1 pad 0 dil 1: L_out = 4.
    const L_OUT: usize = 4;
    let x = tensor::<T>(&[1, 2, 6], 0.19);
    let weight = tensor::<T>(&[3, 2, 3], 0.53);
    let w = weighting::<T>(&[1, 3, L_OUT]);

    gradcheck(&[x, weight], |v| {
        let layer = Conv1d::<T>::from_vars(
            v[1].clone(),
            None,
            coeus_nn::conv::ConvParams::new(2, 3, 3, 1, 0, 1),
        );
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid conv1d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("conv1d backward must match central differences");
}

#[test]
fn conv1d_backward_matches_finite_differences() {
    conv1d_case::<f64>();
    conv1d_case::<f32>();
}

fn conv3d_case<T: GradcheckScalar>() {
    // [N=1, Cin=1, D=H=W=4], kernel 2 stride 1 pad 0 dil 1: out spatial 3 each.
    let x = tensor::<T>(&[1, 1, 4, 4, 4], 0.23);
    let weight = tensor::<T>(&[2, 1, 2, 2, 2], 0.61);
    let w = weighting::<T>(&[1, 2, 3, 3, 3]);

    gradcheck(&[x, weight], |v| {
        let layer = Conv3d::<T>::from_vars(
            v[1].clone(),
            None,
            coeus_nn::conv::ConvParams::new(1, 2, 2, 1, 0, 1),
        );
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid conv3d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("conv3d backward must match central differences");
}

#[test]
fn conv3d_backward_matches_finite_differences() {
    conv3d_case::<f64>();
    conv3d_case::<f32>();
}

fn conv_transpose1d_case<T: GradcheckScalar>() {
    const CIN: usize = 2;
    const COUT: usize = 3;
    const K: usize = 3;
    let l_out = conv_transpose1d_output_len(6, K, 1, 0, 0, 1);
    let x = tensor::<T>(&[1, CIN, 6], 0.29);
    let weight = tensor::<T>(&[CIN, COUT, K], 0.67);
    let w = weighting::<T>(&[1, COUT, l_out]);

    gradcheck(&[x, weight], |v| {
        let layer = ConvTranspose::<T, coeus_core::MoiraiBackend, 1> {
            weight: v[1].clone(),
            bias: None,
            in_channels: CIN,
            out_channels: COUT,
            kernel_size: K,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
        };
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid conv_transpose1d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("conv_transpose1d backward must match central differences");
}

#[test]
fn conv_transpose1d_backward_matches_finite_differences() {
    conv_transpose1d_case::<f64>();
    conv_transpose1d_case::<f32>();
}

fn conv_transpose2d_case<T: GradcheckScalar>() {
    const CIN: usize = 2;
    const COUT: usize = 2;
    const K: usize = 2;
    let (h_out, w_out) = conv_transpose2d_output_dims(4, 4, K, K, 2, 0, 0, 1);
    let x = tensor::<T>(&[1, CIN, 4, 4], 0.31);
    let weight = tensor::<T>(&[CIN, COUT, K, K], 0.71);
    let w = weighting::<T>(&[1, COUT, h_out, w_out]);

    gradcheck(&[x, weight], |v| {
        let layer = ConvTranspose::<T, coeus_core::MoiraiBackend, 2> {
            weight: v[1].clone(),
            bias: None,
            in_channels: CIN,
            out_channels: COUT,
            kernel_size: K,
            stride: 2,
            padding: 0,
            output_padding: 0,
            dilation: 1,
        };
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid conv_transpose2d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("conv_transpose2d backward must match central differences");
}

#[test]
fn conv_transpose2d_backward_matches_finite_differences() {
    conv_transpose2d_case::<f64>();
    conv_transpose2d_case::<f32>();
}

fn conv_transpose3d_case<T: GradcheckScalar>() {
    const CIN: usize = 1;
    const COUT: usize = 2;
    const K: usize = 2;
    let (d_out, h_out, w_out) = conv_transpose3d_output_dims(3, 3, 3, K, K, K, 1, 0, 0, 1);
    let x = tensor::<T>(&[1, CIN, 3, 3, 3], 0.37);
    let weight = tensor::<T>(&[CIN, COUT, K, K, K], 0.73);
    let w = weighting::<T>(&[1, COUT, d_out, h_out, w_out]);

    gradcheck(&[x, weight], |v| {
        let layer = ConvTranspose::<T, coeus_core::MoiraiBackend, 3> {
            weight: v[1].clone(),
            bias: None,
            in_channels: CIN,
            out_channels: COUT,
            kernel_size: K,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
        };
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid conv_transpose3d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("conv_transpose3d backward must match central differences");
}

#[test]
fn conv_transpose3d_backward_matches_finite_differences() {
    conv_transpose3d_case::<f64>();
    conv_transpose3d_case::<f32>();
}

fn unfold1d_case<T: GradcheckScalar>() {
    // [N=1, C=2, L=5], kernel 3 stride 1 pad 0 dil 1: L_out = 3, out [1,6,3].
    let x = tensor::<T>(&[1, 2, 5], 0.41);
    let w = weighting::<T>(&[1, 6, 3]);
    let layer = Unfold1d::<T>::new(3, 1, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid unfold1d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("unfold1d backward must match central differences");
}

#[test]
fn unfold1d_backward_matches_finite_differences() {
    unfold1d_case::<f64>();
    unfold1d_case::<f32>();
}

fn fold1d_case<T: GradcheckScalar>() {
    // Inverse of the unfold1d fixture above: [1,6,3] -> [1,2,5].
    let x = tensor::<T>(&[1, 6, 3], 0.43);
    let w = weighting::<T>(&[1, 2, 5]);
    let layer = Fold1d::<T>::new(5, 3, 1, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid fold1d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("fold1d backward must match central differences");
}

#[test]
fn fold1d_backward_matches_finite_differences() {
    fold1d_case::<f64>();
    fold1d_case::<f32>();
}

fn unfold2d_case<T: GradcheckScalar>() {
    // [N=1, C=2, H=4, W=4], kernel 2 stride 1 pad 0 dil 1: out [1,8,9].
    let x = tensor::<T>(&[1, 2, 4, 4], 0.47);
    let w = weighting::<T>(&[1, 8, 9]);
    let layer = Unfold2d::<T>::new(2, 1, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid unfold2d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("unfold2d backward must match central differences");
}

#[test]
fn unfold2d_backward_matches_finite_differences() {
    unfold2d_case::<f64>();
    unfold2d_case::<f32>();
}

fn fold2d_case<T: GradcheckScalar>() {
    // Inverse of the unfold2d fixture above: [1,8,9] -> [1,2,4,4].
    let x = tensor::<T>(&[1, 8, 9], 0.53);
    let w = weighting::<T>(&[1, 2, 4, 4]);
    let layer = Fold2d::<T>::new(4, 4, 2, 1, 0, 1);

    gradcheck(&[x], |v| {
        let y = layer
            .forward(&v[0])
            .expect("invariant: valid fold2d fixture completes forward");
        weighted(&y, &w)
    })
    .expect("fold2d backward must match central differences");
}

#[test]
fn fold2d_backward_matches_finite_differences() {
    fold2d_case::<f64>();
    fold2d_case::<f32>();
}
