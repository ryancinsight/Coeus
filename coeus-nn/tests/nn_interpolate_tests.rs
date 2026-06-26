//! Analytical-reference tests for `interpolate_1d` and `interpolate_2d`.
//!
//! Both functions use align-half-pixel convention (PyTorch align_corners=False):
//!   src_coord = (dst_coord + 0.5) * (src_size / dst_size) - 0.5   (bilinear)
//!   src_idx   = floor((dst_coord + 0.5) * src_size / dst_size)     (nearest)
//!
//! Reference values are chosen so results are IEEE-exact integers, so all
//! assertions use `assert_eq!` without epsilon.
//!
//! Tests cover: nearest upsample, nearest downsample, bilinear same-size
//! (identity), bilinear upsample, both 1-D and 2-D spatial variants,
//! multi-batch and multi-channel shapes.

use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut, SequentialBackend};
use coeus_nn::interpolate::{interpolate_1d, interpolate_2d, InterpolateMode};
use coeus_tensor::Tensor;

fn t1d(vals: &[f64], l: usize) -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice(vec![1, 1, l], vals)
}

fn t2d(vals: &[f64], h: usize, w: usize) -> Tensor<f64, SequentialBackend> {
    Tensor::from_slice(vec![1, 1, h, w], vals)
}

fn check_1d_nearest_upsample() {
    // Input [1,1,2] = [10, 20], upsample to new_l=4.
    // src_idx = floor((xi+0.5)*2/4)
    //   xi=0: 0.25→0→10  xi=1: 0.75→0→10  xi=2: 1.25→1→20  xi=3: 1.75→1→20
    let inp = t1d(&[10.0, 20.0], 2);
    let out = interpolate_1d(&inp, 4, InterpolateMode::Nearest);
    assert_eq!(out.shape(), &[1, 1, 4], "1d nearest upsample shape");
    assert_eq!(
        out.as_slice(),
        &[10.0_f64, 10.0, 20.0, 20.0],
        "1d nearest upsample"
    );
}

fn check_1d_nearest_downsample() {
    // Input [1,1,4] = [1,2,3,4], downsample to new_l=2.
    // src_idx = floor((xi+0.5)*4/2)
    //   xi=0: 1.0→1→2  xi=1: 3.0→3→4
    let inp = t1d(&[1.0, 2.0, 3.0, 4.0], 4);
    let out = interpolate_1d(&inp, 2, InterpolateMode::Nearest);
    assert_eq!(out.shape(), &[1, 1, 2], "1d nearest downsample shape");
    assert_eq!(out.as_slice(), &[2.0_f64, 4.0], "1d nearest downsample");
}

fn check_1d_nearest_identity() {
    // Same-size nearest: each xi maps to itself.
    let inp = t1d(&[5.0, 3.0, 8.0], 3);
    let out = interpolate_1d(&inp, 3, InterpolateMode::Nearest);
    assert_eq!(out.as_slice(), inp.as_slice(), "1d nearest identity");
}

fn check_1d_bilinear_identity() {
    // Same-size bilinear: align-half-pixel maps xi → xi exactly.
    // frac = (xi+0.5)*(L/L) - 0.5 = xi.0, floor=xi, x0=xi, x1=xi+1
    // but x1=min(xi+1, L-1) and w1=0.0 when frac is exact integer, so weight all on x0.
    // For xi=0: frac=0.0, x0=0, x1=1, w1=0.0 → in[0].
    // For xi=L-1: frac=L-1, x0=L-1, x1=L-1 (clamped), w1=0.0 → in[L-1].
    let inp = t1d(&[7.0, 11.0, 3.0], 3);
    let out = interpolate_1d(&inp, 3, InterpolateMode::Bilinear);
    assert_eq!(out.as_slice(), inp.as_slice(), "1d bilinear identity");
}

fn check_2d_nearest_upsample() {
    // Input [1,1,2,2] = [[1,2],[3,4]], upsample to [4,4].
    // src_y = floor((yi+0.5)*2/4), src_x = floor((xi+0.5)*2/4)
    // Row yi=0,1: sy=0  yi=2,3: sy=1
    // Col xi=0,1: sx=0  xi=2,3: sx=1
    // Result row-major: 4 rows × 4 cols
    //   row 0,1: [1,1,2,2]  row 2,3: [3,3,4,4]
    let inp = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let out = interpolate_2d(&inp, 4, 4, InterpolateMode::Nearest);
    assert_eq!(out.shape(), &[1, 1, 4, 4], "2d nearest upsample shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0],
        "2d nearest upsample"
    );
}

fn check_2d_nearest_downsample() {
    // Input [1,1,4,4] integers 0..15, downsample to [2,2].
    // src_y = floor((yi+0.5)*4/2): yi=0→1  yi=1→3
    // src_x = floor((xi+0.5)*4/2): xi=0→1  xi=1→3
    // out[0,0]=in[1,1]=5  out[0,1]=in[1,3]=7
    // out[1,0]=in[3,1]=13 out[1,1]=in[3,3]=15
    let data: Vec<f64> = (0u64..16).map(|i| i as f64).collect();
    let inp = t2d(&data, 4, 4);
    let out = interpolate_2d(&inp, 2, 2, InterpolateMode::Nearest);
    assert_eq!(out.shape(), &[1, 1, 2, 2], "2d nearest downsample shape");
    assert_eq!(
        out.as_slice(),
        &[5.0_f64, 7.0, 13.0, 15.0],
        "2d nearest downsample"
    );
}

fn check_2d_bilinear_identity() {
    // Same-size bilinear: align-half-pixel maps (yi,xi) → (yi,xi) exactly (same
    // reasoning as 1d bilinear identity).
    let inp = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2);
    let out = interpolate_2d(&inp, 2, 2, InterpolateMode::Bilinear);
    assert_eq!(out.as_slice(), inp.as_slice(), "2d bilinear identity");
}

fn check_multichannel_nearest() {
    // N=1, C=2, L=2: channels are independent, verify each is handled correctly.
    // ch0=[1,2], ch1=[3,4] → upsample to L=4 → ch0=[1,1,2,2], ch1=[3,3,4,4]
    let inp = Tensor::<f64, SequentialBackend>::from_slice(vec![1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let out = interpolate_1d(&inp, 4, InterpolateMode::Nearest);
    assert_eq!(out.shape(), &[1, 2, 4], "multichannel shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        "multichannel nearest"
    );
}

#[test]
fn interpolate_1d_nearest_upsample() {
    check_1d_nearest_upsample();
}

#[test]
fn interpolate_1d_nearest_downsample() {
    check_1d_nearest_downsample();
}

#[test]
fn interpolate_1d_nearest_identity() {
    check_1d_nearest_identity();
}

#[test]
fn interpolate_1d_bilinear_identity() {
    check_1d_bilinear_identity();
}

#[test]
fn interpolate_2d_nearest_upsample() {
    check_2d_nearest_upsample();
}

#[test]
fn interpolate_2d_nearest_downsample() {
    check_2d_nearest_downsample();
}

#[test]
fn interpolate_2d_bilinear_identity() {
    check_2d_bilinear_identity();
}

#[test]
fn interpolate_1d_multichannel_nearest() {
    check_multichannel_nearest();
}
