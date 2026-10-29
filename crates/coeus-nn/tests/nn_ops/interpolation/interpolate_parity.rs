//! Differential parity for `interpolate_1d` and `interpolate_2d`.
//!
//! All reference values are IEEE-exact integers or identity assignments —
//! derived from the sampling rules in the function docs:
//!   nearest: src_idx = floor(dst_idx * src_size / dst_size)   (torch `nearest`)
//!   bilinear same-size: maps (xi) → (xi) exactly, so output equals input.
//!
//! A non-integer ratio is included because nearest and `nearest-exact` agree on
//! integer ratios and diverge otherwise, so only a fractional case pins the one
//! that is implemented.
//!
//! All assertions use `assert_eq!`.
//! SequentialBackend and MoiraiBackend must return bitwise-identical results.

use coeus_core::{
    Backend, CpuAddressableStorage, CpuAddressableStorageMut, FloatElement, MoiraiBackend,
    SequentialBackend,
};
use coeus_nn::interpolate::{interpolate_1d, interpolate_2d, ResizeMode};
use coeus_tensor::Tensor;

fn t1d<B: Backend>(vals: &[f64], l: usize, backend: &B) -> Tensor<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(vec![1, 1, l], vals, backend)
        .expect("invariant: test backend operation succeeds")
}

fn t2d<B: Backend>(vals: &[f64], h: usize, w: usize, backend: &B) -> Tensor<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(vec![1, 1, h, w], vals, backend)
        .expect("invariant: test backend operation succeeds")
}

fn check_interpolate<B: Backend>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // 1-D nearest upsample: [10,20] → [4]: [10,10,20,20]
    // src_idx = floor((xi+0.5)*2/4): xi=0→0, xi=1→0, xi=2→1, xi=3→1
    let inp1 = t1d(&[10.0, 20.0], 2, backend);
    let out1 =
        interpolate_1d(&inp1, 4, ResizeMode::Nearest).expect("invariant: test operation succeeds");
    assert_eq!(out1.shape(), &[1, 1, 4], "1d nearest upsample shape");
    assert_eq!(
        out1.as_slice(),
        &[10.0_f64, 10.0, 20.0, 20.0],
        "1d nearest upsample"
    );

    // 1-D nearest downsample: [1,2,3,4] → [2]: [1,3]
    // src_idx = floor(xi*4/2): xi=0→0, xi=1→2
    let inp2 = t1d(&[1.0, 2.0, 3.0, 4.0], 4, backend);
    let out2 =
        interpolate_1d(&inp2, 2, ResizeMode::Nearest).expect("invariant: test operation succeeds");
    assert_eq!(out2.as_slice(), &[1.0_f64, 3.0], "1d nearest downsample");

    // 1-D nearest with a non-integer ratio: [10,20,30] → [5].
    // src_idx = floor(xi*3/5): 0, 0, 1, 1, 2.
    let inp2b = t1d(&[10.0, 20.0, 30.0], 3, backend);
    let out2b =
        interpolate_1d(&inp2b, 5, ResizeMode::Nearest).expect("invariant: test operation succeeds");
    assert_eq!(
        out2b.as_slice(),
        &[10.0_f64, 10.0, 20.0, 20.0, 30.0],
        "1d nearest non-integer ratio"
    );

    // 1-D bilinear at the same non-integer ratio, ratio doubled so every
    // weight is a binary fraction: [0,4,8] → [6].
    // src = (xi+0.5)*0.5 - 0.5 = -0.25, 0.25, 0.75, 1.25, 1.75, 2.25.
    let inp2c = t1d(&[0.0, 4.0, 8.0], 3, backend);
    let out2c = interpolate_1d(&inp2c, 6, ResizeMode::Bilinear)
        .expect("invariant: test operation succeeds");
    assert_eq!(
        out2c.as_slice(),
        &[0.0_f64, 1.0, 3.0, 5.0, 7.0, 8.0],
        "1d bilinear non-integer ratio"
    );

    // 1-D nearest identity: same L.
    let inp3 = t1d(&[5.0, 3.0, 8.0], 3, backend);
    let out3 =
        interpolate_1d(&inp3, 3, ResizeMode::Nearest).expect("invariant: test operation succeeds");
    assert_eq!(out3.as_slice(), inp3.as_slice(), "1d nearest identity");

    // 1-D bilinear same-size: align-half-pixel maps xi → xi, so output = input.
    let inp4 = t1d(&[7.0, 11.0, 3.0], 3, backend);
    let out4 =
        interpolate_1d(&inp4, 3, ResizeMode::Bilinear).expect("invariant: test operation succeeds");
    assert_eq!(out4.as_slice(), inp4.as_slice(), "1d bilinear identity");

    // 2-D nearest upsample: [[1,2],[3,4]] → [4,4]
    // sy=floor((yi+0.5)*2/4), sx=floor((xi+0.5)*2/4)
    // rows 0,1: sy=0; rows 2,3: sy=1. cols 0,1: sx=0; cols 2,3: sx=1.
    let inp5 = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2, backend);
    let out5 = interpolate_2d(&inp5, 4, 4, ResizeMode::Nearest)
        .expect("invariant: test operation succeeds");
    assert_eq!(out5.shape(), &[1, 1, 4, 4], "2d nearest upsample shape");
    assert_eq!(
        out5.as_slice(),
        &[1.0_f64, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,],
        "2d nearest upsample"
    );

    // 2-D nearest downsample: 4×4 integers 0..15 → [2,2]
    // sy = floor(yi*4/2): yi=0→0, yi=1→2.
    // sx = floor(xi*4/2): xi=0→0, xi=1→2.
    // out[0,0]=in[0,0]=0, out[0,1]=in[0,2]=2, out[1,0]=in[2,0]=8, out[1,1]=in[2,2]=10
    let data5: Vec<f64> = (0u64..16).map(|i| i as f64).collect();
    let inp6 = t2d(&data5, 4, 4, backend);
    let out6 = interpolate_2d(&inp6, 2, 2, ResizeMode::Nearest)
        .expect("invariant: test operation succeeds");
    assert_eq!(out6.shape(), &[1, 1, 2, 2], "2d nearest downsample shape");
    assert_eq!(
        out6.as_slice(),
        &[0.0_f64, 2.0, 8.0, 10.0],
        "2d nearest downsample"
    );

    // 2-D nearest with a non-integer ratio: 5×5 integers 0..24 → [2,2].
    // floor(yi*5/2): 0, 2 — the same for x.
    let data6: Vec<f64> = (0usize..25).map(f64::from_count).collect();
    let inp8 = t2d(&data6, 5, 5, backend);
    let out8 = interpolate_2d(&inp8, 2, 2, ResizeMode::Nearest)
        .expect("invariant: test operation succeeds");
    assert_eq!(
        out8.as_slice(),
        &[0.0_f64, 2.0, 10.0, 12.0],
        "2d nearest non-integer ratio"
    );

    // 2-D bilinear same-size: output = input.
    let inp7 = t2d(&[10.0, 20.0, 30.0, 40.0], 2, 2, backend);
    let out7 = interpolate_2d(&inp7, 2, 2, ResizeMode::Bilinear)
        .expect("invariant: test operation succeeds");
    assert_eq!(out7.as_slice(), inp7.as_slice(), "2d bilinear identity");
}

#[test]
fn sequential_interpolate_match_reference() {
    check_interpolate(&SequentialBackend);
}

#[test]
fn moirai_interpolate_match_reference() {
    check_interpolate(&MoiraiBackend);
}
