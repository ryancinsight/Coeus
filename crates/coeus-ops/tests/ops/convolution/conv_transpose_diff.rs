//! Differential parity for `conv_transpose1d`, `conv_transpose2d`, and
//! `conv_transpose3d`.
//!
//! Reference values are derived from the scatter-accumulate definition:
//! `out[n, oc, p * stride + k * dilation - padding] += input[n, ic, p] *
//! weight[ic, oc, k]`, lifted to the spatial rank. The selected fixtures use
//! `C_in = C_out = 1` and `N = 1`, so the expected values are hand-auditable.

use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_tensor::Tensor;

fn t<B>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend)
}

fn check_conv_transpose1d<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::Error: std::fmt::Debug,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    let inp = t(&[1, 1, 3], &[1.0, 2.0, 3.0], backend);
    let wgt = t(&[1, 1, 2], &[1.0, 1.0], backend);
    let out = coeus_ops::conv_transpose1d(&inp, &wgt, None, 1, 0, 0, 1, backend)
        .expect("rank-one transposed convolution must succeed");
    assert_eq!(out.shape(), &[1, 1, 4], "conv_transpose1d shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 3.0, 5.0, 3.0],
        "conv_transpose1d all-ones kernel"
    );

    let wgt2 = t(&[1, 1, 2], &[1.0, 2.0], backend);
    let out2 = coeus_ops::conv_transpose1d(&inp, &wgt2, None, 1, 0, 0, 1, backend)
        .expect("asymmetric rank-one transposed convolution must succeed");
    assert_eq!(
        out2.as_slice(),
        &[1.0_f64, 4.0, 7.0, 6.0],
        "conv_transpose1d asymmetric kernel"
    );

    let out3 = coeus_ops::conv_transpose1d(&inp, &wgt, None, 2, 0, 0, 1, backend)
        .expect("strided rank-one transposed convolution must succeed");
    assert_eq!(out3.shape(), &[1, 1, 6], "conv_transpose1d stride=2 shape");
    assert_eq!(
        out3.as_slice(),
        &[1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0],
        "conv_transpose1d stride=2"
    );

    let eye = t(&[1, 1, 1], &[1.0], backend);
    let id = coeus_ops::conv_transpose1d(&inp, &eye, None, 1, 0, 0, 1, backend)
        .expect("identity rank-one transposed convolution must succeed");
    assert_eq!(
        id.as_slice(),
        inp.as_slice(),
        "conv_transpose1d K=1 identity"
    );
}

fn check_conv_transpose2d<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::Error: std::fmt::Debug,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    let inp = t(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let wgt = t(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0], backend);
    let out = coeus_ops::conv_transpose2d(&inp, &wgt, None, 1, 0, 0, 1, backend)
        .expect("rank-two transposed convolution must succeed");
    assert_eq!(out.shape(), &[1, 1, 3, 3], "conv_transpose2d shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0],
        "conv_transpose2d diagonal kernel"
    );

    let eye = t(&[1, 1, 1, 1], &[1.0], backend);
    let id = coeus_ops::conv_transpose2d(&inp, &eye, None, 1, 0, 0, 1, backend)
        .expect("identity rank-two transposed convolution must succeed");
    assert_eq!(
        id.as_slice(),
        inp.as_slice(),
        "conv_transpose2d K=1 identity"
    );
}

fn check_conv_transpose3d<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_ops::CpuBackend + Default,
    B::Error: std::fmt::Debug,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    let inp = t(
        &[1, 1, 2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        backend,
    );
    let wgt = t(&[1, 1, 2, 2, 2], &[1.0; 8], backend);
    let out = coeus_ops::conv_transpose3d(&inp, &wgt, None, 1, 0, 0, 1, backend)
        .expect("rank-three transposed convolution must succeed");
    assert_eq!(out.shape(), &[1, 1, 3, 3, 3], "conv_transpose3d shape");
    assert_eq!(
        out.as_slice(),
        &[
            1.0_f64, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0, 6.0, 14.0, 8.0, 16.0, 36.0, 20.0,
            10.0, 22.0, 12.0, 5.0, 11.0, 6.0, 12.0, 26.0, 14.0, 7.0, 15.0, 8.0,
        ],
        "conv_transpose3d all-ones 2x2x2 kernel"
    );

    let eye = t(&[1, 1, 1, 1, 1], &[1.0], backend);
    let id = coeus_ops::conv_transpose3d(&inp, &eye, None, 1, 0, 0, 1, backend)
        .expect("identity rank-three transposed convolution must succeed");
    assert_eq!(
        id.as_slice(),
        inp.as_slice(),
        "conv_transpose3d K=1 identity"
    );
}

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + coeus_ops::CpuBackend + Default,
    B::Error: std::fmt::Debug,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_conv_transpose1d(backend);
    check_conv_transpose2d(backend);
    check_conv_transpose3d(backend);
}

#[test]
fn sequential_conv_transpose_match_reference() {
    let backend = SequentialBackend;
    check_all(&backend);
}

#[test]
fn moirai_conv_transpose_match_reference() {
    let backend = MoiraiBackend;
    check_all(&backend);
}
