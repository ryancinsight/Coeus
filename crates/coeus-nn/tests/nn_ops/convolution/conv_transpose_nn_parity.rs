//! Differential parity for the `ConvTranspose1d` and `ConvTranspose2d` *Module*
//! structs in `coeus-nn`.
//!
//! The corresponding free-function ops are already covered in
//! `crates/coeus-ops/tests/conv_transpose_diff.rs`.  This file verifies that the
//! Module wrapper routes the same computation on both backends and that the
//! `output_len` / `output_size` helpers agree with the observed output shape.
//!
//! Reference values reuse the oracles from the ops-level test (scatter-accumulate):
//!   input [1,1,3]=[1,2,3], weight [1,1,2]=[1,1] → [1,3,5,3]
//!   input [1,1,2,2]=[[1,2],[3,4]], diagonal weight → [1,2,0,3,5,2,0,3,4]
//!
//! Weight fields are `pub` so we bypass the random kaiming init by constructing
//! the struct directly with known values.
//!
//! All assertions use `assert_eq!`.
//! SequentialBackend and MoiraiBackend must produce bitwise-identical results.

use coeus_autograd::Var;
use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, MoiraiBackend, SequentialBackend,
};
use coeus_nn::{ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Module};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

fn t<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Tensor<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Tensor::from_slice_on(shape.to_vec(), vals, backend).expect("construct tensor")
}

fn v<B: BackendOps<f64> + Default>(shape: &[usize], vals: &[f64], backend: &B) -> Var<f64, B>
where
    B::DeviceBuffer<f64>: CpuAddressableStorageMut<f64>,
{
    Var::new(t(shape, vals, backend), false).expect("construct variable")
}

fn check_conv_transpose1d<B: BackendOps<f64> + coeus_ops::CpuBackend + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Build ConvTranspose1d with weight=[1,1] (ones), no bias.
    // in_channels=1, out_channels=1, kernel_size=2, stride=1.
    // L_out = (3-1)*1 + 1*(2-1) + 1 = 4
    //
    // scatter-accumulate (same oracle as conv_transpose_diff.rs):
    //   ti=0 → out[0]+=1, out[1]+=1
    //   ti=1 → out[1]+=2, out[2]+=2
    //   ti=2 → out[2]+=3, out[3]+=3
    // Expected: [1,3,5,3]
    let ct1 = ConvTranspose1d::<f64, B> {
        weight: Var::new(t(&[1, 1, 2], &[1.0, 1.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };

    let inp = v(&[1, 1, 3], &[1.0, 2.0, 3.0], backend);
    assert_eq!(ct1.output_len(3), 4, "ConvTranspose1d output_len");

    let out = Module::<f64, B>::forward(&ct1, &inp).expect("run forward");
    assert_eq!(
        out.tensor.shape(),
        &[1, 1, 4],
        "ConvTranspose1d output shape"
    );
    assert_eq!(
        out.tensor.as_slice(),
        &[1.0_f64, 3.0, 5.0, 3.0],
        "ConvTranspose1d [1,1] weight → [1,3,5,3]"
    );

    // Identity kernel K=1: output equals input.
    let ct_id = ConvTranspose1d::<f64, B> {
        weight: Var::new(t(&[1, 1, 1], &[1.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };
    let out_id = Module::<f64, B>::forward(&ct_id, &inp).expect("run forward");
    assert_eq!(
        out_id.tensor.as_slice(),
        inp.tensor.as_slice(),
        "ConvTranspose1d K=1 identity"
    );

    // Stride=2: L_out = (3-1)*2 + 1*(2-1) + 1 = 6
    // Each input is separated by a stride gap; weight=[1,1]:
    // out = [1,1, 2,2, 3,3]
    let ct_s2 = ConvTranspose1d::<f64, B> {
        weight: Var::new(t(&[1, 1, 2], &[1.0, 1.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 2,
        stride: 2,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };
    assert_eq!(
        ct_s2.output_len(3),
        6,
        "ConvTranspose1d stride=2 output_len"
    );
    let out_s2 = Module::<f64, B>::forward(&ct_s2, &inp).expect("run forward");
    assert_eq!(
        out_s2.tensor.shape(),
        &[1, 1, 6],
        "ConvTranspose1d stride=2 shape"
    );
    assert_eq!(
        out_s2.tensor.as_slice(),
        &[1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0],
        "ConvTranspose1d stride=2 → [1,1,2,2,3,3]"
    );
}

fn check_conv_transpose2d<B: BackendOps<f64> + coeus_ops::CpuBackend + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Weight [1,1,2,2] = diagonal [[1,0],[0,1]], input [1,1,2,2]=[[1,2],[3,4]].
    // H_out = W_out = (2-1)*1 + 1*(2-1) + 1 = 3.
    // Expected [3,3] (row-major): [1,2,0, 3,5,2, 0,3,4]
    // (same oracle as conv_transpose_diff.rs)
    let ct2 = ConvTranspose2d::<f64, B> {
        weight: Var::new(t(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 2,
        stride: 1,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };

    let inp = v(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let out = Module::<f64, B>::forward(&ct2, &inp).expect("run forward");
    assert_eq!(out.tensor.shape(), &[1, 1, 3, 3], "ConvTranspose2d shape");
    assert_eq!(
        out.tensor.as_slice(),
        &[1.0_f64, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0],
        "ConvTranspose2d diagonal kernel"
    );

    // Identity K=1×1: output equals input.
    let ct2_id = ConvTranspose2d::<f64, B> {
        weight: Var::new(t(&[1, 1, 1, 1], &[1.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };
    let out_id = Module::<f64, B>::forward(&ct2_id, &inp).expect("run forward");
    assert_eq!(
        out_id.tensor.as_slice(),
        inp.tensor.as_slice(),
        "ConvTranspose2d K=1 identity"
    );
}

fn check_all<B: BackendOps<f64> + coeus_ops::CpuBackend + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_conv_transpose1d(backend);
    check_conv_transpose2d(backend);
    check_conv_transpose3d(backend);
}

fn check_conv_transpose3d<B: BackendOps<f64> + coeus_ops::CpuBackend + Default>(backend: &B)
where
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // Weight [1,1,1,1,1] = [1.0] (identity impulse). Input [1,1,2,2,2] = [1..8].
    // D=H=W=2 → D_out=H_out=W_out=2 (stride=1, padding=0).
    // output_dims() agrees with the public formula: 2 = (2-1)*1 + 1*(1-1) + 0 + 1.
    let ct3 = ConvTranspose3d::<f64, B> {
        weight: Var::new(t(&[1, 1, 1, 1, 1], &[1.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };
    assert_eq!(
        ct3.output_dims(2, 2, 2),
        (2, 2, 2),
        "ConvTranspose3d::output_dims(K=1 stride=1 padding=0)"
    );
    let inp = v(
        &[1, 1, 2, 2, 2],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        backend,
    );
    let out = Module::<f64, B>::forward(&ct3, &inp).expect("run forward");
    assert_eq!(
        out.tensor.shape(),
        &[1, 1, 2, 2, 2],
        "ConvTranspose3d identity shape"
    );
    assert_eq!(
        out.tensor.as_slice(),
        inp.tensor.as_slice(),
        "ConvTranspose3d K=1 identity"
    );

    // Scale K=1, weight=[2.0]: output = 2 × input. output_dims unchanged.
    let ct3_scale = ConvTranspose3d::<f64, B> {
        weight: Var::new(t(&[1, 1, 1, 1, 1], &[2.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 1,
        stride: 1,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };
    let out_s = Module::<f64, B>::forward(&ct3_scale, &inp).expect("run forward");
    assert_eq!(
        out_s.tensor.as_slice(),
        &[2.0_f64, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        "ConvTranspose3d K=1 weight=2"
    );

    // Stride=2 K=1 identity: each axis doubles.
    // D_out = (2-1)*2 + 1*(1-1) + 0 + 1 = 3.
    let ct3_s2 = ConvTranspose3d::<f64, B> {
        weight: Var::new(t(&[1, 1, 1, 1, 1], &[1.0], backend), false).expect("construct variable"),
        bias: None,
        in_channels: 1,
        out_channels: 1,
        kernel_size: 1,
        stride: 2,
        padding: 0,
        output_padding: 0,
        dilation: 1,
    };
    assert_eq!(ct3_s2.output_dims(2, 2, 2), (3, 3, 3), "stride=2 D_out");
    let out_s2 = Module::<f64, B>::forward(&ct3_s2, &inp).expect("run forward");
    assert_eq!(
        out_s2.tensor.shape(),
        &[1, 1, 3, 3, 3],
        "ConvTranspose3d stride=2 shape"
    );
    assert_eq!(
        out_s2.tensor.as_slice(),
        // Even (d,h,w) positions get the input value at (d/2,h/2,w/2); odd positions are 0.
        // Row-major with W fastest, then H, then D:
        &[
            1.0_f64, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, // d=0
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // d=1: all zeros
            5.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0, 8.0, // d=2
        ],
        "ConvTranspose3d stride=2 K=1"
    );
}

#[test]
fn sequential_conv_transpose3d_nn_match_reference() {
    check_conv_transpose3d(&SequentialBackend);
}

#[test]
fn moirai_conv_transpose3d_nn_match_reference() {
    check_conv_transpose3d(&MoiraiBackend);
}

#[test]
fn sequential_conv_transpose_nn_match_reference() {
    check_all(&SequentialBackend);
}

#[test]
fn moirai_conv_transpose_nn_match_reference() {
    check_all(&MoiraiBackend);
}
