//! Differential parity for `conv_transpose1d` and `conv_transpose2d`.
//!
//! Reference values derived from the scatter-accumulate definition:
//!   For each (ni, ic, ti): output[ni, oc, ti*stride + ki*dilation - padding]
//!     += input[ni, ic, ti] * weight[ic, oc, ki]
//!
//! Chosen inputs use C_in=C_out=1, single-batch (N=1), so there is only one
//! weight "path" and the expected outputs are computable by hand.
//!
//! All reference values are integer-valued so assertions use `assert_eq!`.
//! SequentialBackend and MoiraiBackend must return bitwise-identical results.

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

// CONV_TRANSPOSE1D

fn check_conv_transpose1d<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // input [1,1,3] = [[[1,2,3]]], weight [1,1,2] = [[[1,1]]],
    // stride=1, padding=0, output_padding=0, dilation=1.
    // L_out = (3-1)*1 + 1*(2-1) + 0 + 1 - 0 = 4
    //
    // Scatter:
    //   ti=0,ki=0 -> out[0] += 1*1 = 1
    //   ti=0,ki=1 -> out[1] += 1*1
    //   ti=1,ki=0 -> out[1] += 2*1   => out[1] = 3
    //   ti=1,ki=1 -> out[2] += 2*1
    //   ti=2,ki=0 -> out[2] += 3*1   => out[2] = 5
    //   ti=2,ki=1 -> out[3] += 3*1 = 3
    // Expected: [1,3,5,3]
    let inp = t(&[1, 1, 3], &[1.0, 2.0, 3.0], backend);
    let wgt = t(&[1, 1, 2], &[1.0, 1.0], backend);
    let out = coeus_ops::conv_transpose1d(&inp, &wgt, None, 1, 0, 0, 1, backend);
    assert_eq!(out.shape(), &[1, 1, 4], "conv_transpose1d shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 3.0, 5.0, 3.0],
        "conv_transpose1d basic"
    );

    // weight [1,1,2] = [[[1,2]]] (asymmetric kernel).
    // ti=0,ki=0 -> out[0] += 1*1; ti=0,ki=1 -> out[1] += 1*2
    // ti=1,ki=0 -> out[1] += 2*1  => out[1]=4; ti=1,ki=1 -> out[2] += 2*2
    // ti=2,ki=0 -> out[2] += 3*1  => out[2]=7; ti=2,ki=1 -> out[3] += 3*2=6
    // Expected: [1,4,7,6]
    let wgt2 = t(&[1, 1, 2], &[1.0, 2.0], backend);
    let out2 = coeus_ops::conv_transpose1d(&inp, &wgt2, None, 1, 0, 0, 1, backend);
    assert_eq!(
        out2.as_slice(),
        &[1.0_f64, 4.0, 7.0, 6.0],
        "conv_transpose1d asymmetric kernel"
    );

    // stride=2: L_out = (3-1)*2 + 1*(2-1) + 0 + 1 = 6
    // ti=0,ki=0 -> t_out=0; ti=0,ki=1 -> t_out=1
    // ti=1,ki=0 -> t_out=2; ti=1,ki=1 -> t_out=3
    // ti=2,ki=0 -> t_out=4; ti=2,ki=1 -> t_out=5
    // weight=[1,1], input=[1,2,3]:
    // out = [1,1,2,2,3,3]
    let out3 = coeus_ops::conv_transpose1d(&inp, &wgt, None, 2, 0, 0, 1, backend);
    assert_eq!(out3.shape(), &[1, 1, 6], "conv_transpose1d stride=2 shape");
    assert_eq!(
        out3.as_slice(),
        &[1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0],
        "conv_transpose1d stride=2"
    );

    // identity weight [1,1,1] = [[[1]]]: output equals input (L_out = L).
    let eye = t(&[1, 1, 1], &[1.0], backend);
    let id = coeus_ops::conv_transpose1d(&inp, &eye, None, 1, 0, 0, 1, backend);
    assert_eq!(
        id.as_slice(),
        inp.as_slice(),
        "conv_transpose1d K=1 identity"
    );
}

// CONV_TRANSPOSE2D

fn check_conv_transpose2d<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    // input [1,1,2,2] = [[[[1,2],[3,4]]]], weight [1,1,2,2] = [[[[1,0],[0,1]]]],
    // stride=1, padding=0, output_padding=0, dilation=1.
    // H_out = W_out = (2-1)*1 + 1*(2-1) + 0 + 1 = 3.
    //
    // This is an identity-like kernel (diagonal weight).
    // The output at (rh, rw) = sum_{hi,wi,ki,kj} input[hi,wi] * weight[ki,kj]
    //   where rh = hi*stride + ki, rw = wi*stride + kj.
    //
    // With weight=[[1,0],[0,1]]:
    // ki=0,kj=0,w=1 at each (hi,wi) -> out[hi,wi] += input[hi,wi]*1
    // ki=1,kj=1,w=1 at each (hi,wi) -> out[hi+1,wi+1] += input[hi,wi]*1
    //
    // out[0,0] = input[0,0]*1 = 1
    // out[0,1] = 0  (no contributions from ki=0,kj=1 since w=0; ki=1,kj=0 would be hi+1,wi-1 out of range)
    // out[rh,rw] = sum over (hi,wi,ki,kj):
    //   input[hi,wi] * weight[ki,kj] where rh=hi+ki, rw=wi+kj, all in bounds.
    //
    // For output shape [3,3]:
    // out[0,0]: (hi=0,wi=0,ki=0,kj=0) -> input[0,0]*1 = 1
    // out[0,1]: (hi=0,wi=1,ki=0,kj=0) -> input[0,1]*1 = 2
    //           (hi=0,wi=0,ki=0,kj=1) -> input[0,0]*0 = 0
    //         = 2
    // out[0,2]: (hi=0,wi=1,ki=0,kj=1) -> 2*0 = 0 only. = 0
    // out[1,0]: (hi=1,wi=0,ki=0,kj=0) -> 3*1; (hi=0,wi=0,ki=1,kj=0) -> 1*0 = 0; = 3
    // out[1,1]: (hi=1,wi=1,ki=0,kj=0) -> 4*1; (hi=0,wi=0,ki=1,kj=1) -> 1*1 = 1;
    //           (hi=0,wi=1,ki=1,kj=0) -> 2*0=0; (hi=1,wi=0,ki=0,kj=1) -> 3*0=0;
    //         = 5
    // out[1,2]: (hi=1,wi=1,ki=0,kj=1) -> 4*0=0; (hi=0,wi=1,ki=1,kj=1) -> 2*1=2; = 2
    // out[2,0]: (hi=1,wi=0,ki=1,kj=0) -> 3*0=0; = 0
    // out[2,1]: (hi=1,wi=1,ki=1,kj=0) -> 4*0=0; (hi=1,wi=0,ki=1,kj=1) -> 3*1=3; = 3
    // out[2,2]: (hi=1,wi=1,ki=1,kj=1) -> 4*1=4; = 4
    //
    // Expected [3,3] (row-major): [1,2,0, 3,5,2, 0,3,4]
    let inp = t(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0], backend);
    let wgt = t(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0], backend);
    let out = coeus_ops::conv_transpose2d(&inp, &wgt, None, 1, 0, 0, 1, backend);
    assert_eq!(out.shape(), &[1, 1, 3, 3], "conv_transpose2d shape");
    assert_eq!(
        out.as_slice(),
        &[1.0_f64, 2.0, 0.0, 3.0, 5.0, 2.0, 0.0, 3.0, 4.0],
        "conv_transpose2d diagonal kernel"
    );

    // identity K=1x1 weight [1,1,1,1] = [[[[1]]]]: output equals input.
    let eye = t(&[1, 1, 1, 1], &[1.0], backend);
    let id = coeus_ops::conv_transpose2d(&inp, &eye, None, 1, 0, 0, 1, backend);
    assert_eq!(
        id.as_slice(),
        inp.as_slice(),
        "conv_transpose2d K=1 identity"
    );
}

// wrappers

fn check_all<B>(backend: &B)
where
    B: coeus_ops::BackendOps<f64> + Default,
    B::DeviceBuffer<f64>: CpuAddressableStorage<f64> + CpuAddressableStorageMut<f64>,
{
    check_conv_transpose1d(backend);
    check_conv_transpose2d(backend);
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
