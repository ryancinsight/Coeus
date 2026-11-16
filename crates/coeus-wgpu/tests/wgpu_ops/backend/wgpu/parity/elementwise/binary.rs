//! Parity of the binary element-wise operations.

use coeus_tensor::Tensor;

use super::super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_parity_add() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0i16..16).map(f32::from).collect::<Vec<_>>())
        .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0i16..16)
            .map(|x| f32::from(x) * 0.5 - 4.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: test backend operation succeeds");
    let cpu = coeus_ops::add(&a, &b, &s).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::add(&to_gpu(&a), &to_gpu(&b), &wgpu())
            .expect("invariant: test operation succeeds"),
    );
    assert_parity("add", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_sub() {
    let s = seq();
    let a = Tensor::from_slice(vec![4, 4], &(0i16..16).map(f32::from).collect::<Vec<_>>())
        .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0i16..16).map(|x| f32::from(x) * 0.5).collect::<Vec<_>>(),
    )
    .expect("invariant: test backend operation succeeds");
    let cpu = coeus_ops::sub(&a, &b, &s).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::sub(&to_gpu(&a), &to_gpu(&b), &wgpu())
            .expect("invariant: test operation succeeds"),
    );
    assert_parity("sub", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_mul() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0i16..16)
            .map(|x| f32::from(x) * 0.1 + 0.5)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0i16..16)
            .map(|x| f32::from(x) * 0.2 - 1.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: test backend operation succeeds");
    let cpu = coeus_ops::mul(&a, &b, &s).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::mul(&to_gpu(&a), &to_gpu(&b), &wgpu())
            .expect("invariant: test operation succeeds"),
    );
    assert_parity("mul", cpu.as_slice(), gpu.as_slice());
}

#[test]
fn test_wgpu_parity_div() {
    let s = seq();
    let a = Tensor::from_slice(
        vec![4, 4],
        &(0i16..16)
            .map(|x| (f32::from(x) + 1.0) * 0.5)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0i16..16)
            .map(|x| (f32::from(x) + 1.0) * 0.25)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: test backend operation succeeds");
    let cpu = coeus_ops::div(&a, &b, &s).expect("invariant: test operation succeeds");
    let gpu = to_cpu(
        &coeus_ops::div(&to_gpu(&a), &to_gpu(&b), &wgpu())
            .expect("invariant: test operation succeeds"),
    );
    assert_parity("div", cpu.as_slice(), gpu.as_slice());
}
