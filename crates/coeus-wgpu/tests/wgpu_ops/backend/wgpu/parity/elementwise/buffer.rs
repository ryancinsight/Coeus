//! Ownership of the output buffers the backend reuses.

use coeus_core::ComputeBackend;
use coeus_ops::ElementwiseOps;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

use super::super::{assert_parity, seq, to_cpu, to_gpu, wgpu};

#[test]
fn test_wgpu_hephaestus_contiguous_binary_reuses_output_buffer() {
    let s = seq();
    let w = wgpu();
    let a = Tensor::from_slice(vec![4, 4], &(0i16..16).map(f32::from).collect::<Vec<_>>())
        .expect("invariant: test backend operation succeeds");
    let b = Tensor::from_slice(
        vec![4, 4],
        &(0i16..16)
            .map(|x| f32::from(x) * 0.5 - 4.0)
            .collect::<Vec<_>>(),
    )
    .expect("invariant: test backend operation succeeds");
    let a_gpu = to_gpu(&a);
    let b_gpu = to_gpu(&b);
    let mut out_gpu = Tensor::<f32, WgpuBackend>::zeros_on(vec![4, 4], &w)
        .expect("invariant: test backend operation succeeds");
    let out_layout = out_gpu.layout().clone();
    let allocation_id = out_gpu.storage().allocation_id();

    w.elementwise_binary(
        coeus_ops::BinaryOp::Add,
        a_gpu.storage(),
        a_gpu.layout(),
        b_gpu.storage(),
        b_gpu.layout(),
        out_gpu
            .storage_mut()
            .expect("invariant: test backend operation succeeds"),
        &out_layout,
    )
    .expect("valid WGPU addition output buffer");

    assert!(
        out_gpu.storage().allocation_id() == allocation_id,
        "delegated binary path reallocated output buffer"
    );

    let expected = coeus_ops::add(&a, &b, &s).expect("invariant: test operation succeeds");
    let got = to_cpu(&out_gpu);
    assert_parity(
        "hephaestus_binary_into_add",
        expected.as_slice(),
        got.as_slice(),
    );
}

#[test]
fn test_wgpu_hephaestus_contiguous_unary_reuses_output_buffer() {
    let s = seq();
    let w = wgpu();
    let x = Tensor::from_slice(vec![8], &[-4.0f32, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let x_gpu = to_gpu(&x);
    let mut out_gpu = Tensor::<f32, WgpuBackend>::zeros_on(vec![8], &w)
        .expect("invariant: test backend operation succeeds");
    let out_layout = out_gpu.layout().clone();
    let allocation_id = out_gpu.storage().allocation_id();

    w.elementwise_unary(
        coeus_ops::UnaryOp::Recip,
        x_gpu.storage(),
        x_gpu.layout(),
        out_gpu
            .storage_mut()
            .expect("invariant: test backend operation succeeds"),
        &out_layout,
    )
    .expect("valid WGPU reciprocal output buffer");

    assert!(
        out_gpu.storage().allocation_id() == allocation_id,
        "delegated unary path reallocated output buffer"
    );

    let expected = coeus_ops::recip(&x, &s).expect("invariant: test operation succeeds");
    let got = to_cpu(&out_gpu);
    assert_parity(
        "hephaestus_unary_into_recip",
        expected.as_slice(),
        got.as_slice(),
    );
}

#[test]
fn test_wgpu_neg_preserves_output_clones() {
    let w = wgpu();
    let values = [-4.0_f32, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    let input = to_gpu(
        &Tensor::from_slice([values.len()], &values)
            .expect("invariant: test backend operation succeeds"),
    );
    let mut output = input.storage().clone();

    w.elementwise_unary(
        coeus_ops::UnaryOp::Neg,
        input.storage(),
        input.layout(),
        &mut output,
        input.layout(),
    )
    .expect("negation detaches the shared destination");

    let mut actual = [0.0; 8];
    w.copy_to_host(&output, &mut actual)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(actual, values.map(|value| -value));
    assert_eq!(to_cpu(&input).as_slice(), &values);
}

#[test]
fn test_wgpu_elu_preserves_output_clones() {
    let w = wgpu();
    let values = [-2.0_f32, -0.5, 0.5, 2.0];
    let input = to_gpu(
        &Tensor::from_slice([2, 2], &values).expect("invariant: test backend operation succeeds"),
    );
    // ELU evaluates each element independently: changing only destination
    // ownership and traversal layout must preserve the provider's exact result.
    // The separate sequential parity cases cover the exponential approximation.
    let expected = to_cpu(&coeus_ops::elu(&input, &w).expect("invariant: test operation succeeds"));
    let transposed = input.t();

    for view in [&input, &transposed] {
        let mut output = view.storage().clone();
        w.elementwise_unary(
            coeus_ops::UnaryOp::Elu,
            view.storage(),
            view.layout(),
            &mut output,
            view.layout(),
        )
        .expect("ELU detaches dense and transposed destinations");

        // Download the entire backing allocation: a transposed output must
        // write the same physical elements without compacting its layout.
        let mut actual = [0.0; 4];
        w.copy_to_host(&output, &mut actual)
            .expect("invariant: test backend operation succeeds");
        assert_eq!(actual.as_slice(), expected.as_slice());
        assert_eq!(to_cpu(&input).as_slice(), &values);
        assert_eq!(to_cpu(&transposed).as_slice(), &[-2.0, 0.5, -0.5, 2.0]);
    }
}

#[test]
fn test_wgpu_add_preserves_output_clones() {
    let w = wgpu();
    let a_data: Vec<f32> = (0i16..16).map(|x| f32::from(x) * 0.25 - 2.0).collect();
    let b_data: Vec<f32> = (0i16..16).map(|x| f32::from(x) * 0.1 + 0.5).collect();
    let expected: Vec<_> = a_data.iter().zip(&b_data).map(|(a, b)| a + b).collect();
    let a_gpu = to_gpu(
        &Tensor::from_slice([4, 4], &a_data).expect("invariant: test backend operation succeeds"),
    );
    let b_gpu = to_gpu(
        &Tensor::from_slice([4, 4], &b_data).expect("invariant: test backend operation succeeds"),
    );
    let mut output = a_gpu.storage().clone();

    w.elementwise_binary(
        coeus_ops::BinaryOp::Add,
        a_gpu.storage(),
        a_gpu.layout(),
        b_gpu.storage(),
        b_gpu.layout(),
        &mut output,
        a_gpu.layout(),
    )
    .expect("addition detaches the shared destination");

    let mut actual = [0.0; 16];
    w.copy_to_host(&output, &mut actual)
        .expect("invariant: test backend operation succeeds");
    assert_eq!(actual.as_slice(), expected.as_slice());
    assert_eq!(to_cpu(&a_gpu).as_slice(), a_data.as_slice());
    assert_eq!(to_cpu(&b_gpu).as_slice(), b_data.as_slice());
}
